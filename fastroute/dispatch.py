"""Ride dispatch: deciding which driver takes which ride.

This is the decision routing exists for. Every 30 seconds the dispatcher
looks at the riders waiting and the drivers free, prices every driver against
every rider with one many-to-many Contraction Hierarchies query, and decides
who goes where. Three policies are compared on exactly the same day of
requests and the same starting fleet:

* **Nearest driver.** Riders, oldest first, each take the closest free
  driver. Simple and common, but an early rider can take the one driver a
  later rider had no alternative to.
* **Optimal matching.** The Hungarian algorithm assigns the whole batch at
  once to minimise total pickup time.
  Each rider's cost is discounted by how long they have already waited, so
  riders who are awkward to reach are not left until they give up.
* **Optimal matching + repositioning.** As above, and every two minutes
  drivers who have been idle for a while move towards neighbourhoods where
  expected demand over the next 15 minutes most exceeds the drivers already
  there. Two versions differ only in where the expectation comes from: the
  historical average for that square and time, or the ML forecast, which
  also reads what has happened so far today. The gap between them is what
  the model is worth.

Two more policies build on the last one, to answer product questions rather
than algorithmic ones:

* **Shared rides.** Riders waiting in the same batch whose trips line up
  (pickups within 1.2 km, heading the same way) share a car when the
  combined route saves at least two minutes of driving and neither rider
  is delayed by more than 40% of their own trip, or six minutes. Every
  pickup and drop order is priced and the best feasible one kept; the
  pairs that save most are taken first.
* **Surge pricing.** Every two minutes each neighbourhood gets a price
  multiplier from how far waiting and forecast riders outnumber nearby
  free drivers. Riders who see a higher price may decline: each accepts
  with probability ``exp(-elasticity x (multiplier - 1))``. Idle drivers are
  drawn towards surging areas. Surge here only rations riders and moves
  drivers; it does not bring extra drivers online, so the comparison shows
  the rationing effect on its own.

A ride is cancelled if no driver is assigned within ten minutes, and no
driver is sent on a pickup longer than twelve. Travel times come from the
router at the simulated time of day.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .ch import ContractionHierarchy
from .demand import (
    SLOT_MIN,
    SLOTS_PER_DAY,
    DemandDay,
    DemandWorld,
    _neighbour_sums,
    forecast_features,
)
from .eta import predict_from_json
from .graph import haversine_m
from .matching import greedy_matches, optimal_matches

POLICIES = (
    "nearest",
    "optimal",
    "optimal+history",
    "optimal+forecast",
    "optimal+pool",
    "optimal+surge",
)
POLICY_NAMES = {
    "nearest": "Nearest driver",
    "optimal": "Optimal matching",
    "optimal+history": "Optimal + repositioning on historical averages",
    "optimal+forecast": "Optimal + repositioning on the ML forecast",
    "optimal+pool": "+ shared rides",
    "optimal+surge": "+ surge pricing",
}
FORECAST_POLICIES = ("optimal+forecast", "optimal+pool", "optimal+surge")
# Seconds of pickup a rider's every second of waiting so far is worth: keeps
# optimal matching from starving riders who are awkward to reach.
WAIT_PRIORITY = 1.0


@dataclass(frozen=True)
class DispatchConfig:
    drivers: int = 115
    start_hour: float = 7.0
    hours: float = 3.0
    warmup_min: float = 20.0
    batch_s: float = 30.0
    max_pickup_s: float = 12 * 60
    patience_s: float = 10 * 60
    reposition_every_s: float = 120.0
    idle_before_reposition_s: float = 120.0
    max_reposition_s: float = 8 * 60
    min_trip_km: float = 1.0
    # Shared rides.
    pool_radius_km: float = 1.2
    max_detour_frac: float = 0.4
    max_extra_s: float = 6 * 60
    min_saving_s: float = 120.0
    # Surge pricing.
    surge_elasticity: float = 1.0
    surge_threshold: float = 0.5  # riders per free driver nearby before prices rise
    surge_sensitivity: float = 1.0
    surge_cap: float = 2.5


@dataclass
class Ride:
    created: float
    origin: int
    dest: int
    trip_s: float
    pickup_s: float = math.nan
    picked_up: float = math.nan
    assigned_at: float = math.nan
    cancelled: bool = False
    assigned: bool = False
    shared: bool = False
    onboard_s: float = math.nan
    priced_out: bool = False
    multiplier: float = 1.0


@dataclass
class Driver:
    node: int
    free_at: float = 0.0
    idle_since: float = 0.0
    busy_s: float = 0.0
    reposition_s: float = 0.0
    heading_to_cell: int = -1
    # The leg (index into legs) of a solo trip in progress, which a shared
    # ride could still join; -1 otherwise.
    trip_leg: int = -1


LEG_KINDS = ("pickup", "trip", "reposition", "shared")


@dataclass
class DispatchResult:
    policy: str
    metrics: dict[str, float]
    rides: list[Ride] = field(default_factory=list)
    start_nodes: list[int] = field(default_factory=list)
    # (driver, kind index into LEG_KINDS, start s, from node, to node, seconds, ride or -1)
    legs: list[tuple[int, int, float, int, int, float, int]] = field(default_factory=list)
    # Surge multipliers over time: (seconds, {square: multiplier > 1}).
    surge: list[tuple[float, dict[int, float]]] = field(default_factory=list)


def generate_rides(
    day: DemandDay,
    ch: ContractionHierarchy,
    config: DispatchConfig,
    rng: np.random.Generator,
) -> list[Ride]:
    """A day's ride requests over the simulated window (Poisson arrivals)."""
    net = day.world.network
    rides: list[Ride] = []
    t = 0.0
    end = config.hours * 3600
    while True:
        hour = config.start_hour + t / 3600
        rate = day.rate_per_min(hour) / 60
        t += rng.exponential(1 / max(rate, 1e-9))
        if t >= end:
            break
        hour = config.start_hour + t / 3600
        w_o, w_d = day.weights(hour, "origin"), day.weights(hour, "dest")
        for _ in range(10):
            o = int(rng.choice(net.num_nodes, p=w_o))
            d = int(rng.choice(net.num_nodes, p=w_d))
            km = haversine_m(net.lat[o], net.lon[o], net.lat[d], net.lon[d]) / 1000
            if km >= config.min_trip_km:
                break
        trip = ch.query(o, d)
        if trip.found:
            rides.append(Ride(t, o, d, trip.seconds))
    return rides


class Forecast:
    """Next-15-minute requests per square.

    With a model: predicted from what has been seen so far today. With
    ``history`` instead: the historical average for that square and time.
    """

    def __init__(
        self,
        world: DemandWorld,
        model_json: dict[str, object] | None,
        day: DemandDay,
        history: np.ndarray | None = None,
    ) -> None:
        self.grid = world.grid
        self.model = model_json
        self.history = history
        self.weekend = day.weekend
        self.counts = np.zeros((SLOTS_PER_DAY, self.grid.num_cells))
        self.cells = self.grid.active

    def observe(self, hour: float, origin: int) -> None:
        slot = int(hour * 60 // SLOT_MIN) % SLOTS_PER_DAY
        self.counts[slot, self.grid.cell[origin]] += 1

    def next_slot(self, hour: float) -> np.ndarray:
        """Expected requests per square in the slot starting at ``hour``."""
        slot = int(hour * 60 // SLOT_MIN) % SLOTS_PER_DAY
        if self.history is not None:
            return self.history[int(self.weekend), slot].copy()
        X = forecast_features(self.grid, self.counts, slot, self.weekend, self.cells)
        out = np.zeros(self.grid.num_cells)
        out[self.cells] = [math.exp(predict_from_json(self.model, x)) for x in X]
        return out


def simulate(
    world: DemandWorld,
    day: DemandDay,
    rides: list[Ride],
    ch: ContractionHierarchy,
    policy: str,
    config: DispatchConfig,
    *,
    forecast_model: dict[str, object] | None = None,
    seed: int = 0,
    prefill: np.ndarray | None = None,
    history: np.ndarray | None = None,
) -> DispatchResult:
    """Play one policy over a day's rides; returns metrics on rides after warm-up."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    rides = [Ride(r.created, r.origin, r.dest, r.trip_s) for r in rides]  # fresh copies
    rng = np.random.default_rng(seed)
    net = world.network
    drivers = [Driver(int(n)) for n in rng.choice(net.num_nodes, size=config.drivers)]
    start_nodes = [d.node for d in drivers]
    legs: list[tuple[int, int, float, int, int, float, int]] = []
    grid = world.grid
    use_forecast = policy in (*FORECAST_POLICIES, "optimal+history")
    pool = policy == "optimal+pool"
    surge = policy == "optimal+surge"
    if policy in FORECAST_POLICIES and forecast_model is None:
        raise ValueError("forecast repositioning needs a forecast model")
    if policy == "optimal+history" and history is None:
        raise ValueError("historical repositioning needs historical averages")
    forecast = (
        Forecast(world, forecast_model, day, history if policy == "optimal+history" else None)
        if use_forecast
        else None
    )
    if forecast is not None and prefill is not None:
        forecast.counts[:] = prefill  # the day's requests before the simulated window
    cell_targets = [int(grid.centre_node[c]) for c in grid.active]
    cell_buckets = ch.target_buckets(cell_targets) if use_forecast else None

    end = config.hours * 3600
    next_ride = 0
    waiting: list[int] = []
    t = 0.0
    # Each rider's willingness to pay, drawn once so it is the same under every policy.
    willing = np.random.default_rng((seed, 99)).random(len(rides))
    multiplier = np.ones(grid.num_cells)
    surge_frames: list[tuple[float, dict[int, float]]] = []
    trip_cache: dict[tuple[int, int], float] = {}
    path_cache: dict[tuple[int, int], list[int]] = {}
    while t <= end or waiting:
        hour = config.start_hour + t / 3600
        while next_ride < len(rides) and rides[next_ride].created <= t:
            ride = rides[next_ride]
            if forecast is not None:
                forecast.observe(config.start_hour + ride.created / 3600, ride.origin)
            if surge:
                ride.multiplier = float(multiplier[grid.cell[ride.origin]])
                accept = math.exp(-config.surge_elasticity * (ride.multiplier - 1.0))
                if willing[next_ride] >= accept:
                    ride.priced_out = True
                    next_ride += 1
                    continue
            waiting.append(next_ride)
            next_ride += 1
        for r in waiting:
            if t - rides[r].created > config.patience_s:
                rides[r].cancelled = True
        waiting = [r for r in waiting if not rides[r].cancelled]

        if pool and waiting:
            joined = _join_en_route(
                waiting, rides, drivers, t, legs, net, config,
                lambda u, v: _time(ch, trip_cache, u, v), lambda u, v: _path(ch, path_cache, u, v),
            )  # fmt: skip
            waiting = [r for r in waiting if r not in joined]
        idle = [i for i, d in enumerate(drivers) if d.free_at <= t]
        if idle and waiting:
            if pool:
                jobs = _pool_jobs(
                    waiting, rides, net, config, lambda u, v: _time(ch, trip_cache, u, v)
                )
            else:
                jobs = [_single(r, rides[r]) for r in waiting]
            cost = ch.many_to_many([drivers[i].node for i in idle], [j[0][0][0] for j in jobs])
            if policy == "nearest":
                pairs = greedy_matches(cost, config.max_pickup_s)
            else:
                waited = np.array([max(t - rides[r].created for r in _riders(j)) for j in jobs])
                priority = cost - WAIT_PRIORITY * waited[None, :]
                allowed = np.where(cost <= config.max_pickup_s, priority, np.inf)
                pairs = optimal_matches(allowed, limit=np.inf)
            matched = set()
            for i, j in pairs:
                _assign(drivers[idle[i]], idle[i], jobs[j], float(cost[i, j]), t, rides, legs)
                matched.update(_riders(jobs[j]))
            waiting = [r for r in waiting if r not in matched]

        if forecast is not None and t % config.reposition_every_s < config.batch_s:
            weight = None
            if surge:
                multiplier = _surge_multipliers(
                    grid, waiting, rides, drivers, t, hour, forecast, config
                )
                surge_frames.append(
                    (
                        t,
                        {
                            int(c): float(m)
                            for c in np.flatnonzero(multiplier > 1)
                            for m in [multiplier[c]]
                        },
                    )
                )
                weight = multiplier
            _reposition(
                drivers, t, hour, forecast, ch, cell_targets, cell_buckets, config, legs, weight
            )

        t += config.batch_s
        if t > end + config.patience_s:
            break

    return DispatchResult(
        policy, _metrics(rides, drivers, config), rides, start_nodes, legs, surge_frames
    )


# A job is one car's work: stops in order as (node, ride, is_pickup), and the
# drive between consecutive stops.
Job = tuple[list[tuple[int, int, bool]], list[float]]


def _single(index: int, ride: Ride) -> Job:
    return [(ride.origin, index, True), (ride.dest, index, False)], [ride.trip_s]


def _riders(job: Job) -> list[int]:
    return [r for _, r, pick in job[0] if pick]


def _time(ch: ContractionHierarchy, cache: dict[tuple[int, int], float], u: int, v: int) -> float:
    key = (u, v)
    if key not in cache:
        cache[key] = 0.0 if u == v else ch.query(u, v).seconds
    return cache[key]


def _path(ch: ContractionHierarchy, cache: dict[tuple[int, int], list[int]], u: int, v: int):
    key = (u, v)
    if key not in cache:
        cache[key] = [u] if u == v else ch.query(u, v).path
    return cache[key]


def _within_limits(onboard: float, ride: Ride, config: DispatchConfig) -> bool:
    return onboard <= min(
        ride.trip_s * (1 + config.max_detour_frac), ride.trip_s + config.max_extra_s
    )


def _join_en_route(
    waiting: list[int],
    rides: list[Ride],
    drivers: list[Driver],
    t: float,
    legs: list[tuple[int, int, float, int, int, float, int]],
    net,
    config: DispatchConfig,
    seconds: Callable[[int, int], float],
    path: Callable[[int, int], list[int]],
) -> set[int]:
    """Let waiting riders join cars already carrying one rider the same way.

    The car's position is the last junction on its route it has reached:
    every prefix of a quickest route is itself quickest, so a binary search
    over CH queries from the leg's start finds it. A rider joins if the car
    can collect them within the pickup limit, neither rider's time in the car
    breaks the detour limits, and the extra driving is well under what a
    separate car would need for the trip.
    """
    lat, lon = net.lat, net.lon
    options = []
    for di, d in enumerate(drivers):
        if d.trip_leg < 0 or d.free_at - t < 4 * 60:
            continue
        drv, kind, t0, origin, dest, dur, a_i = legs[d.trip_leg]
        if kind != 1 or not t0 < t < t0 + dur:
            continue
        a = rides[a_i]
        route = path(origin, dest)
        lo, hi = 0, len(route) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if seconds(origin, route[mid]) <= t - t0:
                lo = mid
            else:
                hi = mid - 1
        x = route[lo]
        if x == dest:
            continue
        rem_a = seconds(x, dest)
        for b_i in waiting:
            b = rides[b_i]
            if (
                haversine_m(lat[x], lon[x], lat[b.origin], lon[b.origin])
                > config.pool_radius_km * 1000
            ):
                continue
            va = (lat[dest] - lat[x], lon[dest] - lon[x])
            vb = (lat[b.dest] - lat[b.origin], lon[b.dest] - lon[b.origin])
            norm = math.hypot(*va) * math.hypot(*vb)
            if norm == 0 or (va[0] * vb[0] + va[1] * vb[1]) / norm < 0.5:
                continue
            t1 = seconds(x, b.origin)
            if t1 > config.max_pickup_s:
                continue
            so_far = t - a.picked_up
            for a_first in (True, False):
                first, second = (dest, b.dest) if a_first else (b.dest, dest)
                t2 = seconds(b.origin, first)
                t3 = seconds(first, second)
                on_a = so_far + t1 + t2 + (0.0 if a_first else t3)
                on_b = t2 + (t3 if a_first else 0.0)
                if not (_within_limits(on_a, a, config) and _within_limits(on_b, b, config)):
                    continue
                extra = t1 + t2 + t3 - rem_a
                gain = b.trip_s - extra
                if extra <= 0.75 * b.trip_s and gain >= config.min_saving_s:
                    options.append((gain, di, b_i, x, t1, t2, t3, a_first))
    options.sort(key=lambda o: -o[0])
    used_drivers: set[int] = set()
    joined: set[int] = set()
    for _gain, di, b_i, x, t1, t2, t3, a_first in options:
        if di in used_drivers or b_i in joined:
            continue
        used_drivers.add(di)
        joined.add(b_i)
        d = drivers[di]
        drv, kind, t0, origin, dest, dur, a_i = legs[d.trip_leg]
        a, b = rides[a_i], rides[b_i]
        # Cut the leg short where the car is (it reached x at most one road ago).
        legs[d.trip_leg] = (drv, kind, t0, origin, x, seconds(origin, x), a_i)
        first, second = (dest, b.dest) if a_first else (b.dest, dest)
        legs.append((di, 1, t, x, b.origin, t1, a_i))
        legs.append((di, 3, t + t1, b.origin, first, t2, a_i))
        legs.append((di, 1, t + t1 + t2, first, second, t3, b_i if a_first else a_i))
        b.assigned, b.assigned_at, b.pickup_s, b.picked_up = True, t, t1, t + t1
        a.shared = b.shared = True
        a_drop = t + t1 + t2 + (0.0 if a_first else t3)
        b_drop = t + t1 + t2 + (t3 if a_first else 0.0)
        a.onboard_s = a_drop - a.picked_up
        b.onboard_s = b_drop - b.picked_up
        end = t + t1 + t2 + t3
        d.busy_s += end - d.free_at
        d.free_at = d.idle_since = end
        d.node = second
        d.trip_leg = -1
    return joined


def _pool_jobs(
    waiting: list[int],
    rides: list[Ride],
    net,
    config: DispatchConfig,
    seconds: Callable[[int, int], float],
) -> list[Job]:
    """Pair up waiting riders whose trips line up; everyone else rides alone."""
    lat, lon = net.lat, net.lon
    options: list[tuple[float, int, int, Job]] = []
    for x in range(len(waiting)):
        a_i = waiting[x]
        a = rides[a_i]
        for y in range(x + 1, len(waiting)):
            b_i = waiting[y]
            b = rides[b_i]
            if haversine_m(lat[a.origin], lon[a.origin], lat[b.origin], lon[b.origin]) > (
                config.pool_radius_km * 1000
            ):
                continue
            # Heading the same way: the trips' straight-line directions agree.
            va = (lat[a.dest] - lat[a.origin], lon[a.dest] - lon[a.origin])
            vb = (lat[b.dest] - lat[b.origin], lon[b.dest] - lon[b.origin])
            norm = math.hypot(*va) * math.hypot(*vb)
            if norm == 0 or (va[0] * vb[0] + va[1] * vb[1]) / norm < 0.5:
                continue
            best = None
            for (p, p_i), (q, q_i) in (((a, a_i), (b, b_i)), ((b, b_i), (a, a_i))):
                t1 = seconds(p.origin, q.origin)
                for (d1, d1_i), (d2, d2_i) in (((p, p_i), (q, q_i)), ((q, q_i), (p, p_i))):
                    t2 = seconds(q.origin, d1.dest)
                    t3 = seconds(d1.dest, d2.dest)
                    on_p = t1 + t2 + (t3 if d2_i == p_i else 0.0)
                    on_q = t2 + (t3 if d2_i == q_i else 0.0)
                    if not all(
                        on
                        <= min(
                            r.trip_s * (1 + config.max_detour_frac), r.trip_s + config.max_extra_s
                        )
                        for on, r in ((on_p, p), (on_q, q))
                    ):
                        continue
                    saving = a.trip_s + b.trip_s - (t1 + t2 + t3)
                    if best is None or saving > best[0]:
                        stops = [
                            (p.origin, p_i, True),
                            (q.origin, q_i, True),
                            (d1.dest, d1_i, False),
                            (d2.dest, d2_i, False),
                        ]
                        best = (saving, (stops, [t1, t2, t3]))
            if best is not None and best[0] >= config.min_saving_s:
                options.append((best[0], a_i, b_i, best[1]))
    options.sort(key=lambda o: -o[0])
    used: set[int] = set()
    jobs: list[Job] = []
    for _, a_i, b_i, job in options:
        if a_i in used or b_i in used:
            continue
        used.update((a_i, b_i))
        jobs.append(job)
    jobs.extend(_single(r, rides[r]) for r in waiting if r not in used)
    return jobs


def _assign(
    d: Driver,
    driver_index: int,
    job: Job,
    pickup: float,
    t: float,
    rides: list[Ride],
    legs: list[tuple[int, int, float, int, int, float, int]],
) -> None:
    stops, drives = job
    first_node, first_ride, _ = stops[0]
    legs.append((driver_index, 0, t, d.node, first_node, pickup, first_ride))
    for r in _riders(job):
        rides[r].assigned, rides[r].assigned_at = True, t
        rides[r].shared = len(stops) > 2
    clock = t + pickup
    onboard = [first_ride]
    rides[first_ride].picked_up = clock
    rides[first_ride].pickup_s = pickup
    d.trip_leg = len(legs) if len(stops) == 2 else -1
    for (a_node, _, _), (b_node, b_ride, b_pick), drive in zip(
        stops[:-1], stops[1:], drives, strict=True
    ):
        kind = 1 if len(onboard) == 1 else 3
        legs.append((driver_index, kind, clock, a_node, b_node, drive, onboard[0]))
        clock += drive
        if b_pick:
            onboard.append(b_ride)
            rides[b_ride].picked_up = clock
            rides[b_ride].pickup_s = clock - t
        else:
            onboard.remove(b_ride)
            rides[b_ride].onboard_s = clock - rides[b_ride].picked_up
    d.busy_s += clock - t
    d.free_at = clock
    d.idle_since = clock
    d.node = stops[-1][0]
    d.heading_to_cell = -1


def _surge_multipliers(
    grid,
    waiting: list[int],
    rides: list[Ride],
    drivers: list[Driver],
    t: float,
    hour: float,
    forecast: Forecast,
    config: DispatchConfig,
) -> np.ndarray:
    """Price multiplier per square from riders against free drivers nearby."""
    demand = forecast.next_slot(hour + SLOT_MIN / 60) / 3  # about the next five minutes
    for r in waiting:
        demand[grid.cell[rides[r].origin]] += 1
    supply = np.zeros(grid.num_cells)
    for d in drivers:
        if d.free_at <= t + 5 * 60:
            supply[grid.cell[d.node]] += 1
    area_demand = _neighbour_sums(grid, demand) + demand
    area_supply = _neighbour_sums(grid, supply) + supply
    ratio = area_demand / (area_supply + 1.0)
    m = 1.0 + config.surge_sensitivity * (ratio - config.surge_threshold)
    return np.round(np.clip(m, 1.0, config.surge_cap), 1)


def _reposition(
    drivers: list[Driver],
    t: float,
    hour: float,
    forecast: Forecast,
    ch: ContractionHierarchy,
    cell_targets: list[int],
    cell_buckets: dict[int, list[tuple[int, float]]],
    config: DispatchConfig,
    legs: list[tuple[int, int, float, int, int, float, int]],
    weight: np.ndarray | None = None,
) -> None:
    """Send idle drivers to neighbourhoods where demand will most exceed supply.

    Single 500 m squares see too few requests to act on, so need is judged
    over each square and its eight neighbours (about 1.5 km across).
    """
    grid = forecast.grid
    demand = forecast.next_slot(hour + SLOT_MIN / 60)
    supply = np.zeros(grid.num_cells)
    for d in drivers:
        if d.free_at <= t + 5 * 60:  # free now or within five minutes
            supply[grid.cell[d.node] if d.heading_to_cell < 0 else d.heading_to_cell] += 1

    def neighbourhood_gap() -> np.ndarray:
        area = _neighbour_sums(grid, demand) + demand
        cover = _neighbour_sums(grid, supply) + supply
        return area - cover

    gap = neighbourhood_gap()
    candidates = [
        i
        for i, d in enumerate(drivers)
        if d.free_at <= t
        and t - d.idle_since >= config.idle_before_reposition_s
        and gap[grid.cell[d.node]] < 0
    ]
    if not candidates:
        return
    times = ch.many_to_many([drivers[i].node for i in candidates], cell_targets, cell_buckets)
    active = grid.active
    for row, i in enumerate(candidates):
        d = drivers[i]
        here = int(grid.cell[d.node])
        pull = gap[active] if weight is None else gap[active] * weight[active]
        score = pull - times[row] / 300.0  # each minute's drive costs 0.2 of a rider
        score[times[row] > config.max_reposition_s] = -np.inf
        k = int(np.argmax(score))
        target = int(active[k])
        if not np.isfinite(score[k]) or gap[target] < 1.0 or target == here:
            continue
        drive = float(times[row, k])
        legs.append((i, 2, t, d.node, cell_targets[k], drive, -1))
        d.trip_leg = -1
        d.node = cell_targets[k]
        d.free_at = t + drive
        d.idle_since = d.free_at
        d.reposition_s += drive
        d.heading_to_cell = target
        supply[here] -= 1
        supply[target] += 1
        gap = neighbourhood_gap()


def _metrics(rides: list[Ride], drivers: list[Driver], config: DispatchConfig) -> dict[str, float]:
    counted = [r for r in rides if r.created >= config.warmup_min * 60]
    served = [r for r in counted if r.assigned]
    shared = [r for r in served if r.shared and math.isfinite(r.onboard_s)]
    waits = (
        np.array([(r.picked_up - r.created) / 60 for r in served]) if served else np.array([0.0])
    )
    pickups = np.array([r.pickup_s / 60 for r in served]) if served else np.array([0.0])
    window = config.hours * 3600
    return {
        "requests": float(len(counted)),
        "served_pct": 100 * len(served) / max(len(counted), 1),
        "cancelled_pct": 100 * sum(r.cancelled for r in counted) / max(len(counted), 1),
        "mean_wait_min": float(waits.mean()),
        "p90_wait_min": float(np.quantile(waits, 0.9)),
        "mean_pickup_drive_min": float(pickups.mean()),
        "utilisation_pct": 100
        * sum(min(d.busy_s, window) for d in drivers)
        / (window * len(drivers)),
        "repositioning_min_per_driver": sum(d.reposition_s for d in drivers) / 60 / len(drivers),
        "driving_min_per_ride": sum(d.busy_s + d.reposition_s for d in drivers)
        / 60
        / max(sum(r.assigned for r in rides), 1),
        "shared_pct": 100 * len(shared) / max(len(served), 1),
        "extra_onboard_min": float(np.mean([(r.onboard_s - r.trip_s) / 60 for r in shared]))
        if shared
        else 0.0,
        "priced_out_pct": 100 * sum(r.priced_out for r in counted) / max(len(counted), 1),
        "mean_multiplier": float(np.mean([r.multiplier for r in served])) if served else 1.0,
    }


def compare(
    world: DemandWorld,
    day_index: int,
    ch: ContractionHierarchy,
    config: DispatchConfig,
    forecast_model: dict[str, object],
    *,
    history: np.ndarray | None = None,
    seed: int = 0,
    log: Callable[[str], None] = lambda m: None,
) -> list[DispatchResult]:
    """Run every policy on the same day of requests and the same starting fleet."""
    day = world.day(day_index)
    rng = np.random.default_rng((seed, day_index))
    rides = generate_rides(day, ch, config, rng)
    # What the forecaster would have seen earlier that day, before the window.
    prefill = np.zeros((SLOTS_PER_DAY, world.grid.num_cells))
    start_slot = int(config.start_hour * 60 // SLOT_MIN)
    prefill[:start_slot] = day.sample_counts(np.random.default_rng((seed, day_index, 1)))[
        :start_slot
    ]
    log(f"  day {day_index}: {len(rides)} ride requests, {config.drivers} drivers")
    results = []
    for policy in POLICIES:
        results.append(
            simulate(
                world, day, rides, ch, policy, config,
                forecast_model=forecast_model, seed=seed, prefill=prefill, history=history,
            )
        )  # fmt: skip
    return results


def results_markdown(per_policy: dict[str, dict[str, float]]) -> str:
    lines = [
        "| Policy | Avg. wait | 90th pct. wait | Pickup drive | Served | Cancelled |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, m in per_policy.items():
        lines.append(
            f"| {POLICY_NAMES[policy]} | {m['mean_wait_min']:.1f} min "
            f"| {m['p90_wait_min']:.1f} min "
            f"| {m['mean_pickup_drive_min']:.1f} min | {m['served_pct']:.1f}% "
            f"| {m['cancelled_pct']:.1f}% |"
        )
    return "\n".join(lines)


def average(results: list[list[DispatchResult]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        rows = [r.metrics for day in results for r in day if r.policy == policy]
        out[policy] = {k: float(np.mean([m[k] for m in rows])) for k in rows[0]}
    return out


def replay_json(results: list[DispatchResult], config: DispatchConfig) -> dict[str, object]:
    """One simulated window per policy, compact enough for the browser to replay.

    Routes are not stored: the browser rebuilds each leg's path with the same
    Contraction Hierarchy, which gives the same route and the same duration.
    """

    def ride_row(r: Ride) -> list[float]:
        return [
            round(r.created, 1),
            r.origin,
            r.dest,
            round(r.assigned_at, 1) if r.assigned else -1,
            round(r.picked_up, 1) if r.assigned else -1,
            2 if r.priced_out else 1 if r.cancelled else 0,
            1 if r.shared else 0,
            r.multiplier,
        ]

    return {
        "start_hour": config.start_hour,
        "hours": config.hours,
        "warmup_min": config.warmup_min,
        "drivers": config.drivers,
        "leg_kinds": list(LEG_KINDS),
        "policies": [
            {
                "id": r.policy,
                "name": POLICY_NAMES[r.policy],
                "start_nodes": r.start_nodes,
                "rides": [ride_row(x) for x in r.rides],
                "legs": [
                    [a, k, round(t0, 1), u, v, round(dur, 1), ride]
                    for a, k, t0, u, v, dur, ride in r.legs
                ],
                "metrics": r.metrics,
                "surge": [
                    [round(t, 1), [[c, m] for c, m in cells.items()]] for t, cells in r.surge
                ],
            }
            for r in results
        ],
    }
