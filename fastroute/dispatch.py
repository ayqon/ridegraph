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

POLICIES = ("nearest", "optimal", "optimal+history", "optimal+forecast")
POLICY_NAMES = {
    "nearest": "Nearest driver",
    "optimal": "Optimal matching",
    "optimal+history": "Optimal + repositioning on historical averages",
    "optimal+forecast": "Optimal + repositioning on the ML forecast",
}
# Seconds of pickup a rider's every second of waiting so far is worth: keeps
# optimal matching from starving riders who are awkward to reach.
WAIT_PRIORITY = 1.0


@dataclass(frozen=True)
class DispatchConfig:
    drivers: int = 110
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


@dataclass
class Driver:
    node: int
    free_at: float = 0.0
    idle_since: float = 0.0
    busy_s: float = 0.0
    reposition_s: float = 0.0
    heading_to_cell: int = -1


LEG_KINDS = ("pickup", "trip", "reposition")


@dataclass
class DispatchResult:
    policy: str
    metrics: dict[str, float]
    rides: list[Ride] = field(default_factory=list)
    start_nodes: list[int] = field(default_factory=list)
    # (driver, kind index into LEG_KINDS, start s, from node, to node, seconds, ride or -1)
    legs: list[tuple[int, int, float, int, int, float, int]] = field(default_factory=list)


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
    use_forecast = policy in ("optimal+forecast", "optimal+history")
    if policy == "optimal+forecast" and forecast_model is None:
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
    while t <= end or waiting:
        hour = config.start_hour + t / 3600
        while next_ride < len(rides) and rides[next_ride].created <= t:
            if forecast is not None:
                forecast.observe(
                    config.start_hour + rides[next_ride].created / 3600, rides[next_ride].origin
                )
            waiting.append(next_ride)
            next_ride += 1
        for r in waiting:
            if t - rides[r].created > config.patience_s:
                rides[r].cancelled = True
        waiting = [r for r in waiting if not rides[r].cancelled]

        idle = [i for i, d in enumerate(drivers) if d.free_at <= t]
        if idle and waiting:
            cost = ch.many_to_many(
                [drivers[i].node for i in idle], [rides[r].origin for r in waiting]
            )
            if policy == "nearest":
                pairs = greedy_matches(cost, config.max_pickup_s)
            else:
                waited = np.array([t - rides[r].created for r in waiting])
                priority = cost - WAIT_PRIORITY * waited[None, :]
                allowed = np.where(cost <= config.max_pickup_s, priority, np.inf)
                pairs = optimal_matches(allowed, limit=np.inf)
            matched = set()
            for i, j in pairs:
                d, ride = drivers[idle[i]], rides[waiting[j]]
                pickup = float(cost[i, j])
                ride.assigned, ride.pickup_s = True, pickup
                ride.assigned_at = t
                ride.picked_up = t + pickup
                legs.append((idle[i], 0, t, d.node, ride.origin, pickup, waiting[j]))
                legs.append(
                    (idle[i], 1, t + pickup, ride.origin, ride.dest, ride.trip_s, waiting[j])
                )
                d.busy_s += pickup + ride.trip_s
                d.free_at = t + pickup + ride.trip_s
                d.idle_since = d.free_at
                d.node = ride.dest
                d.heading_to_cell = -1
                matched.add(waiting[j])
            waiting = [r for r in waiting if r not in matched]

        if forecast is not None and t % config.reposition_every_s < config.batch_s:
            _reposition(drivers, t, hour, forecast, ch, cell_targets, cell_buckets, config, legs)

        t += config.batch_s
        if t > end + config.patience_s:
            break

    return DispatchResult(policy, _metrics(rides, drivers, config), rides, start_nodes, legs)


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
        score = gap[active] - times[row] / 300.0  # each minute's drive costs 0.2 of a rider
        score[times[row] > config.max_reposition_s] = -np.inf
        k = int(np.argmax(score))
        target = int(active[k])
        if not np.isfinite(score[k]) or gap[target] < 1.0 or target == here:
            continue
        drive = float(times[row, k])
        legs.append((i, 2, t, d.node, cell_targets[k], drive, -1))
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
            1 if r.cancelled else 0,
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
            }
            for r in results
        ],
    }
