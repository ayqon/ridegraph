"""Where and when people want rides, and a model that forecasts it.

Ride requests are not spread evenly. Commuters pour out of the main-line
stations in the morning and back into them in the evening, the City fills up
and empties, the West End peaks at night. On top of that rhythm, every day
differs: a match, a strike, a concert makes one place hot for a few hours.

``DemandWorld`` generates requests with that structure. ``DemandForecaster``
learns to predict how many requests each 500 m grid square will see in the
next 15 minutes from what has been observed so far today, and is compared
with the two forecasts a dispatcher gets for free: the historical average for
that square and time, and "same as the last 15 minutes". The dispatch
simulator then uses the forecast to move idle drivers to where riders are
about to appear, so the model is judged by the waiting time it saves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .graph import RoadNetwork

KM_PER_DEG_LAT = 111.32
SLOT_MIN = 15
SLOTS_PER_DAY = 24 * 60 // SLOT_MIN
CELL_KM = 0.5

# Relative strength by hour (00..23) of each kind of place as a source of rides
# (origin) and as a destination.
PROFILES: dict[str, dict[str, tuple[float, ...]]] = {
    "station": {
        "origin": (0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.9, 2.2, 2.6, 1.6, 0.8, 0.7,
                   0.7, 0.7, 0.7, 0.8, 1.0, 1.1, 1.0, 0.9, 0.7, 0.6, 0.5, 0.3),
        "dest": (0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.5, 0.6, 0.6, 0.5, 0.5, 0.6,
                 0.6, 0.6, 0.7, 0.9, 1.6, 2.2, 2.4, 1.6, 0.9, 0.7, 0.5, 0.3),
    },
    "office": {
        "origin": (0.05, 0.02, 0.02, 0.02, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7,
                   0.9, 0.8, 0.6, 0.7, 1.2, 2.2, 2.4, 1.4, 0.7, 0.4, 0.2, 0.1),
        "dest": (0.05, 0.02, 0.02, 0.02, 0.05, 0.2, 0.8, 2.0, 2.6, 1.6, 0.8, 0.7,
                 0.8, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05),
    },
    "nightlife": {
        "origin": (1.8, 1.4, 0.8, 0.3, 0.1, 0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6,
                   0.8, 0.8, 0.7, 0.7, 0.8, 1.0, 1.1, 1.0, 1.0, 1.3, 1.8, 2.1),
        "dest": (0.6, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8,
                 1.0, 0.9, 0.8, 0.8, 1.0, 1.4, 1.9, 2.2, 2.0, 1.6, 1.2, 0.9),
    },
    "background": {
        "origin": (0.3, 0.2, 0.1, 0.1, 0.1, 0.2, 0.5, 0.9, 1.0, 0.8, 0.7, 0.7,
                   0.8, 0.8, 0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.4),
        "dest": (0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.4, 0.7, 0.8, 0.7, 0.7, 0.7,
                 0.8, 0.8, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.6),
    },
}  # fmt: skip

# Overall request rate by hour, relative to the peak.
DEMAND_CURVE: tuple[float, ...] = (
    0.35, 0.25, 0.15, 0.1, 0.1, 0.15, 0.35, 0.75, 1.0, 0.75, 0.55, 0.55,
    0.6, 0.6, 0.55, 0.6, 0.75, 0.95, 1.0, 0.85, 0.7, 0.6, 0.55, 0.45,
)  # fmt: skip

CENTRE_SPREAD_KM = 0.45
EVENT_SPREAD_KM = 0.35
EVENTS_PER_DAY = 2.0  # matches, concerts, closures: unannounced local surges
BACKGROUND_SHARE = 0.35  # of all requests, spread evenly over the road network


@dataclass(frozen=True)
class Centre:
    name: str
    kind: str
    x_km: float
    y_km: float
    strength: float


class Grid:
    """500 m squares over the network, with node-to-square lookups."""

    def __init__(self, network: RoadNetwork) -> None:
        self.lat0 = float((network.lat.min() + network.lat.max()) / 2)
        self.lon0 = float((network.lon.min() + network.lon.max()) / 2)
        self.km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(self.lat0))
        self.x = (network.lon - self.lon0) * self.km_per_deg_lon
        self.y = (network.lat - self.lat0) * KM_PER_DEG_LAT
        self.x0, self.y0 = float(self.x.min()), float(self.y.min())
        self.cols = int((self.x.max() - self.x0) // CELL_KM) + 1
        self.rows = int((self.y.max() - self.y0) // CELL_KM) + 1
        self.cell = (((self.y - self.y0) // CELL_KM).astype(np.int64) * self.cols) + (
            (self.x - self.x0) // CELL_KM
        ).astype(np.int64)
        self.num_cells = self.rows * self.cols
        nodes_per_cell = np.bincount(self.cell, minlength=self.num_cells)
        self.active = np.flatnonzero(nodes_per_cell > 0)
        # A representative node per square: the one nearest the square's centre.
        cx = self.x0 + (np.arange(self.num_cells) % self.cols + 0.5) * CELL_KM
        cy = self.y0 + (np.arange(self.num_cells) // self.cols + 0.5) * CELL_KM
        self.centre_xy = np.stack([cx, cy], axis=1)
        d = (self.x - cx[self.cell]) ** 2 + (self.y - cy[self.cell]) ** 2
        order = np.lexsort((d, self.cell))
        first = np.ones(len(order), dtype=bool)
        first[1:] = self.cell[order][1:] != self.cell[order][:-1]
        self.centre_node = np.full(self.num_cells, -1, dtype=np.int64)
        self.centre_node[self.cell[order][first]] = order[first]

    def xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (lon - self.lon0) * self.km_per_deg_lon, (lat - self.lat0) * KM_PER_DEG_LAT

    def neighbours(self, c: int) -> list[int]:
        r, k = divmod(c, self.cols)
        out = []
        for dr in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if (dr or dk) and 0 <= r + dr < self.rows and 0 <= k + dk < self.cols:
                    out.append((r + dr) * self.cols + k + dk)
        return out


@dataclass
class DemandWorld:
    """Hidden truth: how many requests appear where, and when.

    Each simulated day draws its own overall level and its own surprise for
    every hotspot (lognormal), which a forecaster can only pick up from the
    day's requests so far.
    """

    network: RoadNetwork
    peak_per_min: float = 4.0
    seed: int = 0
    day_sigma: float = 0.15
    surprise_sigma: float = 0.5
    centres: list[Centre] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.grid = Grid(self.network)
        if not self.centres:
            self.centres = self._default_centres()
        self._nx = self.grid.x
        self._ny = self.grid.y
        # Each node's pull towards each centre, fixed for the network.
        self.kernels = np.stack(
            [
                np.exp(
                    -((self._nx - c.x_km) ** 2 + (self._ny - c.y_km) ** 2)
                    / (2 * CENTRE_SPREAD_KM**2)
                )
                for c in self.centres
            ]
        )
        self.kernels /= self.kernels.sum(axis=1, keepdims=True)
        self.kinds = [c.kind for c in self.centres]

    @classmethod
    def for_places(
        cls, network: RoadNetwork, places: Sequence[tuple[str, str, float, float, float]], **kw
    ) -> DemandWorld:
        """A demand world around named places: (name, kind, lat, lon, strength)."""
        grid = Grid(network)
        centres = []
        for name, kind, lat, lon, strength in places:
            x, y = grid.xy(lat, lon)
            if grid.x.min() <= x <= grid.x.max() and grid.y.min() <= y <= grid.y.max():
                centres.append(Centre(name, kind, x, y, strength))
        if len(centres) < 3:  # places outside this network: fall back to generic centres
            return cls(network, **kw)
        return cls(network, centres=centres, **kw)

    def _default_centres(self) -> list[Centre]:
        inside = []
        from .cities import LONDON

        for name, kind, lat, lon, strength in LONDON.demand_centres:
            x, y = self.grid.xy(lat, lon)
            if (
                self.grid.x.min() <= x <= self.grid.x.max()
                and self.grid.y.min() <= y <= self.grid.y.max()
            ):
                inside.append(Centre(name, kind, x, y, strength))
        if len(inside) >= 6:
            return inside
        # Not London: place one centre of each kind at random junctions.
        rng = np.random.default_rng(self.seed + 99)
        kinds = ["station", "station", "office", "office", "nightlife", "nightlife"]
        picks = rng.choice(self.network.num_nodes, size=len(kinds), replace=False)
        return [
            Centre(f"{k} {i + 1}", k, float(self.grid.x[n]), float(self.grid.y[n]), 1.0)
            for i, (k, n) in enumerate(zip(kinds, picks, strict=True))
        ]

    def day(self, index: int) -> DemandDay:
        rng = np.random.default_rng((self.seed, index))
        level = float(np.exp(rng.normal(0, self.day_sigma)))
        surprise = np.exp(rng.normal(0, self.surprise_sigma, size=len(self.centres)))
        events = []
        for _ in range(rng.poisson(EVENTS_PER_DAY)):
            node = int(rng.integers(self.network.num_nodes))
            start = float(rng.uniform(6, 22))
            events.append(
                Event(
                    float(self.grid.x[node]),
                    float(self.grid.y[node]),
                    start,
                    start + float(rng.uniform(1.0, 2.5)),
                    float(rng.uniform(0.4, 1.0)),
                )
            )
        return DemandDay(self, index, level, surprise, index % 7 >= 5, events)

    def event_kernel(self, event: Event) -> np.ndarray:
        k = np.exp(
            -((self._nx - event.x_km) ** 2 + (self._ny - event.y_km) ** 2)
            / (2 * EVENT_SPREAD_KM**2)
        )
        return k / k.sum()


def _hourly(curve: tuple[float, ...], hour: float) -> float:
    h = hour % 24
    lo = int(h)
    f = h - lo
    return curve[lo] * (1 - f) + curve[(lo + 1) % 24] * f


@dataclass(frozen=True)
class Event:
    """A local surge in ride requests, e.g. crowds leaving a venue."""

    x_km: float
    y_km: float
    start_hour: float
    end_hour: float
    share: float  # extra requests, as a share of the city's normal rate at the time


@dataclass
class DemandDay:
    world: DemandWorld
    index: int
    level: float
    surprise: np.ndarray
    weekend: bool
    events: list[Event] = field(default_factory=list)

    def live_events(self, hour: float) -> list[Event]:
        h = hour % 24
        return [e for e in self.events if e.start_hour <= h < e.end_hour]

    def base_rate_per_min(self, hour: float) -> float:
        scale = 0.75 if self.weekend else 1.0
        return self.world.peak_per_min * self.level * scale * _hourly(DEMAND_CURVE, hour)

    def rate_per_min(self, hour: float) -> float:
        extra = sum(e.share for e in self.live_events(hour))
        return self.base_rate_per_min(hour) * (1 + extra)

    def weights(self, hour: float, end: str) -> np.ndarray:
        """Probability of each node being a ride's origin (or destination) at ``hour``."""
        w = self.world
        mix = np.array(
            [
                c.strength * s * _hourly(PROFILES[c.kind][end], hour)
                for c, s in zip(w.centres, self.surprise, strict=True)
            ]
        )
        hot = mix @ w.kernels
        hot *= (1 - BACKGROUND_SHARE) / max(hot.sum(), 1e-12)
        flat = BACKGROUND_SHARE * _hourly(PROFILES["background"][end], hour)
        weights = hot + flat / w.network.num_nodes
        weights /= weights.sum()
        if end == "origin":
            # Events add requests on top of the normal pattern while they last.
            live = self.live_events(hour)
            if live:
                extra = sum(e.share * w.event_kernel(e) for e in live)
                weights = weights + extra
                weights /= weights.sum()
        return weights

    def expected_counts(self, slot: int) -> np.ndarray:
        """Expected requests per grid square in a 15-minute slot of this day."""
        hour = (slot + 0.5) * SLOT_MIN / 60
        per_node = self.weights(hour, "origin") * self.rate_per_min(hour) * SLOT_MIN
        return np.bincount(
            self.world.grid.cell, weights=per_node, minlength=self.world.grid.num_cells
        )

    def sample_counts(self, rng: np.random.Generator) -> np.ndarray:
        """Requests per (slot, grid square) for the whole day."""
        return np.stack([rng.poisson(self.expected_counts(s)) for s in range(SLOTS_PER_DAY)])


# ------------------------------------------------------------------ forecasting

FORECAST_FEATURES: tuple[str, ...] = (
    "x_km",
    "y_km",
    "hour_sin",
    "hour_cos",
    "weekend",
    "last_15",
    "last_hour",
    "neighbours_last_15",
    "neighbours_last_hour",
    "city_last_15",
    "city_last_hour",
)


def forecast_features(
    grid: Grid, counts: np.ndarray, slot: int, weekend: bool, cells: np.ndarray
) -> np.ndarray:
    """Features for predicting ``counts[slot]`` from slots before it (same day).

    ``counts`` is (slots, cells) of requests observed so far; only rows before
    ``slot`` are read.
    """
    prev = counts[slot - 1] if slot >= 1 else np.zeros(grid.num_cells)
    hour_window = counts[max(0, slot - 4) : slot].sum(axis=0) if slot else np.zeros(grid.num_cells)
    nb_prev = _neighbour_sums(grid, prev)
    nb_hour = _neighbour_sums(grid, hour_window)
    angle = 2 * math.pi * (slot * SLOT_MIN / 60) / 24
    xy = grid.centre_xy[cells]
    n = len(cells)
    return np.column_stack(
        [
            xy[:, 0],
            xy[:, 1],
            np.full(n, math.sin(angle)),
            np.full(n, math.cos(angle)),
            np.full(n, float(weekend)),
            prev[cells],
            hour_window[cells],
            nb_prev[cells],
            nb_hour[cells],
            np.full(n, prev.sum()),
            np.full(n, hour_window.sum()),
        ]
    )


def _neighbour_sums(grid: Grid, values: np.ndarray) -> np.ndarray:
    v = values.reshape(grid.rows, grid.cols)
    padded = np.pad(v, 1)
    total = sum(
        padded[1 + dr : 1 + dr + grid.rows, 1 + dc : 1 + dc + grid.cols]
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    )
    return (total - v).reshape(-1)


@dataclass
class ForecastResult:
    model: object
    metrics: dict[str, dict[str, float]]
    model_json: dict[str, object]
    historical_mean: np.ndarray  # (weekend, slot, cell)


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-9)
    term = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0)
    return float(2 * np.mean(term - (y - mu)))


def train_forecaster(
    world: DemandWorld, *, train_days: int = 42, test_days: int = 14, seed: int = 0
) -> ForecastResult:
    """Learn next-15-minute requests per grid square and compare with baselines."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    from .eta import export_trees

    grid = world.grid
    cells = grid.active
    rng = np.random.default_rng(seed + 7)
    days = [world.day(i) for i in range(train_days + test_days)]
    counts = [d.sample_counts(rng) for d in days]

    # Historical mean per (weekend, slot, square) from training days only.
    hist = np.zeros((2, SLOTS_PER_DAY, grid.num_cells))
    seen = np.zeros(2)
    for d, c in zip(days[:train_days], counts[:train_days], strict=True):
        hist[int(d.weekend)] += c
        seen[int(d.weekend)] += 1
    hist /= np.maximum(seen, 1)[:, None, None]

    def rows(day_range: range) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, y, hist_pred, last = [], [], [], []
        for i in day_range:
            d, c = days[i], counts[i]
            for s in range(4, SLOTS_PER_DAY):
                X.append(forecast_features(grid, c, s, d.weekend, cells))
                y.append(c[s][cells])
                hist_pred.append(hist[int(d.weekend), s][cells])
                last.append(c[s - 1][cells])
        return np.vstack(X), np.concatenate(y), np.concatenate(hist_pred), np.concatenate(last)

    Xtr, ytr, _, _ = rows(range(train_days))
    Xte, yte, hist_te, last_te = rows(range(train_days, train_days + test_days))
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        random_state=seed,
    ).fit(Xtr, ytr)
    pred = model.predict(Xte)

    def score(mu: np.ndarray) -> dict[str, float]:
        # Ranking matters most to a dispatcher: are the busiest squares found?
        return {
            "mae": float(np.mean(np.abs(mu - yte))),
            "deviance": poisson_deviance(yte, mu),
        }

    metrics = {
        "Same as the last 15 minutes": score(last_te.astype(float)),
        "Historical average for the time": score(hist_te),
        "Gradient-boosted forecast": score(pred),
    }
    model_json = export_trees(model, FORECAST_FEATURES)
    model_json["kind"] = (
        "gradient-boosted trees (Poisson) predicting requests per square per 15 min"
    )
    model_json["link"] = "log"  # prediction = exp(sum of trees)
    return ForecastResult(model, metrics, model_json, hist)


def forecast_markdown(metrics: dict[str, dict[str, float]]) -> str:
    lines = ["| Forecast | Mean abs. error (requests) | Poisson deviance |", "|---|---:|---:|"]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['mae']:.3f} | {m['deviance']:.3f} |")
    return "\n".join(lines)
