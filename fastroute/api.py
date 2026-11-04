"""An HTTP routing service: the engine behind a JSON API.

The map runs in the browser; a product would also want a backend. This
module wraps the package as a FastAPI application:

* ``GET /route``: the quickest route at any time of day, optionally with
  alternatives and Congestion Charge avoidance, as GeoJSON, with the learned
  ETA and its arrival window when a model is loaded.
* ``GET /reach``: area reachable within some minutes, leaving or arriving.
* ``POST /tour``: the best order for a round of stops.
* ``GET /health``: what is loaded.

**Any time of day, not just presets.** Queries run on a Customizable
Contraction Hierarchy (``cch.py``) built once at start-up. A request for
08:15 loads that quarter-hour's travel times into it, which takes tens of
milliseconds, and the result is kept in a small cache, so later requests in
the same quarter-hour pay nothing. A plain CH would need seconds of
preprocessing per time.

Searches share one hierarchy, so they run under a lock; each takes a few
milliseconds, and the service scales by running more worker processes.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from . import charges
from .alternatives import alternatives
from .cch import CustomizableCH
from .graph import Graph, OffNetworkError, RoadNetwork, csr_sorted, snap
from .reach import reach
from .tour import plan_matrix
from .traffic import traffic_cost

MAX_SNAP_M = 500.0
CACHE_SIZE = 8


def parse_point(text: str) -> tuple[float, float]:
    try:
        lat, lon = (float(x) for x in text.split(","))
    except ValueError as exc:
        raise ValueError(f"expected 'lat,lon', got {text!r}") from exc
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"{text!r} is not a valid coordinate")
    return lat, lon


def parse_time(text: str | None) -> float:
    if not text:
        return 8.0
    h, _, m = text.partition(":")
    hour = int(h) + (int(m) if m else 0) / 60
    if not 0 <= hour < 24:
        raise ValueError(f"time must be between 00:00 and 23:59, got {text!r}")
    return hour


@dataclass
class _Metric:
    up: np.ndarray
    down: np.ndarray
    road_seconds: np.ndarray
    direct: dict[tuple[int, int], float] | None = None  # built on first unpack
    lists: tuple[np.ndarray, np.ndarray, list[float], list[float]] | None = None


class RoutingService:
    """Everything the endpoints need, loaded once."""

    def __init__(self, network: RoadNetwork, eta_model: dict[str, Any] | None = None) -> None:
        started = time.perf_counter()
        self.network = csr_sorted(network)
        self.cch = CustomizableCH.build(self.network)
        self.zone = charges.zone_nodes(self.network)
        self._roads = Graph(self.network)  # lengths and adjacency, whatever the time
        self.eta_model = eta_model
        self.ctx = None
        if eta_model is not None:
            from .eta import RouteContext

            self.ctx = RouteContext(self.network)
        self._metrics: OrderedDict[tuple[int, bool], _Metric] = OrderedDict()
        self._graphs: OrderedDict[tuple[int, bool], Graph] = OrderedDict()
        self._lock = threading.Lock()
        self._current: tuple[int, bool] | None = None
        self.ready_seconds = time.perf_counter() - started

    # ------------------------------------------------------------ times

    @staticmethod
    def _key(hour: float, weekend: bool) -> tuple[int, bool]:
        return int(round(hour * 4)) % 96, weekend  # quarter-hours

    def _seconds(self, key: tuple[int, bool]) -> np.ndarray:
        cost = traffic_cost(key[0] / 4, weekend=key[1])
        return np.asarray(cost(self.network), dtype=np.float64)

    def _use(self, key: tuple[int, bool]) -> None:
        """Load a quarter-hour's times into the hierarchy (caller holds the lock)."""
        previous = self._metrics.get(self._current) if self._current else None
        if previous is not None:
            # Keep what the hierarchy built for these times, so switching back is free.
            previous.direct = self.cch._direct
            previous.lists = self.cch._list_cache
        metric = self._metrics.get(key)
        if metric is None:
            seconds = self._seconds(key)
            self.cch.customize(seconds)
            metric = _Metric(self.cch.up, self.cch.down, seconds)
            self._metrics[key] = metric
            if len(self._metrics) > CACHE_SIZE:
                self._metrics.popitem(last=False)
        else:
            self._metrics.move_to_end(key)
            self.cch.up, self.cch.down = metric.up, metric.down
            self.cch._road_seconds = metric.road_seconds
            self.cch._direct = metric.direct
            self.cch._list_cache = metric.lists
        self._current = key

    def graph(self, hour: float, weekend: bool) -> Graph:
        key = self._key(hour, weekend)
        g = self._graphs.get(key)
        if g is None:
            g = Graph(self.network, traffic_cost(key[0] / 4, weekend=key[1]))
            self._graphs[key] = g
            if len(self._graphs) > 4:
                self._graphs.popitem(last=False)
        return g

    def snap(self, point: tuple[float, float]) -> int:
        return snap(self.network, *point, max_m=MAX_SNAP_M)

    # ------------------------------------------------------------ answers

    def geometry(self, nodes: list[int]) -> dict[str, Any]:
        net = self.network
        coords = [[round(float(net.lon[v]), 6), round(float(net.lat[v]), 6)] for v in nodes]
        return {"type": "LineString", "coordinates": coords}

    def charge(self, nodes: list[int], hour: float, weekend: bool) -> dict[str, Any]:
        enters = bool(self.zone[nodes].any()) if nodes else False
        applies = enters and charges.charging(hour, weekend=weekend)
        return {
            "enters_zone": enters,
            "charged": applies,
            "amount_gbp": charges.DAILY_CHARGE_GBP if applies else 0.0,
        }

    def learned_eta(self, nodes: list[int], hour: float, weekend: bool) -> dict[str, Any] | None:
        if self.eta_model is None or self.ctx is None or len(nodes) < 2:
            return None
        from .eta import predict_from_json

        x = self.ctx.features(nodes, hour, weekend)
        log_ratio = predict_from_json(self.eta_model, x)
        out: dict[str, Any] = {"seconds": float(x[0] * math.exp(log_ratio))}
        q = self.eta_model.get("quantiles")
        if q:
            margin = float(self.eta_model.get("conformal_margin", 0.0))
            low = min(predict_from_json(q["p10"], x) - margin, log_ratio)
            high = max(predict_from_json(q["p90"], x) + margin, log_ratio)
            out["window_seconds"] = [float(x[0] * math.exp(low)), float(x[0] * math.exp(high))]
        return out

    def route(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        hour: float,
        weekend: bool,
        *,
        alternatives_count: int = 0,
        avoid_charge: bool = False,
    ) -> dict[str, Any]:
        s, t = self.snap(a), self.snap(b)
        key = self._key(hour, weekend)
        started = time.perf_counter()
        with self._lock:
            self._use(key)
            seconds, nodes = self.cch.query(s, t)
        query_ms = (time.perf_counter() - started) * 1000
        if not nodes:
            raise LookupError("these two points are not connected by road")
        routes = [self._feature("quickest", nodes, seconds, hour, weekend)]
        if alternatives_count or avoid_charge:
            g = self.graph(hour, weekend)
            if alternatives_count:
                found = alternatives(g, s, t, count=1 + alternatives_count, zone=self.zone)
                for i, alt in enumerate(found[1:], 1):
                    routes.append(
                        self._feature(f"alternative {i}", alt.nodes, alt.seconds, hour, weekend)
                    )
            if avoid_charge:
                if self.zone[s] or self.zone[t]:
                    raise ValueError("the trip starts or ends inside the Congestion Charge zone")
                found = alternatives(g, s, t, count=1, zone=self.zone, avoid_zone=True)
                if found:
                    f = found[0]
                    routes.append(
                        self._feature("avoiding the zone", f.nodes, f.seconds, hour, weekend)
                    )
        return {
            "type": "FeatureCollection",
            "features": routes,
            "time": f"{key[0] // 4:02d}:{key[0] % 4 * 15:02d}",
            "weekend": weekend,
            "query_ms": round(query_ms, 3),
        }

    def _feature(
        self, name: str, nodes: list[int], seconds: float, hour: float, weekend: bool
    ) -> dict[str, Any]:
        metres = self._metres(nodes)
        return {
            "type": "Feature",
            "geometry": self.geometry(nodes),
            "properties": {
                "name": name,
                "seconds": round(float(seconds), 1),
                "metres": round(metres, 1),
                "charge": self.charge(nodes, hour, weekend),
                "learned_eta": self.learned_eta(nodes, hour, weekend),
            },
        }

    def _metres(self, nodes: list[int]) -> float:
        """Length of a junction path, taking the shortest road between each pair."""
        g, total = self._roads, 0.0
        for u, v in zip(nodes, nodes[1:], strict=False):
            total += min(
                g.length_m[e] for e in range(g.first_out[u], g.first_out[u + 1]) if g.head[e] == v
            )
        return total

    def reach(
        self,
        a: tuple[float, float],
        hour: float,
        weekend: bool,
        minutes: tuple[int, ...],
        arriving: bool,
    ) -> dict[str, Any]:
        origin = self.snap(a)
        r = reach(self.graph(hour, weekend), origin, minutes, arriving=arriving)
        return {
            "bands": [
                {"minutes": m, "area_km2": round(area, 2), "junctions": n}
                for m, area, n in zip(r.bands_min, r.area_km2, r.junctions, strict=True)
            ],
            "arriving": arriving,
        }

    def tour(
        self, stops: list[tuple[float, float]], hour: float, weekend: bool, round_trip: bool
    ) -> dict[str, Any]:
        nodes = [self.snap(p) for p in stops]
        key = self._key(hour, weekend)
        with self._lock:
            self._use(key)
            matrix = np.array(
                [[self.cch.query(a, b)[0] if a != b else 0.0 for b in nodes] for a in nodes]
            )
        if not np.isfinite(matrix).all():
            raise LookupError("some stops cannot be reached from the others")
        result = plan_matrix(matrix, return_to_start=round_trip)
        best = min(result.seconds, key=lambda name: result.seconds[name])
        return {
            "order": result.orders[best],
            "method": best,
            "seconds": {k: round(v, 1) for k, v in result.seconds.items()},
            "round_trip": round_trip,
        }


class TourRequest(BaseModel):
    """Stops as [lat, lon] pairs, depot first."""

    stops: list[list[float]] = Field(
        ..., examples=[[[51.5174, -0.12], [51.5308, -0.1238], [51.5033, -0.1145]]]
    )
    at: str | None = Field(None, description="time of day, HH:MM (default 08:00)")
    weekend: bool = False
    round_trip: bool = True


def create_app(service: RoutingService):
    """The FastAPI application around a loaded ``RoutingService``."""
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware

    from . import __version__

    app = FastAPI(
        title="fastroute",
        version=__version__,
        description=(
            "Quickest routes, alternatives, Congestion Charge avoidance, travel-time maps "
            "and delivery rounds on a real road network, at any time of day."
        ),
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"])

    def guard(fn):
        try:
            return fn()
        except (ValueError, OffNetworkError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

    @app.get("/health")
    def health() -> dict[str, Any]:
        net = service.network
        return {
            "status": "ok",
            "nodes": net.num_nodes,
            "edges": net.num_edges,
            "hierarchy_edges": service.cch.num_edges,
            "learned_eta": service.eta_model is not None,
            "start_up_seconds": round(service.ready_seconds, 2),
        }

    @app.get("/route")
    def route(
        origin: str = Query(..., alias="from", examples=["51.5154,-0.1410"]),
        destination: str = Query(..., alias="to", examples=["51.5033,-0.1145"]),
        at: str | None = Query(None, description="time of day, HH:MM (default 08:00)"),
        weekend: bool = False,
        alternatives: int = Query(0, ge=0, le=2),
        avoid_charge: bool = False,
    ) -> dict[str, Any]:
        return guard(
            lambda: service.route(
                parse_point(origin),
                parse_point(destination),
                parse_time(at),
                weekend,
                alternatives_count=alternatives,
                avoid_charge=avoid_charge,
            )
        )

    @app.get("/reach")
    def reach_(
        origin: str = Query(..., alias="from", examples=["51.5174,-0.1200"]),
        at: str | None = None,
        weekend: bool = False,
        minutes: str = Query("10,20,30", description="comma-separated minutes"),
        arriving: bool = False,
    ) -> dict[str, Any]:
        def run():
            bands = tuple(sorted(int(m) for m in minutes.split(",")))
            if not bands or bands[0] <= 0 or bands[-1] > 60:
                raise ValueError("minutes must be between 1 and 60")
            return service.reach(parse_point(origin), parse_time(at), weekend, bands, arriving)

        return guard(run)

    @app.post("/tour")
    def tour(request: TourRequest) -> dict[str, Any]:
        def run():
            if not 2 <= len(request.stops) <= 13:
                raise ValueError("give between 2 and 13 stops, depot first")
            points = [parse_point(f"{p[0]},{p[1]}") for p in request.stops]
            return service.tour(points, parse_time(request.at), request.weekend, request.round_trip)

        return guard(run)

    return app


def load_service(network_path: Path, eta_model_path: Path | None = None) -> RoutingService:
    import json

    model = json.loads(eta_model_path.read_text()) if eta_model_path else None
    return RoutingService(RoadNetwork.load(network_path), model)


def app_from_env():
    """App factory for ``uvicorn --factory`` and multi-worker serving.

    Reads ``FASTROUTE_NETWORK`` (default ``data/central-london.npz``) and the
    optional ``FASTROUTE_ETA_MODEL``.
    """
    import os

    network = Path(os.environ.get("FASTROUTE_NETWORK", "data/central-london.npz"))
    model = os.environ.get("FASTROUTE_ETA_MODEL")
    return create_app(load_service(network, Path(model) if model else None))
