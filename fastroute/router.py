"""One entry point for routing under a scenario: time of day, weekday or weekend, turns.

The algorithms only know about graphs and node ids. ``Router`` builds the
right graph for a scenario and translates both ways, so callers think in
intersections and coordinates while the search runs on whichever graph
(plain or turn-expanded) the scenario needs.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from .astar import astar
from .bidirectional import bidirectional_dijkstra
from .ch import ContractionHierarchy
from .dijkstra import dijkstra
from .graph import Graph, RoadNetwork, free_flow_cost
from .result import Route
from .traffic import traffic_cost
from .turns import TurnGraph

SEARCHES: dict[str, Callable[[Graph, int, int], Route]] = {
    "dijkstra": dijkstra,
    "bidirectional": bidirectional_dijkstra,
    "astar": astar,
}


@dataclass(frozen=True)
class Scenario:
    hour: float | None = None  # None means free-flow, no traffic
    weekend: bool = False
    turns: bool = False

    def label(self) -> str:
        if self.hour is None:
            when = "free flow"
        else:
            h, m = divmod(round(self.hour * 60), 60)
            when = f"{h:02d}:{m:02d} {'weekend' if self.weekend else 'weekday'}"
        return when + (", with turn costs" if self.turns else "")

    def file_tag(self) -> str:
        """Distinguishes cached hierarchies built for different scenarios."""
        parts = []
        if self.hour is not None:
            h, m = divmod(round(self.hour * 60), 60)
            parts.append(f"{h:02d}{m:02d}{'we' if self.weekend else 'wd'}")
        if self.turns:
            parts.append("turns")
        return "-".join(parts)


@dataclass(frozen=True)
class Trip:
    """A route expressed in intersections of the original road network."""

    seconds: float
    metres: float
    intersections: list[int]
    settled: int
    algorithm: str
    manoeuvres: Counter[str] = field(default_factory=Counter)

    @property
    def found(self) -> bool:
        return bool(self.intersections)


class Router:
    def __init__(
        self,
        network: RoadNetwork,
        scenario: Scenario = Scenario(),  # noqa: B008 - frozen, safe to share
        *,
        turn_graph: TurnGraph | None = None,
    ) -> None:
        self.network = network
        self.scenario = scenario
        cost = (
            free_flow_cost
            if scenario.hour is None
            else traffic_cost(scenario.hour, weekend=scenario.weekend)
        )
        # The plain graph always exists: it answers coordinate lookups and
        # measures route length in metres.
        self.roads = Graph(network, cost)
        if scenario.turns:
            self.turn_graph: TurnGraph | None = turn_graph or TurnGraph(network)
            self.graph = self.turn_graph.graph(cost)
        else:
            self.turn_graph = None
            self.graph = self.roads

    def nearest(self, lat: float, lon: float) -> int:
        """Nearest intersection to a coordinate."""
        return self.roads.nearest_node(lat, lon)

    def search_nodes(self, origin: int, destination: int) -> tuple[int, int]:
        """Graph node ids to search between for two intersections."""
        if self.turn_graph is None:
            return origin, destination
        return self.turn_graph.start(origin), self.turn_graph.end(destination)

    def random_pairs(self, count: int, seed: int = 0) -> list[tuple[int, int]]:
        rng = random.Random(seed)
        n = self.network.num_nodes
        return [self.search_nodes(rng.randrange(n), rng.randrange(n)) for _ in range(count)]

    def to_trip(self, route: Route) -> Trip:
        if not route.found:
            return Trip(route.seconds, 0.0, [], route.settled, route.algorithm)
        moves: Counter[str]
        if self.turn_graph is None:
            nodes, moves = route.path, Counter()
        else:
            nodes = self.turn_graph.intersections(route.path)
            moves = self.turn_graph.manoeuvres(route.path)
        metres = self.graph.path_length_m(route.path)
        return Trip(route.seconds, metres, nodes, route.settled, route.algorithm, moves)

    def trip(
        self,
        origin: int,
        destination: int,
        algorithm: str = "bidirectional",
        ch: ContractionHierarchy | None = None,
    ) -> Trip:
        """Quickest trip between two intersections."""
        s, t = self.search_nodes(origin, destination)
        if algorithm == "ch":
            if ch is None:
                raise ValueError("algorithm 'ch' needs a ContractionHierarchy for this scenario")
            route = ch.query(s, t)
        else:
            route = SEARCHES[algorithm](self.graph, s, t)
        return self.to_trip(route)
