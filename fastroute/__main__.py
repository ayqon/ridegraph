"""Command line interface.

python -m fastroute fetch            # download central London from OpenStreetMap
python -m fastroute build            # raw OSM JSON -> compact road network
python -m fastroute info             # graph statistics
python -m fastroute route 51.5154,-0.1410 51.5033,-0.1196
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Central London: Paddington and Notting Hill across to Whitechapel,
# King's Cross down to Vauxhall and Elephant & Castle.
DEFAULT_BBOX = (51.485, -0.205, 51.540, -0.065)
DEFAULT_RAW = Path("data/raw/central-london.json.gz")
DEFAULT_NETWORK = Path("data/central-london.npz")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _parse_point(text: str) -> tuple[float, float]:
    try:
        lat, lon = (float(x) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected 'lat,lon', got {text!r}") from exc
    return lat, lon


def cmd_fetch(args: argparse.Namespace) -> int:
    # Standard library only, so this step runs on any machine with Python.
    from .overpass import overpass_query

    query = overpass_query(*args.bbox)
    body = urllib.parse.urlencode({"data": query}).encode()
    request = urllib.request.Request(
        args.url, data=body, headers={"User-Agent": "fastroute (github.com/ayqon/fastroute)"}
    )
    print(f"Querying Overpass for bbox {args.bbox} ...", file=sys.stderr)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        raw = response.read()
    payload = json.loads(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wb") as fh:
        fh.write(raw)
    n_ways = sum(1 for e in payload["elements"] if e["type"] == "way")
    n_nodes = sum(1 for e in payload["elements"] if e["type"] == "node")
    print(
        f"Saved {n_ways:,} ways / {n_nodes:,} nodes to {args.out} "
        f"({len(raw) / 1e6:.1f} MB raw) in {time.perf_counter() - started:.1f}s",
        file=sys.stderr,
    )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from .osm import network_from_overpass

    with gzip.open(args.raw, "rb") as fh:
        payload = json.load(fh)
    network = network_from_overpass(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    network.save(args.out)
    print(f"Built {network.num_nodes:,} nodes / {network.num_edges:,} edges -> {args.out}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .graph import Graph, RoadNetwork

    graph = Graph(RoadNetwork.load(args.network))
    total_km = sum(graph.length_m) / 1000
    print(f"nodes:      {graph.num_nodes:,}")
    print(f"edges:      {graph.num_edges:,}")
    print(f"road length {total_km:,.0f} km (directed)")
    print(f"top speed   {graph.max_speed_mps * 3.6:.0f} km/h")
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    from .dijkstra import dijkstra
    from .graph import Graph, RoadNetwork

    graph = Graph(RoadNetwork.load(args.network))
    s = graph.nearest_node(*args.origin)
    t = graph.nearest_node(*args.destination)
    started = time.perf_counter()
    route = dijkstra(graph, s, t)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not route.found:
        print("No route found.")
        return 1
    km = graph.path_length_m(route.path) / 1000
    minutes, seconds = divmod(round(route.seconds), 60)
    print(f"{route.algorithm}: {minutes} min {seconds:02d} s, {km:.2f} km")
    print(f"  {len(route.path):,} nodes on path, {route.settled:,} settled, {elapsed_ms:.1f} ms")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fastroute", description="Quickest-route search on real road networks.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download drivable roads from OpenStreetMap")
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=DEFAULT_BBOX,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
    )
    p.add_argument("--out", type=Path, default=DEFAULT_RAW)
    p.add_argument("--url", default=OVERPASS_URL)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build", help="convert raw OSM JSON into a road network")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path, default=DEFAULT_NETWORK)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("info", help="print graph statistics")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("route", help="quickest route between two coordinates")
    p.add_argument("origin", type=_parse_point, help="lat,lon")
    p.add_argument("destination", type=_parse_point, help="lat,lon")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.set_defaults(func=cmd_route)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
