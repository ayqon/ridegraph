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
from pathlib import Path

# Central London: Paddington and Notting Hill across to Whitechapel,
# King's Cross down to Vauxhall and Elephant & Castle.
DEFAULT_BBOX = (51.485, -0.205, 51.540, -0.065)
DEFAULT_RAW = Path("data/raw/central-london.json.gz")
DEFAULT_NETWORK = Path("data/central-london.npz")


def _parse_point(text: str) -> tuple[float, float]:
    try:
        lat, lon = (float(x) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected 'lat,lon', got {text!r}") from exc
    return lat, lon


def _require(path: Path, how_to_make: str) -> Path:
    """Exit with a helpful message if a pipeline step's input is missing."""
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m fastroute {how_to_make}` first.")
    return path


def cmd_fetch(args: argparse.Namespace) -> int:
    # Standard library only, so this step runs on any machine with Python.
    from .overpass import OVERPASS_URLS, OverpassError, download, overpass_query

    urls = [args.url] if args.url else list(OVERPASS_URLS)
    print(f"Downloading roads in bbox {tuple(args.bbox)} from OpenStreetMap", file=sys.stderr)
    started = time.perf_counter()
    try:
        raw = download(overpass_query(*args.bbox), urls)
    except OverpassError as exc:
        raise SystemExit(str(exc)) from None
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

    with gzip.open(_require(args.raw, "fetch"), "rb") as fh:
        payload = json.load(fh)
    network = network_from_overpass(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    network.save(args.out)
    print(f"Built {network.num_nodes:,} nodes / {network.num_edges:,} edges -> {args.out}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .graph import Graph, RoadNetwork

    graph = Graph(RoadNetwork.load(_require(args.network, "build")))
    total_km = sum(graph.length_m) / 1000
    print(f"nodes:      {graph.num_nodes:,}")
    print(f"edges:      {graph.num_edges:,}")
    print(f"road length {total_km:,.0f} km (directed)")
    print(f"top speed   {graph.max_speed_mps * 3.6:.0f} km/h")
    return 0


ALGORITHMS = ("ch", "astar", "bidirectional", "dijkstra")
ETA_TIMES = ("03:00", "07:00", "08:00", "10:00", "13:00", "16:00", "17:30", "19:00", "22:00")


def _add_scenario_args(p: argparse.ArgumentParser) -> None:
    from .traffic import parse_time

    def time_arg(text: str) -> float:
        try:
            return parse_time(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    p.add_argument("--at", type=time_arg, metavar="HH:MM", help="time of day for traffic")
    p.add_argument("--weekend", action="store_true", help="use the weekend traffic curve")
    p.add_argument("--turns", action="store_true", help="charge for turns and traffic lights")


def _scenario(args: argparse.Namespace):
    from .router import Scenario

    return Scenario(hour=args.at, weekend=args.weekend, turns=args.turns)


def _ch_path(network_path: Path, scenario) -> Path:
    tag = scenario.file_tag()
    stem = network_path.with_suffix("")
    return Path(f"{stem}.{tag}.ch.npz" if tag else f"{stem}.ch.npz")


def _load_network(path: Path):
    from .graph import RoadNetwork

    return RoadNetwork.load(_require(path, "build"))


def _load_ch(network_path: Path, scenario):
    from .ch import ContractionHierarchy

    path = _ch_path(network_path, scenario)
    if not path.exists():
        flags = " ".join(_scenario_flags(scenario))
        raise SystemExit(
            f"{path} not found. Run `python -m fastroute contract {flags}`.".replace("  ", " ")
        )
    return ContractionHierarchy.load(path)


def _scenario_flags(scenario) -> list[str]:
    flags = []
    if scenario.hour is not None:
        h, m = divmod(round(scenario.hour * 60), 60)
        flags.append(f"--at {h:02d}:{m:02d}")
    if scenario.weekend:
        flags.append("--weekend")
    if scenario.turns:
        flags.append("--turns")
    return flags


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes} min {secs:02d} s"


def _format_manoeuvres(trip) -> str:
    order = ["right", "left", "sharp right", "sharp left", "u turn", "traffic lights"]
    parts = [f"{trip.manoeuvres[k]} {k}" for k in order if trip.manoeuvres.get(k)]
    return ", ".join(parts) if parts else "no turns or lights"


def cmd_contract(args: argparse.Namespace) -> int:
    from .ch import ContractionHierarchy
    from .router import Router

    scenario = _scenario(args)
    router = Router(_load_network(args.network), scenario)
    graph = router.graph

    def progress(done: int, total: int) -> None:
        print(f"\r  contracted {done:,}/{total:,} nodes", end="", file=sys.stderr, flush=True)

    print(f"Contracting for {scenario.label()}", file=sys.stderr)
    ch = ContractionHierarchy.build(graph, progress=progress)
    print(file=sys.stderr)
    out = _ch_path(args.network, scenario)
    ch.save(out)
    print(
        f"Contracted {graph.num_nodes:,} nodes in {ch.build_seconds:.1f}s, "
        f"{ch.num_shortcuts:,} shortcuts ({ch.num_shortcuts / graph.num_edges:.2f}x original "
        f"edges) -> {out}"
    )
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    from .router import Router

    scenario = _scenario(args)
    router = Router(_load_network(args.network), scenario)
    ch = _load_ch(args.network, scenario) if args.algorithm == "ch" else None
    s = router.nearest(*args.origin)
    t = router.nearest(*args.destination)

    started = time.perf_counter()
    trip = router.trip(s, t, args.algorithm, ch)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not trip.found:
        print("No route found.")
        return 1
    print(
        f"{trip.algorithm}, {scenario.label()}: "
        f"{_format_duration(trip.seconds)}, {trip.metres / 1000:.2f} km"
    )
    if scenario.turns:
        print(f"  {_format_manoeuvres(trip)}")
    print(
        f"  {len(trip.intersections):,} intersections on route, "
        f"{trip.settled:,} settled, {elapsed_ms:.1f} ms"
    )
    return 0


def cmd_eta(args: argparse.Namespace) -> int:
    from .graph import Graph
    from .router import Router, Scenario
    from .traffic import parse_time
    from .turns import TurnGraph

    network = _load_network(args.network)
    turn_graph = TurnGraph(network) if args.turns else None
    lookup = Graph(network)
    s = lookup.nearest_node(*args.origin)
    t = lookup.nearest_node(*args.destination)

    first = None
    rows = []
    for text in args.times or ETA_TIMES:
        try:
            hour = parse_time(text)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        scenario = Scenario(hour=hour, weekend=args.weekend, turns=args.turns)
        router = Router(network, scenario, turn_graph=turn_graph)
        trip = router.trip(s, t, "bidirectional")
        if not trip.found:
            print("No route found.")
            return 1
        if first is None:
            first = trip
            note = "reference route"
        elif trip.intersections == first.intersections:
            note = "same route"
        else:
            shared = len(set(trip.intersections) & set(first.intersections))
            note = f"different route ({shared / len(trip.intersections):.0%} shared)"
        rows.append((scenario.label(), _format_duration(trip.seconds), trip.metres / 1000, note))

    width = max(len(r[0]) for r in rows)
    for label, eta, km, note in rows:
        print(f"{label:<{width}}  {eta:>12}  {km:5.2f} km  {note}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from . import bench
    from .ch import ContractionHierarchy
    from .router import Router

    scenario = _scenario(args)
    if args.grid:
        from .synthetic import grid_network

        router = Router(grid_network(args.grid, args.grid, seed=args.seed), scenario)
        print(f"Synthetic {args.grid}x{args.grid} grid, contracting...", file=sys.stderr)
        ch = ContractionHierarchy.build(router.graph)
    else:
        router = Router(_load_network(args.network), scenario)
        ch = _load_ch(args.network, scenario)
    graph = router.graph
    print(
        f"{scenario.label()}: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges, "
        f"{args.queries} random queries\n",
        file=sys.stderr,
    )
    pairs = router.random_pairs(args.queries, args.seed)
    print(bench.to_markdown(bench.run(graph, ch, pairs=pairs)))
    return 0


def cmd_export_web(args: argparse.Namespace) -> int:
    from .web_export import export

    network = _load_network(args.network)
    raw = None
    if args.raw.exists():
        with gzip.open(args.raw, "rb") as fh:
            raw = json.load(fh)
    else:
        print(f"{args.raw} not found; exporting without rivers", file=sys.stderr)
    print(f"Exporting {network.num_nodes:,}-node network to {args.out}", file=sys.stderr)
    export(network, args.out, raw=raw, name=args.name, eta_trips=args.eta_trips)
    return 0


def cmd_eta_train(args: argparse.Namespace) -> int:
    from .eta import metrics_markdown, train

    network = _load_network(args.network)
    print(f"Training ETA models on {args.trips:,} simulated trips", file=sys.stderr)
    result = train(
        network, trips=args.trips, seed=args.seed, log=lambda m: print(m, file=sys.stderr)
    )
    print(metrics_markdown(result.metrics))
    iv = result.interval
    print(
        f"\nArrival window (10th to 90th percentile): {iv['coverage_pct']:.1f}% of held-out "
        f"trips inside (target 80%, {iv['uncalibrated_coverage_pct']:.1f}% before conformal "
        f"calibration), {iv['below_pct']:.1f}% faster, {iv['above_pct']:.1f}% "
        f"slower; mean width {iv['mean_width_pct']:.0f}% of the ETA"
    )
    print("\nMost useful features (permutation importance):")
    for name, value in result.importance[:8]:
        print(f"  {name:<15} {value:.3f}")
    if args.out:
        args.out.write_text(json.dumps(result.model_json, separators=(",", ":")))
        print(f"\nSaved model to {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fastroute", description="Quickest-route search on real road networks."
    )
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
    p.add_argument("--url", help="use only this Overpass server (default: try several)")
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
    p.add_argument("--algorithm", "-a", choices=ALGORITHMS, default="ch")
    _add_scenario_args(p)
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("eta", help="how one trip's time and route change through the day")
    p.add_argument("origin", type=_parse_point, help="lat,lon")
    p.add_argument("destination", type=_parse_point, help="lat,lon")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--times", nargs="+", metavar="HH:MM", help="times of day to compare")
    p.add_argument("--weekend", action="store_true")
    p.add_argument("--turns", action="store_true")
    p.set_defaults(func=cmd_eta)

    p = sub.add_parser("contract", help="preprocess the network into a Contraction Hierarchy")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    _add_scenario_args(p)
    p.set_defaults(func=cmd_contract)

    p = sub.add_parser("bench", help="compare all algorithms on random queries")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--grid", type=int, help="use a synthetic N x N grid instead")
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    _add_scenario_args(p)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("eta-train", help="train and compare learned ETA models")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--trips", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="save the gradient-boosted model as JSON")
    p.set_defaults(func=cmd_eta_train)

    p = sub.add_parser("export-web", help="export graph and hierarchies for the map demo")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--out", type=Path, default=Path("web/data"))
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="raw OSM download, for rivers")
    p.add_argument(
        "--eta-trips",
        type=int,
        default=20_000,
        help="simulated trips to train the ETA model on (0 to skip; needs scikit-learn)",
    )
    p.add_argument("--name", default="Central London")
    p.set_defaults(func=cmd_export_web)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
