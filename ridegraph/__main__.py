"""Command line interface.

python -m ridegraph fetch            # download central London from OpenStreetMap
python -m ridegraph build            # raw OSM JSON -> compact road network
python -m ridegraph info             # graph statistics
python -m ridegraph route 51.5154,-0.1410 51.5033,-0.1196
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

# Central London: Paddington and Notting Hill across to Whitechapel,
# King's Cross down to Vauxhall and Elephant & Castle.
DEFAULT_RAW = Path("data/raw/central-london.json.gz")
DEFAULT_NETWORK = Path("data/central-london.npz")


def _city(args: argparse.Namespace):
    """The chosen city, filling in any per-city defaults the user left out."""
    from . import cities

    city = cities.get(args.city)
    defaults = {
        "bbox": list(city.bbox),
        "raw": Path(city.raw_path),
        "network": Path(city.network_path),
        "name": city.name,
        "eta_trips": city.eta_trips,
        "dispatch_days": city.dispatch_days,
    }
    for key, value in defaults.items():
        if getattr(args, key, "absent") is None:
            setattr(args, key, value)
    if getattr(args, "out", "absent") is None:
        args.out = {
            "fetch": Path(city.raw_path),
            "build": Path(city.network_path),
            "export-web": Path("web/data") / city.key,
        }[args.command]
    return city


def _parse_point(text: str) -> tuple[float, float]:
    try:
        lat, lon = (float(x) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected 'lat,lon', got {text!r}") from exc
    return lat, lon


MAX_SNAP_M = 500.0


def _snap(network, point: tuple[float, float]) -> int:
    """Nearest junction to a coordinate, refusing points off the map."""
    from .graph import OffNetworkError, snap

    try:
        return snap(network, *point, max_m=MAX_SNAP_M)
    except OffNetworkError as exc:
        raise SystemExit(f"{exc}. Pick a point inside the area you fetched.") from None


def _require(path: Path, how_to_make: str) -> Path:
    """Exit with a helpful message if a pipeline step's input is missing."""
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m ridegraph {how_to_make}` first.")
    return path


def cmd_fetch(args: argparse.Namespace) -> int:
    # Standard library only, so this step runs on any machine with Python.
    from .overpass import OVERPASS_URLS, OverpassError, download, overpass_query

    _city(args)
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

    _city(args)
    with gzip.open(_require(args.raw, f"fetch --city {args.city}"), "rb") as fh:
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


def _add_scenario_args(p: argparse.ArgumentParser, *, turns: bool = True) -> None:
    from .traffic import parse_time

    def time_arg(text: str) -> float:
        try:
            return parse_time(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    p.add_argument("--at", type=time_arg, metavar="HH:MM", help="time of day for traffic")
    p.add_argument("--weekend", action="store_true", help="use the weekend traffic curve")
    if turns:
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


def _contract_command(network_path: Path, scenario) -> str:
    flags = _scenario_flags(scenario)
    if Path(network_path) != DEFAULT_NETWORK:
        flags.insert(0, f"--network {network_path}")
    return " ".join(["python -m ridegraph contract", *flags])


def _load_ch(network_path: Path, scenario, *, required: bool = True):
    from .ch import ContractionHierarchy

    path = _ch_path(network_path, scenario)
    if not path.exists():
        if required:
            command = _contract_command(network_path, scenario)
            raise SystemExit(f"{path} not found. Run `{command}`.")
        return None
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
    if args.live or args.tfl:
        return _route_live(args, router, scenario)
    if args.alternatives or args.avoid_charge:
        return _route_options(args, router, scenario)
    s = _snap(router.roads.network, args.origin)
    t = _snap(router.roads.network, args.destination)
    algorithm = args.algorithm
    ch = _load_ch(args.network, scenario, required=False) if algorithm == "ch" else None
    if algorithm == "ch" and ch is None:
        algorithm = "bidirectional"
        print(
            f"No saved hierarchy for {scenario.label()}, so searching with bidirectional "
            f"Dijkstra. Run `{_contract_command(args.network, scenario)}` once for "
            "instant queries.",
            file=sys.stderr,
        )

    started = time.perf_counter()
    trip = router.trip(s, t, algorithm, ch)
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


def _route_live(args: argparse.Namespace, router, scenario) -> int:
    from . import live
    from .cch import CustomizableCH

    if scenario.turns:
        raise SystemExit("--live works without --turns.")
    if args.tfl:
        payload = json.loads(_require(args.tfl, "live --out FILE").read_text())["payload"]
        source = str(args.tfl)
    else:
        try:
            payload = live.fetch()
        except OSError as exc:
            raise SystemExit(f"Could not reach the TfL feed: {exc}") from None
        source = "TfL, live"
    network = router.network
    s = _snap(network, args.origin)
    t = _snap(network, args.destination)
    stretches = live.parse(payload)
    factor, hits = live.match(network, stretches)
    summ = live.summary(stretches, hits)
    print(
        f"Disruptions ({source}): {summ['on_this_map']} on this map, "
        f"{summ['closed']} closed, {summ['slowed']} slowed",
        file=sys.stderr,
    )
    base = _edge_seconds(router, scenario)
    print("Building the Customizable CH (once per map)...", file=sys.stderr)
    cch = CustomizableCH.build(network)
    cch.customize(base)
    before, _ = cch.query(s, t)
    first_ms = cch.customize_seconds * 1000
    cch.customize(base * factor)
    after, path = cch.query(s, t)
    print(f"{scenario.label()}:")
    print(f"  usual traffic        {_format_duration(before):>14}")
    if not path:
        print("  with disruptions     no route: the closures cut these points off")
        return 1
    print(f"  with disruptions     {_format_duration(after):>14}  ({len(path):,} junctions)")
    print(
        f"  re-customised for the new times in {cch.customize_seconds * 1000:.0f} ms "
        f"(first customisation {first_ms:.0f} ms; preprocessing {cch.build_seconds:.1f} s)"
    )
    return 0


def _edge_seconds(router, scenario):
    """Travel time per edge in the network's own edge order."""
    import numpy as np

    from .graph import free_flow_cost
    from .traffic import traffic_cost

    if scenario.hour is None:
        return np.asarray(free_flow_cost(router.network), dtype=np.float64)
    cost = traffic_cost(scenario.hour, weekend=scenario.weekend)
    return np.asarray(cost(router.network), dtype=np.float64)


def _route_options(args: argparse.Namespace, router, scenario) -> int:
    from . import charges
    from .alternatives import alternatives

    if scenario.turns:
        raise SystemExit("--alternatives and --avoid-charge work without --turns.")
    graph = router.roads
    s = _snap(graph.network, args.origin)
    t = _snap(graph.network, args.destination)
    zone = charges.zone_nodes(graph.network)
    applies = charges.charging(scenario.hour, weekend=scenario.weekend)
    routes = alternatives(graph, s, t, count=3 if args.alternatives else 1, zone=zone)
    if not routes:
        print("No route found.")
        return 1
    rows = [("Quickest" if i == 0 else f"Alternative {i}", r) for i, r in enumerate(routes)]
    if args.avoid_charge:
        try:
            avoid = alternatives(graph, s, t, count=1, zone=zone, avoid_zone=True)
        except ValueError as exc:
            print(f"Cannot avoid the zone: {exc}.")
            avoid = []
        if avoid:
            rows.append(("Avoiding the zone", avoid[0]))
    print(f"{scenario.label()}:")
    fee = f"£{charges.DAILY_CHARGE_GBP:.0f}"
    for name, r in rows:
        if not r.enters_zone:
            note = "outside the Congestion Charge zone"
        elif applies:
            note = f"Congestion Charge {fee}"
        else:
            note = "in the zone, outside charging hours"
        extra = r.seconds - routes[0].seconds
        delta = f"+{extra / 60:.0f} min" if extra >= 30 else ""
        print(
            f"  {name:<18} {_format_duration(r.seconds):>14}  {r.metres / 1000:5.2f} km  "
            f"{delta:>8}  {note}"
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
    s = _snap(lookup.network, args.origin)
    t = _snap(lookup.network, args.destination)

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
    from .web_export import export, write_city_index

    city = _city(args)
    network = _load_network(args.network)
    raw = None
    if args.raw.exists():
        with gzip.open(args.raw, "rb") as fh:
            raw = json.load(fh)
    else:
        print(f"{args.raw} not found; exporting without rivers", file=sys.stderr)
    print(f"Exporting {network.num_nodes:,}-node network to {args.out}", file=sys.stderr)
    export(
        network,
        args.out,
        raw=raw,
        name=args.name,
        eta_trips=args.eta_trips,
        dispatch_days=args.dispatch_days,
        city=city,
        **(_tfl_snapshot(args.tfl) if city.tfl_disruptions else {}),
    )
    index = write_city_index(args.out.parent)
    keys = ", ".join(str(c["key"]) for c in index)
    print(f"Cities in {args.out.parent}: {keys}", file=sys.stderr)
    return 0


def _tfl_snapshot(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        print(f"{path} not found; exporting without a disruption snapshot", file=sys.stderr)
        return {}
    data = json.loads(path.read_text())
    return {"tfl": data.get("payload"), "tfl_fetched": data.get("fetched")}


def cmd_live(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    from . import live

    try:
        payload = live.fetch()
    except OSError as exc:
        raise SystemExit(f"Could not reach the TfL feed: {exc}") from None
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017
    stretches = live.parse(payload)
    print(f"{len(stretches)} disrupted stretches in London right now")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"fetched": fetched, "payload": payload}))
        print(f"Saved to {args.out}", file=sys.stderr)
    if args.network.exists():
        factor, hits = live.match(_load_network(args.network), stretches)
        s = live.summary(stretches, hits)
        print(
            f"{s['on_this_map']} on this map: {s['closed']} closed, {s['slowed']} slowed "
            f"({int((factor > 1).sum()):,} road segments)"
        )
    return 0


def cmd_osrm_prepare(args: argparse.Namespace) -> int:
    from .osrm import write_osm_xml, write_segment_speeds

    with gzip.open(_require(args.raw, "fetch"), "rb") as fh:
        payload = json.load(fh)
    args.out.mkdir(parents=True, exist_ok=True)
    nodes, ways = write_osm_xml(payload, args.out / "map.osm")
    segments = write_segment_speeds(_load_network(args.network), args.out / "speeds.csv")
    print(f"Wrote {ways:,} ways, {nodes:,} nodes and {segments:,} segment speeds to {args.out}")
    return 0


def cmd_compare_osrm(args: argparse.Namespace) -> int:
    from functools import partial

    from .ch import ContractionHierarchy
    from .graph import Graph
    from .osrm import compare, markdown, osrm_route

    network = _load_network(args.network)
    print("Contracting the free-flow network...", file=sys.stderr)
    ch = ContractionHierarchy.build(Graph(network))

    def route(s: int, t: int):
        r = ch.query(s, t)
        return r.seconds, r.path

    summary = compare(
        network, route, partial(osrm_route, args.url), queries=args.queries, seed=args.seed
    )
    print(markdown(summary))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .validation import markdown, simulated_speeds

    print(f"Timing {args.trips} random trips per hour in the simulated city...", file=sys.stderr)
    print(markdown(simulated_speeds(_load_network(args.network), trips=args.trips)))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    try:
        import uvicorn
    except ImportError:
        raise SystemExit('Serving needs the API extras: pip install -e ".[api]"') from None
    _require(args.network, "build")
    os.environ["RIDEGRAPH_NETWORK"] = str(args.network)
    if args.eta_model:
        os.environ["RIDEGRAPH_ETA_MODEL"] = str(_require(args.eta_model, "eta-train --out FILE"))
    print(
        f"Loading {args.network} and building the Customizable CH, then serving on "
        f"http://{args.host}:{args.port} (API docs at /docs)",
        file=sys.stderr,
    )
    uvicorn.run(
        "ridegraph.api:app_from_env",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="warning",
    )
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
    print(
        "Learned slowdown map vs the hidden hotspots it was never shown: "
        f"r = {iv['learned_hotspot_r']:.2f}"
    )
    print("\nMost useful features (permutation importance):")
    for name, value in result.importance[:8]:
        print(f"  {name:<15} {value:.3f}")
    if args.out:
        args.out.write_text(json.dumps(result.model_json, separators=(",", ":")))
        print(f"\nSaved model to {args.out}", file=sys.stderr)
    return 0


def cmd_dispatch(args: argparse.Namespace) -> int:
    from .ch import ContractionHierarchy
    from .demand import DemandWorld, forecast_markdown, train_forecaster
    from .dispatch import DispatchConfig, average, compare, results_markdown
    from .graph import csr_sorted
    from .router import Router, Scenario

    network = csr_sorted(_load_network(args.network))
    config = DispatchConfig(drivers=args.drivers, start_hour=args.start, hours=args.hours)
    print("Building the hierarchy and training the demand forecast...", file=sys.stderr)
    ch = ContractionHierarchy.build(Router(network, Scenario(hour=args.start + 1)).graph)
    world = DemandWorld(network, peak_per_min=args.demand)
    forecast = train_forecaster(world)
    print("Demand forecast on held-out days:\n" + forecast_markdown(forecast.metrics) + "\n")
    results = [
        compare(
            world,
            1000 + k,
            ch,
            config,
            forecast.model_json,
            history=forecast.historical_mean,
            log=lambda m: print(m, file=sys.stderr),
        )
        for k in range(args.days)
    ]
    print(f"Dispatch, {args.drivers} drivers, averaged over {args.days} simulated mornings:")
    print(results_markdown(average(results)))
    return 0


def cmd_reach(args: argparse.Namespace) -> int:
    from .graph import Graph
    from .reach import reach
    from .traffic import traffic_cost

    network = _load_network(args.network)
    times = [args.at if args.at is not None else 8.0, args.compare]
    origin = _snap(network, args.point)
    print(
        f"Area {'from which you can reach' if args.arriving else 'reachable from'} "
        f"{args.point[0]:.4f},{args.point[1]:.4f}:"
    )
    header = "          " + "".join(f"{b:>9} min" for b in args.bands)
    print(header)
    for hour in times:
        graph = Graph(network, traffic_cost(hour, weekend=args.weekend))
        r = reach(graph, origin, tuple(args.bands), arriving=args.arriving)
        h, mnt = divmod(round(hour * 60), 60)
        cells = "".join(f"{a:>9.1f} km²" for a in r.area_km2)
        print(f"  {h:02d}:{mnt:02d}  {cells}")
    return 0


def cmd_tour(args: argparse.Namespace) -> int:
    from .ch import ContractionHierarchy
    from .graph import Graph
    from .tour import plan
    from .traffic import traffic_cost

    if len(args.stops) < 2:
        raise SystemExit("Give a depot and at least one stop.")
    network = _load_network(args.network)
    stops = [_snap(network, p) for p in args.stops]
    hour = args.at if args.at is not None else 13.0
    graph = Graph(network, traffic_cost(hour, weekend=args.weekend))
    ch = ContractionHierarchy.build(graph)
    result = plan(ch, stops, return_to_start=not args.open)
    n = len(stops) - 1
    print(
        f"{n} stop{'s' if n != 1 else ''} from the depot, "
        f"{'one way' if args.open else 'round trip'}:"
    )
    entered = result.seconds["As entered"]
    for name, secs in result.seconds.items():
        order = " > ".join(str(i) for i in result.orders[name])
        change = 100 * (secs / entered - 1) if entered > 0 else 0.0
        vs = "same as entered" if abs(change) < 0.5 else f"{change:+.0f}% vs entered"
        print(f"  {name:<28} {_format_duration(secs):>14}  {vs:<16}  {order}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The full command-line interface (also used to generate the docs)."""
    parser = argparse.ArgumentParser(
        prog="ridegraph", description="Quickest-route search on real road networks."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    from .cities import CITIES, DEFAULT_CITY

    def add_city(p: argparse.ArgumentParser) -> None:
        p.add_argument("--city", choices=list(CITIES), default=DEFAULT_CITY)

    p = sub.add_parser("fetch", help="download drivable roads from OpenStreetMap")
    add_city(p)
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        help="override the city's area",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--url", help="use only this Overpass server (default: try several)")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build", help="convert raw OSM JSON into a road network")
    add_city(p)
    p.add_argument("--raw", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("info", help="print graph statistics")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("validate", help="compare simulated speeds with published London figures")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--trips", type=int, default=300)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("serve", help="run the HTTP routing API")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--eta-model", type=Path, help="eta-model.json from eta-train or export-web")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=1, help="worker processes")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("osrm-prepare", help="write OSM XML and segment speeds for OSRM")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--out", type=Path, default=Path("data/osrm"))
    p.set_defaults(func=cmd_osrm_prepare)

    p = sub.add_parser("compare-osrm", help="compare routes and times with a running OSRM")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--url", default="http://127.0.0.1:5000")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, help="also save the summary as JSON")
    p.set_defaults(func=cmd_compare_osrm)

    p = sub.add_parser("live", help="fetch current TfL road disruptions")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--out", type=Path, help="save the feed for export-web --tfl")
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("route", help="quickest route between two coordinates")
    p.add_argument("origin", type=_parse_point, help="lat,lon")
    p.add_argument("destination", type=_parse_point, help="lat,lon")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--algorithm", "-a", choices=ALGORITHMS, default="ch")
    p.add_argument(
        "--alternatives", action="store_true", help="also offer up to two different routes"
    )
    p.add_argument(
        "--avoid-charge", action="store_true", help="keep out of the Congestion Charge zone"
    )
    p.add_argument(
        "--live", action="store_true", help="route around current TfL closures (Customizable CH)"
    )
    p.add_argument("--tfl", type=Path, help="use saved TfL disruptions instead of fetching")
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

    p = sub.add_parser("reach", help="area reachable within a time budget")
    p.add_argument("point", type=_parse_point, help="lat,lon")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--bands", type=int, nargs="+", default=[10, 20, 30], metavar="MIN")
    p.add_argument("--arriving", action="store_true", help="journeys into the point")
    p.add_argument("--compare", type=float, default=3.0, help="hour to compare with")
    _add_scenario_args(p, turns=False)
    p.set_defaults(func=cmd_reach)

    p = sub.add_parser("tour", help="best order to visit several stops")
    p.add_argument("stops", type=_parse_point, nargs="+", help="lat,lon (depot first)")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--open", action="store_true", help="end at the last stop")
    _add_scenario_args(p, turns=False)
    p.set_defaults(func=cmd_tour)

    p = sub.add_parser("dispatch", help="simulate ride dispatch and compare policies")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--drivers", type=int, default=115)
    p.add_argument("--demand", type=float, default=4.0, help="peak ride requests per minute")
    p.add_argument("--start", type=float, default=7.0, help="hour the simulated window starts")
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--days", type=int, default=6, help="simulated mornings to average over")
    p.set_defaults(func=cmd_dispatch)

    p = sub.add_parser("export-web", help="export graph and hierarchies for the map demo")
    add_city(p)
    p.add_argument("--network", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="default web/data/<city>")
    p.add_argument("--raw", type=Path, default=None, help="raw OSM download, for rivers")
    p.add_argument(
        "--eta-trips",
        type=int,
        default=None,
        help="simulated trips to train the ETA model on (0 to skip; needs scikit-learn)",
    )
    p.add_argument(
        "--dispatch-days",
        type=int,
        default=None,
        help="simulated mornings for the dispatch comparison (0 to skip; needs scikit-learn)",
    )
    p.add_argument("--name", default=None)
    p.add_argument("--tfl", type=Path, help="saved TfL disruptions (from `live --out`)")
    p.set_defaults(func=cmd_export_web)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
