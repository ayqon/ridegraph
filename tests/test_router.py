from __future__ import annotations

import pytest

from fastroute.__main__ import main
from fastroute.ch import ContractionHierarchy
from fastroute.router import Router, Scenario
from fastroute.synthetic import grid_network

# Opposite corners of the synthetic grid (it sits around Holborn).
A, B = "51.515,-0.12", "51.5249,-0.1041"


def test_scenario_labels_and_tags() -> None:
    assert Scenario().label() == "free flow"
    assert Scenario().file_tag() == ""
    s = Scenario(hour=8.5, turns=True)
    assert s.label() == "08:30 weekday, with turn costs"
    assert s.file_tag() == "0830wd-turns"
    assert Scenario(hour=17.5, weekend=True).file_tag() == "1730we"


def test_trip_is_expressed_in_intersections() -> None:
    network = grid_network(12, 12, seed=1)
    for scenario in (Scenario(), Scenario(hour=8, turns=True)):
        router = Router(network, scenario)
        trip = router.trip(0, network.num_nodes - 1)
        assert trip.intersections[0] == 0
        assert trip.intersections[-1] == network.num_nodes - 1
        assert trip.metres > 0


def test_rush_hour_with_turns_is_slower_than_free_flow() -> None:
    network = grid_network(12, 12, seed=2)
    # Interior points (row 1, col 1) to (row 10, col 10): any route must turn
    # at a real junction. Corner-to-corner routes can turn only at the grid's
    # corners, which are bends between two roads and rightly cost nothing.
    free = Router(network).trip(13, 130)
    peak = Router(network, Scenario(hour=8, turns=True)).trip(13, 130)
    assert peak.seconds > free.seconds
    assert sum(peak.manoeuvres.values()) >= 1


def test_ch_matches_search_under_scenario() -> None:
    network = grid_network(10, 10, seed=3, oneway_share=0.2)
    router = Router(network, Scenario(hour=17.5, turns=True))
    ch = ContractionHierarchy.build(router.graph)
    for s, t in [(0, 99), (45, 3), (99, 0), (12, 12)]:
        assert router.trip(s, t, "ch", ch).seconds == pytest.approx(
            router.trip(s, t, "dijkstra").seconds
        )
    with pytest.raises(ValueError):
        router.trip(0, 1, "ch")


def test_cli_scenarios_end_to_end(tmp_path, capsys) -> None:
    network = tmp_path / "grid.npz"
    grid_network(10, 10, seed=4).save(network)
    net = ["--network", str(network)]

    # A hierarchy is specific to its scenario, and the error says how to build it.
    with pytest.raises(SystemExit, match="contract --at 08:30 --turns"):
        main(["route", A, B, *net, "--at", "08:30", "--turns"])
    assert main(["contract", *net, "--at", "08:30", "--turns"]) == 0
    assert (tmp_path / "grid.0830wd-turns.ch.npz").exists()

    for algorithm in ("ch", "astar", "dijkstra"):
        assert main(["route", A, B, *net, "--at", "08:30", "--turns", "-a", algorithm]) == 0
    out = capsys.readouterr().out
    times = {line.split(": ")[1].split(",")[0] for line in out.splitlines() if " min " in line}
    assert len(times) == 1
    assert "08:30 weekday, with turn costs" in out

    assert main(["bench", *net, "--at", "08:30", "--turns", "--queries", "10"]) == 0


def test_cli_eta(tmp_path, capsys) -> None:
    network = tmp_path / "grid.npz"
    grid_network(10, 10, seed=5).save(network)
    assert main(["eta", A, B, "--network", str(network), "--times", "03:00", "08:00"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("03:00 weekday") and "reference route" in lines[0]
    with pytest.raises(SystemExit):
        main(["eta", A, B, "--network", str(network), "--times", "25:00"])
