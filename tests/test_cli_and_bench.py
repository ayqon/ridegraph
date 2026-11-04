from __future__ import annotations

from ridegraph import bench
from ridegraph.__main__ import main
from ridegraph.ch import ContractionHierarchy
from ridegraph.graph import Graph
from ridegraph.synthetic import grid_network


def test_bench_rows_and_markdown() -> None:
    graph = Graph(grid_network(15, 15, seed=2))
    rows = bench.run(graph, ContractionHierarchy.build(graph), queries=20)
    assert [r.algorithm for r in rows] == [
        "Dijkstra",
        "Bidirectional Dijkstra",
        "A*",
        "Contraction Hierarchies",
    ]
    assert rows[0].speedup_settled == 1.0
    table = bench.to_markdown(rows)
    assert table.count("\n") == len(rows) + 1


def test_cli_end_to_end(tmp_path, capsys) -> None:
    network_path = tmp_path / "grid.npz"
    grid_network(12, 12, seed=4).save(network_path)

    assert main(["info", "--network", str(network_path)]) == 0
    assert main(["contract", "--network", str(network_path)]) == 0
    assert (tmp_path / "grid.ch.npz").exists()

    # Opposite corners of the grid (around Holborn).
    args = ["route", "51.515,-0.12", "51.5249,-0.1041", "--network", str(network_path)]
    for algorithm in ("ch", "astar", "bidirectional", "dijkstra"):
        assert main([*args, "-a", algorithm]) == 0
    out = capsys.readouterr().out
    times = [line.split(":")[1].split(",")[0] for line in out.splitlines() if " min " in line]
    assert len(times) == 4 and len(set(times)) == 1  # all algorithms agree

    assert main(["bench", "--network", str(network_path), "--queries", "10"]) == 0


def test_cli_falls_back_without_a_hierarchy_and_rejects_off_map_points(tmp_path, capsys) -> None:
    import pytest

    network_path = tmp_path / "grid.npz"
    grid_network(12, 12, seed=4).save(network_path)
    args = ["--network", str(network_path)]
    assert main(["route", "51.515,-0.12", "51.5249,-0.1041", "--at", "08:00", *args]) == 0
    captured = capsys.readouterr()
    assert " min " in captured.out
    assert f"contract --network {network_path} --at 08:00" in captured.err
    with pytest.raises(SystemExit, match="from the nearest road"):
        main(["route", "10,10", "51.5249,-0.1041", *args])


def test_cli_reach_and_tour(tmp_path, capsys) -> None:
    network_path = tmp_path / "grid.npz"
    grid_network(12, 12, seed=4).save(network_path)
    args = ["--network", str(network_path)]
    assert main(["reach", "51.515,-0.12", "--bands", "2", "4", *args]) == 0
    assert main(["tour", "51.515,-0.12", "51.5249,-0.1041", *args]) == 0
    out = capsys.readouterr().out
    assert "1 stop from the depot" in out
    assert main(["tour", "51.515,-0.12", "51.5249,-0.1041", "51.52,-0.11", "--open", *args]) == 0
    assert "Optimal (Held-Karp)" in capsys.readouterr().out
