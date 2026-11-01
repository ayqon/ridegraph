from __future__ import annotations

from fastroute import bench
from fastroute.__main__ import main
from fastroute.ch import ContractionHierarchy
from fastroute.graph import Graph
from fastroute.synthetic import grid_network


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
