"""Turn restrictions: storing them on the network, and obeying them."""

from __future__ import annotations

import numpy as np
import pytest

from ridegraph.graph import RoadNetwork

from .conftest import make_network


def with_bans(network: RoadNetwork, bans: list[tuple[int, int, int]]) -> RoadNetwork:
    return RoadNetwork(**{**network.__dict__, "banned_turns": np.array(bans, dtype=np.int32)})


def line() -> RoadNetwork:
    return make_network([(0, 1, 100), (1, 2, 100), (2, 1, 100), (1, 0, 100)])


def test_no_bans_by_default():
    banned = line().banned_turns
    assert banned is not None and banned.shape == (0, 3)


def test_bans_survive_save_and_load(tmp_path):
    network = with_bans(line(), [(0, 1, 2)])
    network.save(tmp_path / "net.npz")
    loaded = RoadNetwork.load(tmp_path / "net.npz")
    assert loaded.banned_turns is not None
    assert loaded.banned_turns.tolist() == [[0, 1, 2]]


@pytest.mark.parametrize("bans", [[(0, 1, 9)], [(-1, 1, 2)]])
def test_bans_must_point_at_real_nodes(bans):
    with pytest.raises(ValueError):
        with_bans(line(), bans)


def test_bans_must_be_triples():
    with pytest.raises(ValueError):
        RoadNetwork(**{**line().__dict__, "banned_turns": np.zeros((1, 2), dtype=np.int32)})
