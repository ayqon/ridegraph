from __future__ import annotations

import pytest

from ridegraph.speeds import (
    DEFAULT_SPEED_KPH,
    LIMIT_REALISATION,
    MPH_TO_KPH,
    free_flow_speed_kph,
    parse_maxspeed,
    travel_time_s,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("50", 50.0),
        ("30 mph", 30 * MPH_TO_KPH),
        ("20mph", 20 * MPH_TO_KPH),
        ("48 km/h", 48.0),
        ("30 mph; 20 mph", 20 * MPH_TO_KPH),
        ("GB:nsl_single", 60 * MPH_TO_KPH),
        ("signals", None),
        ("none", None),
        ("", None),
        (None, None),
        ("0", None),
    ],
)
def test_parse_maxspeed(raw: str | None, expected: float | None) -> None:
    result = parse_maxspeed(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_default_speed_used_when_untagged() -> None:
    assert free_flow_speed_kph("residential", None) == DEFAULT_SPEED_KPH["residential"]


def test_posted_limit_is_discounted() -> None:
    assert free_flow_speed_kph("primary", 48.0) == pytest.approx(48.0 * LIMIT_REALISATION)


def test_travel_time() -> None:
    assert travel_time_s(1000, 36) == pytest.approx(100)


def test_vectorised_speeds_match_scalar_rule() -> None:
    import math

    import numpy as np

    from ridegraph.graph import free_flow_speeds_kph
    from ridegraph.speeds import ROAD_CLASSES
    from ridegraph.synthetic import grid_network

    network = grid_network(8, 8, seed=1)
    limits = np.where(np.arange(network.num_edges) % 3 == 0, 48.0, np.nan).astype(np.float32)
    network = type(network)(**{**network.__dict__, "maxspeed_kph": limits})
    vector = free_flow_speeds_kph(network)
    for i in range(network.num_edges):
        limit = float(limits[i])
        expected = free_flow_speed_kph(
            ROAD_CLASSES[network.road_class[i]], None if math.isnan(limit) else limit
        )
        assert vector[i] == pytest.approx(expected)
