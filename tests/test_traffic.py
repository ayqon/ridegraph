from __future__ import annotations

import numpy as np
import pytest

from ridegraph.dijkstra import dijkstra
from ridegraph.graph import Graph, free_flow_cost
from ridegraph.speeds import CLASS_INDEX
from ridegraph.traffic import DEFAULT_PROFILE, TrafficProfile, parse_time, traffic_cost

from .conftest import make_network


def test_congestion_shape() -> None:
    p = DEFAULT_PROFILE
    assert p.congestion(3) < 0.05
    assert p.congestion(8) == 1.0
    assert p.congestion(8) > p.congestion(13) > p.congestion(3)
    assert p.congestion(8, weekend=True) < p.congestion(8)
    # Linear interpolation and wrap-around at midnight.
    assert p.congestion(8.5) == pytest.approx((p.weekday[8] + p.weekday[9]) / 2)
    assert p.congestion(23.5) == pytest.approx((p.weekday[23] + p.weekday[0]) / 2)


def test_main_roads_slow_more_than_side_streets_at_peak() -> None:
    f = DEFAULT_PROFILE.speed_factors(8)
    assert f[CLASS_INDEX["primary"]] < f[CLASS_INDEX["residential"]] < 1.0
    assert np.all(DEFAULT_PROFILE.speed_factors(3) > 0.95)


def test_traffic_never_speeds_roads_up() -> None:
    network = make_network([(0, 1, 500, "primary"), (1, 2, 300, "residential")])
    base = free_flow_cost(network)
    for hour in np.arange(0, 24, 0.25):
        assert np.all(traffic_cost(float(hour))(network) >= base)


def test_rush_hour_changes_the_route() -> None:
    # Main road: 0 -> 1 -> 3, 1.5 km of primary. Back streets: 0 -> 2 -> 3,
    # 1.1 km of residential. Quiet roads favour the main road; at 8am it clogs.
    network = make_network(
        [
            (0, 1, 750, "primary"),
            (1, 3, 750, "primary"),
            (0, 2, 550, "residential"),
            (2, 3, 550, "residential"),
        ]
    )
    night = dijkstra(Graph(network, traffic_cost(3)), 0, 3)
    peak = dijkstra(Graph(network, traffic_cost(8)), 0, 3)
    assert night.path == [0, 1, 3]
    assert peak.path == [0, 2, 3]
    assert peak.seconds > night.seconds


def test_invalid_profiles_rejected() -> None:
    with pytest.raises(ValueError):
        TrafficProfile(weekday=(0.5,) * 23)
    with pytest.raises(ValueError):
        TrafficProfile(weekend=(1.5,) * 24)


@pytest.mark.parametrize(
    ("text", "hour"), [("8", 8.0), ("8.5", 8.5), ("17:30", 17.5), ("0:05", 1 / 12)]
)
def test_parse_time(text: str, hour: float) -> None:
    assert parse_time(text) == pytest.approx(hour)


def test_parse_time_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        parse_time("24:00")
