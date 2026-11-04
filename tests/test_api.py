from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ridegraph.api import RoutingService, create_app, parse_time  # noqa: E402
from ridegraph.dijkstra import one_to_all  # noqa: E402
from ridegraph.graph import Graph, csr_sorted  # noqa: E402
from ridegraph.synthetic import grid_network  # noqa: E402
from ridegraph.traffic import traffic_cost  # noqa: E402

# The synthetic grid sits around Holborn, straddling the Congestion Charge
# zone's northern edge.
A, B = "51.5150,-0.1200", "51.5350,-0.0870"


@pytest.fixture(scope="module")
def network():
    return csr_sorted(grid_network(24, 24, seed=5, oneway_share=0.2))


@pytest.fixture(scope="module")
def client(network):
    return TestClient(create_app(RoutingService(network)))


def point(text: str) -> tuple[float, float]:
    lat, lon = text.split(",")
    return float(lat), float(lon)


def test_health(client, network) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["nodes"] == network.num_nodes
    assert body["learned_eta"] is False


@pytest.mark.parametrize("at", ["03:00", "08:15", "17:40"])
def test_route_is_the_quickest_at_any_time(client, network, at) -> None:
    body = client.get("/route", params={"from": A, "to": B, "at": at}).json()
    quickest = body["features"][0]
    assert quickest["geometry"]["type"] == "LineString"
    hour = parse_time(at)
    hour = round(hour * 4) / 4  # served at the nearest quarter-hour
    graph = Graph(network, traffic_cost(hour))
    s, t = graph.nearest_node(*point(A)), graph.nearest_node(*point(B))
    assert quickest["properties"]["seconds"] == pytest.approx(one_to_all(graph, s)[t], abs=0.06)
    coords = quickest["geometry"]["coordinates"]
    assert coords[0] == pytest.approx([network.lon[s], network.lat[s]], abs=1e-6)


def test_alternatives_and_avoiding_the_zone(client) -> None:
    body = client.get(
        "/route", params={"from": A, "to": B, "at": "08:00", "alternatives": 2}
    ).json()
    times = [f["properties"]["seconds"] for f in body["features"]]
    assert times == sorted(times) and len(times) >= 2
    assert times[-1] <= 1.3 * times[0] + 1e-6
    # Start and end outside the zone: a route that stays out is offered.
    outside_a, outside_b = "51.5340,-0.1180", "51.5350,-0.0870"
    body = client.get(
        "/route", params={"from": outside_a, "to": outside_b, "avoid_charge": True}
    ).json()
    names = [f["properties"]["name"] for f in body["features"]]
    assert "avoiding the zone" in names
    avoid = body["features"][names.index("avoiding the zone")]
    assert avoid["properties"]["charge"]["enters_zone"] is False
    # Starting inside the zone, it cannot be avoided.
    inside = client.get("/route", params={"from": A, "to": B, "avoid_charge": True})
    assert inside.status_code == 422 and "zone" in inside.json()["detail"]


def test_charge_follows_the_clock(client) -> None:
    night = client.get("/route", params={"from": A, "to": B, "at": "03:00"}).json()
    day = client.get("/route", params={"from": A, "to": B, "at": "10:00"}).json()
    n, d = night["features"][0]["properties"], day["features"][0]["properties"]
    if d["charge"]["enters_zone"]:
        assert d["charge"]["amount_gbp"] == 18.0 and n["charge"]["amount_gbp"] == 0.0


def test_reach_and_tour(client) -> None:
    body = client.get("/reach", params={"from": A, "minutes": "2,4", "at": "08:00"}).json()
    areas = [b["area_km2"] for b in body["bands"]]
    assert areas == sorted(areas) and areas[0] > 0
    stops = [[51.515, -0.12], [51.53, -0.1], [51.52, -0.09], [51.535, -0.115]]
    body = client.post("/tour", json={"stops": stops, "at": "13:00"}).json()
    assert sorted(body["order"]) == [0, 1, 2, 3] and body["order"][0] == 0
    assert body["seconds"][body["method"]] == min(body["seconds"].values())


@pytest.mark.parametrize(
    "params",
    [
        {"from": "10,10", "to": B},
        {"from": "not-a-point", "to": B},
        {"from": A, "to": B, "at": "25:00"},
        {"from": A, "to": B, "alternatives": 5},
    ],
)
def test_bad_requests_are_rejected_clearly(client, params) -> None:
    response = client.get("/route", params=params)
    assert response.status_code == 422
    assert response.json()["detail"]


def test_learned_eta_is_reported_when_a_model_is_loaded(network) -> None:
    from ridegraph.eta import train

    model = train(network, trips=600, seed=1, gnn_epochs=0).model_json
    client = TestClient(create_app(RoutingService(network, model)))
    props = client.get("/route", params={"from": A, "to": B}).json()["features"][0]["properties"]
    eta = props["learned_eta"]
    assert eta and eta["seconds"] > 0
    low, high = eta["window_seconds"]
    assert low <= eta["seconds"] <= high and math.isfinite(high)
    assert np.isfinite(props["seconds"])
