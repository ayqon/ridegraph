from __future__ import annotations

import json

import pytest

from fastroute import cities
from fastroute.__main__ import main
from fastroute.synthetic import grid_network
from fastroute.web_export import export, write_city_index


@pytest.mark.parametrize("city", list(cities.CITIES.values()), ids=list(cities.CITIES))
def test_every_named_place_is_inside_the_city(city) -> None:
    s, w, n, e = city.bbox
    assert s < n and w < e

    def inside(lat: float, lon: float) -> bool:
        return s <= lat <= n and w <= lon <= e

    assert all(inside(lat, lon) for _, lat, lon in city.landmarks)
    assert all(inside(*p) for p in city.default_trip)
    assert all(inside(*p) for p in city.default_tour)
    assert all(inside(lat, lon) for _, _, lat, lon, _ in city.demand_centres)
    assert {k for _, k, *_ in city.demand_centres} == {"station", "office", "nightlife"}
    assert 2 <= len(city.default_tour) <= 12


def test_nicosia_stays_south_of_the_buffer_zone() -> None:
    # The Green Line runs through the walled city at about 35.176 N.
    assert cities.NICOSIA.bbox[2] < 35.176


def test_unknown_city_is_rejected() -> None:
    with pytest.raises(ValueError, match="choose from"):
        cities.get("atlantis")


def test_export_carries_the_city_and_only_its_own_extras(tmp_path) -> None:
    network = grid_network(14, 14, seed=1)
    london = export(network, tmp_path / "london", city=cities.LONDON, log=lambda m: None)
    nyc = export(network, tmp_path / "new-york", city=cities.NEW_YORK, log=lambda m: None)
    assert london["city"]["key"] == "london" and "charge" in london and "live" in london
    assert nyc["city"]["key"] == "new-york" and "charge" not in nyc and "live" not in nyc
    assert nyc["city"]["landmarks"][0]["name"] == "Times Square"
    # Driving on the right swaps which turn crosses oncoming traffic.
    kinds = london["turns"]["kinds"]
    left, right = kinds.index("left"), kinds.index("right")
    assert london["turns"]["seconds"][left] < london["turns"]["seconds"][right]
    assert nyc["turns"]["seconds"][left] > nyc["turns"]["seconds"][right]
    index = write_city_index(tmp_path)
    assert [c["key"] for c in index] == ["london", "new-york"]
    assert json.loads((tmp_path / "cities.json").read_text()) == index


def test_cli_defaults_follow_the_city(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    grid_network(10, 10, seed=2).save(tmp_path / "data" / "nicosia.npz")
    assert (
        main(["export-web", "--city", "nicosia", "--eta-trips", "0", "--dispatch-days", "0"]) == 0
    )
    manifest = json.loads((tmp_path / "web/data/nicosia/manifest.json").read_text())
    assert manifest["name"] == "Nicosia" and manifest["city"]["drive_on_left"] is True
    assert json.loads((tmp_path / "web/data/cities.json").read_text())[0]["key"] == "nicosia"
