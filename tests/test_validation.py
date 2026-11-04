from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from fastroute.synthetic import grid_network  # noqa: E402
from fastroute.validation import REFERENCES, markdown, simulated_speeds  # noqa: E402


def test_simulated_city_is_slower_at_peak_and_slower_than_the_router() -> None:
    rows = simulated_speeds(grid_network(24, 24, seed=2), hours=(3.0, 8.0), trips=40, min_km=0.5)
    night, peak = rows
    assert peak.simulated_kmh < night.simulated_kmh
    for r in rows:
        # Lights, turns and hotspots only ever add time to the router's estimate.
        assert r.simulated_kmh < r.router_kmh
    table = markdown(rows)
    assert "03:00" in table and "08:00" in table
    assert all(ref.what in table for ref in REFERENCES)


def test_reference_figures_are_converted_from_mph() -> None:
    city = next(r for r in REFERENCES if r.what.startswith("City of London"))
    assert city.kmh == pytest.approx(12.39, abs=0.01)
