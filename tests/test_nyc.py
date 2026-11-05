"""New York taxi data: zones, trips and their placement on the graph."""

from __future__ import annotations

import numpy as np
import pytest

from ridegraph.nyc import Zone, points_in_rings, zone_anchors, zone_of_nodes

from .conftest import make_network


def rect(x0: float, y0: float, w: float, h: float) -> np.ndarray:
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]])


def square(x0: float, y0: float, size: float) -> np.ndarray:
    return rect(x0, y0, size, size)


def test_point_in_polygon_with_a_hole():
    outer, hole = square(0, 0, 10), square(4, 4, 2)
    x = np.array([1.0, 5.0, 11.0, 9.5])
    y = np.array([1.0, 5.0, 5.0, 9.5])
    assert points_in_rings(x, y, (outer, hole)).tolist() == [True, False, False, True]


def test_every_junction_gets_its_zone():
    # Two zones side by side; junctions at lon -0.1 + 0.001 * i, lat 51.5.
    coords = [(51.5, -0.1 + 0.001 * i) for i in range(6)]
    network = make_network([(i, i + 1, 70) for i in range(5)], coords=coords)
    west = Zone(1, "Manhattan", "West", (rect(-0.1005, 51.49, 0.0030, 0.02),))
    east = Zone(2, "Manhattan", "East", (rect(-0.0975, 51.49, 0.0030, 0.02),))
    zones = zone_of_nodes(network, [west, east])
    assert zones.tolist() == [1, 1, 1, 2, 2, 2]
    anchors = zone_anchors(network, zones)
    assert anchors == {1: 1, 2: 4}


def test_junctions_outside_every_zone_get_zero():
    coords = [(51.5, -0.1), (51.6, -0.1)]
    network = make_network([(0, 1, 100)], coords=coords)
    zone = Zone(1, "Manhattan", "Only", (square(-0.11, 51.49, 0.02),))
    assert zone_of_nodes(network, [zone]).tolist() == [1, 0]


@pytest.mark.parametrize("hole_first", [False, True])
def test_ring_order_does_not_matter(hole_first):
    outer, hole = square(0, 0, 10), square(4, 4, 2)
    rings = (hole, outer) if hole_first else (outer, hole)
    assert points_in_rings(np.array([5.0]), np.array([5.0]), rings).tolist() == [False]


def test_reading_the_zone_shapefile(tmp_path):
    shapefile = pytest.importorskip("shapefile")
    pyproj = pytest.importorskip("pyproj")
    import zipfile

    # One square zone in New York State Plane feet, as TLC publishes them.
    crs = pyproj.CRS.from_epsg(2263)
    to_plane = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon, lat = [-73.99, -73.98, -73.98, -73.99, -73.99], [40.75, 40.75, 40.76, 40.76, 40.75]
    x, y = to_plane.transform(lon, lat)
    base = tmp_path / "taxi_zones"
    with shapefile.Writer(str(base), shapeType=shapefile.POLYGON) as w:
        for name in ("OBJECTID", "Shape_Leng", "Shape_Area", "zone", "LocationID", "borough"):
            w.field(name, "C" if name in ("zone", "borough") else "N", decimal=0)
        w.poly([list(zip(x, y, strict=True))])
        w.record(1, 0, 0, "Midtown Test", 161, "Manhattan")
    (tmp_path / "taxi_zones.prj").write_text(crs.to_wkt())
    archive = tmp_path / "taxi_zones.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for ext in ("shp", "shx", "dbf", "prj"):
            zf.write(tmp_path / f"taxi_zones.{ext}", f"taxi_zones/taxi_zones.{ext}")

    from ridegraph.nyc import read_zones

    (zone,) = read_zones(archive)
    assert (zone.id, zone.borough, zone.name) == (161, "Manhattan", "Midtown Test")
    ring = zone.rings[0]
    assert np.allclose(ring[:, 0], lon, atol=1e-6) and np.allclose(ring[:, 1], lat, atol=1e-6)


# -- trips ---------------------------------------------------------------------


def trips_of(rows: list[tuple[str, float, int, int, float]]):
    from ridegraph.nyc import Trips

    pickup = np.array([np.datetime64(r[0], "s").astype(np.int64) for r in rows])
    return Trips(
        pickup_s=pickup,
        duration_s=np.array([r[1] for r in rows], dtype=np.float64),
        pickup_zone=np.array([r[2] for r in rows], dtype=np.int32),
        dropoff_zone=np.array([r[3] for r in rows], dtype=np.int32),
        distance_km=np.array([r[4] for r in rows], dtype=np.float64),
    )


def test_cleaning_drops_records_that_cannot_be_real():
    from ridegraph.nyc import clean

    trips = trips_of(
        [
            ("2025-03-04T08:15", 900, 161, 230, 3.0),  # fine
            ("2025-03-04T08:15", 900, 161, 161, 1.0),  # same zone
            ("2025-03-04T08:15", 900, 161, 1, 3.0),  # ends off the map (Newark)
            ("2025-03-04T08:15", 20, 161, 230, 0.1),  # too short
            ("2025-03-04T08:15", 900, 161, 230, 60.0),  # 240 km/h
            ("2025-02-28T23:50", 900, 161, 230, 3.0),  # outside the month
        ]
    )
    kept = clean(trips, {161, 230}, month="2025-03")
    assert len(kept) == 1 and kept.pickup_zone.tolist() == [161]


def test_hour_and_weekend_come_from_the_local_pickup_time():
    trips = trips_of([("2025-03-04T08:30", 900, 1, 2, 3.0), ("2025-03-08T23:00", 900, 1, 2, 3.0)])
    assert trips.hour.tolist() == [8.5, 23.0]
    assert trips.weekend.tolist() == [False, True]  # a Tuesday, then a Saturday


def test_reading_a_trip_file(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from ridegraph.nyc import read_trips

    table = pa.table(
        {
            "VendorID": [2],
            "tpep_pickup_datetime": pa.array([np.datetime64("2025-03-04T08:15:00", "us")]),
            "tpep_dropoff_datetime": pa.array([np.datetime64("2025-03-04T08:30:30", "us")]),
            "trip_distance": [2.0],
            "PULocationID": [161],
            "DOLocationID": [230],
        }
    )
    pq.write_table(table, tmp_path / "yellow.parquet")
    trips = read_trips(tmp_path / "yellow.parquet")
    assert trips.duration_s.tolist() == [930.0]
    assert trips.hour.tolist() == [8.25]
    assert trips.distance_km[0] == pytest.approx(3.218688)
