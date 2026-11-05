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


def test_trips_become_training_rows_with_their_real_times():
    from ridegraph.ch import ContractionHierarchy
    from ridegraph.eta import RouteContext
    from ridegraph.nyc import trip_dataset, zone_anchors
    from ridegraph.router import Router, Scenario
    from ridegraph.synthetic import grid_network

    network = grid_network(12, 12, seed=4)
    ctx = RouteContext(network)
    # Four zones: the quadrants of the grid.
    lat, lon = ctx.network.lat, ctx.network.lon
    north, east = lat > np.median(lat), lon > np.median(lon)
    zones = (1 + north.astype(int) * 2 + east.astype(int)).astype(np.int32)
    anchors = zone_anchors(ctx.network, zones)
    hierarchies = {
        h: ContractionHierarchy.build(Router(ctx.network, Scenario(hour=h)).graph)
        for h in (8.0, 13.0)
    }
    trips = trips_of(
        [
            ("2025-03-04T08:10", 600, 1, 4, 2.0),
            ("2025-03-04T08:40", 660, 1, 4, 2.0),  # same pair and preset: cached route
            ("2025-03-05T13:05", 420, 2, 3, 1.5),
        ]
    )
    data, used = trip_dataset(ctx, hierarchies, trips, anchors)
    assert used.tolist() == [0, 1, 2]
    assert data.observed.tolist() == [600.0, 660.0, 420.0]
    assert (data.graph_eta > 0).all()
    assert np.allclose(data.y, np.log(data.observed / data.graph_eta))
    assert np.array_equal(data.edges[0], data.edges[1])


def write_zone_file(path, cells: dict[int, tuple[float, float, float, float]]) -> None:
    """A TLC-style zone shapefile of rectangles given as (lon0, lat0, lon1, lat1)."""
    shapefile = pytest.importorskip("shapefile")
    pyproj = pytest.importorskip("pyproj")
    import zipfile

    crs = pyproj.CRS.from_epsg(2263)
    to_plane = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    base = path.parent / "taxi_zones"
    with shapefile.Writer(str(base), shapeType=shapefile.POLYGON) as w:
        w.field("LocationID", "N", decimal=0)
        w.field("zone", "C")
        w.field("borough", "C")
        for zone, (x0, y0, x1, y1) in cells.items():
            xs, ys = to_plane.transform([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
            w.poly([list(zip(xs, ys, strict=True))])
            w.record(zone, f"Zone {zone}", "Manhattan")
    (path.parent / "taxi_zones.prj").write_text(crs.to_wkt())
    with zipfile.ZipFile(path, "w") as zf:
        for ext in ("shp", "shx", "dbf", "prj"):
            zf.write(path.parent / f"taxi_zones.{ext}", f"taxi_zones.{ext}")


def write_month(path, month: str, zones: list[int], count: int, seed: int) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = np.random.default_rng(seed)
    start = np.datetime64(f"{month}-01T00:00:00", "us")
    offset = rng.integers(0, 27 * 86_400, count).astype("timedelta64[s]")
    pickup = start + offset
    km = rng.uniform(1.5, 6.0, count)
    duration = (km / 12.0 * 3600 * rng.uniform(0.8, 1.3, count)).astype("timedelta64[s]")
    pu = rng.choice(zones, count)
    do = rng.choice(zones, count)
    table = pa.table(
        {
            "tpep_pickup_datetime": pa.array(pickup),
            "tpep_dropoff_datetime": pa.array(pickup + duration),
            "trip_distance": km / 1.609344,
            "PULocationID": pu,
            "DOLocationID": do,
        }
    )
    pq.write_table(table, path)


def test_the_whole_study_runs_on_a_small_city(tmp_path):
    pytest.importorskip("sklearn")
    from ridegraph.nyc import eta_study
    from ridegraph.synthetic import grid_network

    network = grid_network(16, 16, seed=2)
    lat0, lat1 = float(network.lat.min()), float(network.lat.max())
    lon0, lon1 = float(network.lon.min()), float(network.lon.max())
    mid_lat, mid_lon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    pad = 0.001
    cells = {
        1: (lon0 - pad, lat0 - pad, mid_lon, mid_lat),
        2: (mid_lon, lat0 - pad, lon1 + pad, mid_lat),
        3: (lon0 - pad, mid_lat, mid_lon, lat1 + pad),
        4: (mid_lon, mid_lat, lon1 + pad, lat1 + pad),
    }
    zones = tmp_path / "taxi_zones.zip"
    write_zone_file(zones, cells)
    files = []
    for i, month in enumerate(("2025-01", "2025-02", "2025-03")):
        f = tmp_path / f"yellow_tripdata_{month}.parquet"
        write_month(f, month, [1, 2, 3, 4, 132], 900, seed=i)  # 132 is off the map
        files.append(f)

    study = eta_study(network, zones, files[:2], files[2], per_month=400, gnn_epochs=0)
    assert study.zones == 4
    assert study.train_months == ["2025-01", "2025-02"] and study.test_month == "2025-03"
    assert all(study.trips_used[m] <= 400 for m in study.trips_used)
    metrics = study.result.metrics
    assert "Gradient-boosted trees" in metrics and "Graph ETA (router only)" in metrics
    assert study.by_distance and study.by_distance[0]["trips"] > 0
