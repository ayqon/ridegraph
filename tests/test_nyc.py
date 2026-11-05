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
