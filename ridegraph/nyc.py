"""Real New York taxi trips on the Manhattan road graph.

The New York City Taxi and Limousine Commission (TLC) publishes every yellow
taxi trip: when it started and ended, its distance, and the taxi zone it
started and ended in. Zones are the finest location the public data gives,
so each trip is placed on the graph at a representative junction of its
pickup and drop-off zones. That uncertainty is part of every result built on
this module and is reported alongside it.

Sources:

* trip records: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
* taxi zones: the ``taxi_zones.zip`` shapefile published on the same page.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .graph import RoadNetwork


@dataclass(frozen=True)
class Zone:
    id: int
    borough: str
    name: str
    rings: tuple[np.ndarray, ...]  # each (k, 2): lon, lat


def read_zones(path: str | Path) -> list[Zone]:
    """Taxi zones from TLC's ``taxi_zones.zip`` shapefile, in longitude/latitude.

    The shapefile is in a projected coordinate system (New York State Plane);
    its ``.prj`` file says which, and every ring is converted to WGS84.
    """
    import io

    import shapefile
    from pyproj import CRS, Transformer

    with zipfile.ZipFile(path) as zf:

        def member(suffix: str) -> io.BytesIO:
            name = next(n for n in zf.namelist() if n.lower().endswith(suffix))
            return io.BytesIO(zf.read(name))

        crs = CRS.from_wkt(member(".prj").read().decode())
        reader = shapefile.Reader(shp=member(".shp"), shx=member(".shx"), dbf=member(".dbf"))
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        fields = [f[0].lower() for f in reader.fields[1:]]
        zones = []
        for shape_rec in reader.iterShapeRecords():
            record = dict(zip(fields, shape_rec.record, strict=True))
            shape = shape_rec.shape
            points = np.asarray(shape.points, dtype=np.float64)
            bounds = list(shape.parts) + [len(points)]
            rings = []
            for start, end in zip(bounds[:-1], bounds[1:], strict=True):
                lon, lat = to_wgs84.transform(points[start:end, 0], points[start:end, 1])
                rings.append(np.column_stack([lon, lat]))
            zones.append(
                Zone(
                    id=int(record["locationid"]),
                    borough=str(record.get("borough", "")),
                    name=str(record.get("zone", "")),
                    rings=tuple(rings),
                )
            )
    return zones


def points_in_rings(x: np.ndarray, y: np.ndarray, rings: tuple[np.ndarray, ...]) -> np.ndarray:
    """Even-odd point-in-polygon test over all rings, so holes are handled.

    A point is inside if a ray from it crosses the rings' edges an odd number
    of times; a hole's ring flips it back to outside.
    """
    inside = np.zeros(len(x), dtype=bool)
    for ring in rings:
        rx, ry = ring[:, 0], ring[:, 1]
        box = (x >= rx.min()) & (x <= rx.max()) & (y >= ry.min()) & (y <= ry.max())
        if not box.any():
            continue
        px, py = x[box], y[box]
        crossing = np.zeros(len(px), dtype=bool)
        x0, y0 = rx, ry
        x1, y1 = np.roll(rx, -1), np.roll(ry, -1)
        for ax, ay, bx, by in zip(x0, y0, x1, y1, strict=True):
            if ay == by:
                continue
            straddles = (ay > py) != (by > py)
            at = ax + (py - ay) * (bx - ax) / (by - ay)
            crossing ^= straddles & (px < at)
        inside[box] ^= crossing
    return inside


def zone_of_nodes(network: RoadNetwork, zones: list[Zone]) -> np.ndarray:
    """Taxi zone id of every junction, or 0 for junctions outside every zone."""
    out = np.zeros(network.num_nodes, dtype=np.int32)
    for zone in zones:
        hit = points_in_rings(network.lon, network.lat, zone.rings) & (out == 0)
        out[hit] = zone.id
    return out


def zone_anchors(network: RoadNetwork, zone_of_node: np.ndarray) -> dict[int, int]:
    """One representative junction per zone: the one nearest the zone's middle.

    "Middle" is the mean position of the zone's junctions, which sits where
    its streets are rather than at the centre of a park or a stretch of river.
    """
    anchors: dict[int, int] = {}
    for zone in np.unique(zone_of_node[zone_of_node > 0]).tolist():
        nodes = np.flatnonzero(zone_of_node == zone)
        lat, lon = network.lat[nodes], network.lon[nodes]
        scale = np.cos(np.radians(lat.mean()))
        d = (lat - lat.mean()) ** 2 + ((lon - lon.mean()) * scale) ** 2
        anchors[int(zone)] = int(nodes[np.argmin(d)])
    return anchors
