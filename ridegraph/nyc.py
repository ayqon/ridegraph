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


# -- trips ---------------------------------------------------------------------

TRIP_COLUMNS = (
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
)
KM_PER_MILE = 1.609344


@dataclass(frozen=True)
class Trips:
    """Taxi trips as plain arrays, one entry per trip.

    Times are New York local time, as TLC records them, in seconds since
    1970-01-01 00:00 of that clock.
    """

    pickup_s: np.ndarray  # int64
    duration_s: np.ndarray  # float64
    pickup_zone: np.ndarray  # int32
    dropoff_zone: np.ndarray  # int32
    distance_km: np.ndarray  # float64

    def __len__(self) -> int:
        return len(self.pickup_s)

    def take(self, index: np.ndarray) -> Trips:
        return Trips(
            self.pickup_s[index],
            self.duration_s[index],
            self.pickup_zone[index],
            self.dropoff_zone[index],
            self.distance_km[index],
        )

    @property
    def hour(self) -> np.ndarray:
        """Hour of day of each pickup, with minutes as a fraction."""
        return (self.pickup_s % 86_400) / 3600.0

    @property
    def weekend(self) -> np.ndarray:
        # 1970-01-01 was a Thursday, so day 2 after it was a Saturday.
        return ((self.pickup_s // 86_400 + 3) % 7) >= 5


def clean(
    trips: Trips,
    zones: set[int],
    *,
    month: str | None = None,
    min_s: float = 60.0,
    max_s: float = 3 * 3600.0,
    min_kmh: float = 1.0,
    max_kmh: float = 80.0,
) -> Trips:
    """Keep trips that start and end in different zones of the map and look real.

    TLC's own guidance is that a small share of records are wrong: zero or
    negative durations, meters left running, distances that imply impossible
    speeds, and trips dated outside the month the file covers. Those are
    dropped. Trips that start and end in the same zone are dropped too, since
    with zone-level locations there is no route to compare against.
    """
    zone_list = np.array(sorted(zones), dtype=np.int32)
    keep = np.isin(trips.pickup_zone, zone_list) & np.isin(trips.dropoff_zone, zone_list)
    keep &= trips.pickup_zone != trips.dropoff_zone
    keep &= (trips.duration_s >= min_s) & (trips.duration_s <= max_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        kmh = trips.distance_km / (trips.duration_s / 3600.0)
    keep &= (kmh >= min_kmh) & (kmh <= max_kmh)
    if month is not None:
        start = np.datetime64(f"{month}-01", "s").astype(np.int64)
        next_month = np.datetime64(month, "M") + np.timedelta64(1, "M")
        end = next_month.astype("datetime64[s]").astype(np.int64)
        keep &= (trips.pickup_s >= start) & (trips.pickup_s < end)
    return trips.take(np.flatnonzero(keep))


def read_trips(path: str | Path) -> Trips:
    """One month of TLC yellow taxi trip records from its Parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(TRIP_COLUMNS))

    def seconds(name: str) -> np.ndarray:
        col = table.column(name).to_numpy()
        return col.astype("datetime64[s]").astype(np.int64)

    pickup = seconds("tpep_pickup_datetime")
    dropoff = seconds("tpep_dropoff_datetime")
    return Trips(
        pickup_s=pickup,
        duration_s=(dropoff - pickup).astype(np.float64),
        pickup_zone=table.column("PULocationID").to_numpy().astype(np.int32),
        dropoff_zone=table.column("DOLocationID").to_numpy().astype(np.int32),
        distance_km=table.column("trip_distance").to_numpy().astype(np.float64) * KM_PER_MILE,
    )
