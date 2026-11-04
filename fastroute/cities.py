"""The cities the project is built for, and what is specific to each.

Everything else (parsing, routing, learning, dispatch, the browser) is the
same code for every city; a city only supplies its area, the places a person
would name on the map, where ride demand concentrates, which side of the
road traffic drives on, and which local extras apply (London's Congestion
Charge and TfL's live disruption feed).

The time-of-day traffic pattern is London's for every city. It is a
reasonable shape for any large city's weekday, but only London's has been
checked against published speeds (``validation.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

Place = tuple[str, float, float]  # name, lat, lon
DemandCentre = tuple[str, str, float, float, float]  # name, kind, lat, lon, strength


@dataclass(frozen=True)
class City:
    key: str
    name: str
    bbox: tuple[float, float, float, float]  # south, west, north, east
    landmarks: tuple[Place, ...]
    default_trip: tuple[tuple[float, float], tuple[float, float]]
    default_tour: tuple[tuple[float, float], ...]  # depot first
    demand_centres: tuple[DemandCentre, ...]
    drive_on_left: bool = True
    congestion_charge: bool = False
    tfl_disruptions: bool = False
    eta_trips: int = 20_000
    dispatch_days: int = 6
    notes: tuple[str, ...] = field(default_factory=tuple)
    file_stem: str = ""  # defaults to the key

    @property
    def raw_path(self) -> str:
        return f"data/raw/{self.file_stem or self.key}.json.gz"

    @property
    def network_path(self) -> str:
        return f"data/{self.file_stem or self.key}.npz"

    def browser_info(self) -> dict[str, object]:
        """What the map needs to know about the city."""
        return {
            "key": self.key,
            "name": self.name,
            "landmarks": [{"name": n, "lat": la, "lon": lo} for n, la, lo in self.landmarks],
            "default_trip": [list(p) for p in self.default_trip],
            "default_tour": [list(p) for p in self.default_tour],
            "drive_on_left": self.drive_on_left,
            "notes": list(self.notes),
        }


LONDON = City(
    key="london",
    name="Central London",
    bbox=(51.485, -0.205, 51.540, -0.065),
    landmarks=(
        ("Paddington", 51.5154, -0.1755),
        ("Marble Arch", 51.5131, -0.1589),
        ("Oxford Circus", 51.5152, -0.1419),
        ("Euston", 51.5282, -0.1337),
        ("King's Cross", 51.5308, -0.1238),
        ("Angel", 51.5322, -0.1058),
        ("Holborn", 51.5174, -0.1200),
        ("Piccadilly Circus", 51.5101, -0.1340),
        ("Trafalgar Square", 51.5080, -0.1281),
        ("Westminster", 51.5007, -0.1246),
        ("Waterloo", 51.5033, -0.1145),
        ("Bank", 51.5133, -0.0890),
        ("Liverpool Street", 51.5178, -0.0823),
        ("London Bridge", 51.5055, -0.0865),
        ("Elephant & Castle", 51.4946, -0.1005),
        ("Victoria", 51.4952, -0.1441),
        ("Hyde Park Corner", 51.5027, -0.1527),
        ("South Kensington", 51.4941, -0.1738),
        ("Notting Hill Gate", 51.5094, -0.1967),
        ("Vauxhall", 51.4861, -0.1253),
        ("Aldgate", 51.5143, -0.0755),
        ("Shoreditch", 51.5255, -0.0790),
        ("Covent Garden", 51.5117, -0.1240),
    ),
    default_trip=((51.5152, -0.1419), (51.5033, -0.1145)),  # Oxford Circus to Waterloo
    default_tour=(
        (51.5174, -0.1200),  # Holborn
        (51.5308, -0.1238),
        (51.5033, -0.1145),
        (51.5152, -0.1419),
        (51.5133, -0.0890),
        (51.4952, -0.1441),
        (51.5255, -0.0790),
        (51.5101, -0.1340),
        (51.5055, -0.0865),
    ),
    demand_centres=(
        ("King's Cross", "station", 51.5308, -0.1238, 1.2),
        ("Euston", "station", 51.5282, -0.1337, 0.9),
        ("Paddington", "station", 51.5154, -0.1755, 1.0),
        ("Liverpool Street", "station", 51.5178, -0.0823, 1.1),
        ("Waterloo", "station", 51.5033, -0.1145, 1.2),
        ("Victoria", "station", 51.4952, -0.1441, 1.0),
        ("London Bridge", "station", 51.5055, -0.0865, 0.9),
        ("Bank", "office", 51.5133, -0.0890, 1.3),
        ("Holborn", "office", 51.5174, -0.1200, 0.9),
        ("Westminster", "office", 51.5007, -0.1246, 0.7),
        ("Soho", "nightlife", 51.5136, -0.1340, 1.3),
        ("Covent Garden", "nightlife", 51.5117, -0.1240, 1.0),
        ("Shoreditch", "nightlife", 51.5255, -0.0790, 0.9),
        ("Notting Hill", "nightlife", 51.5094, -0.1967, 0.6),
    ),
    congestion_charge=True,
    tfl_disruptions=True,
    file_stem="central-london",  # the paths the project has always used
)

NICOSIA = City(
    key="nicosia",
    name="Nicosia",
    # The southern, government-controlled part of the city, up to the edge of
    # the buffer zone, so no route crosses the Green Line.
    bbox=(35.130, 33.325, 35.1735, 33.400),
    landmarks=(
        ("Eleftheria Square", 35.1703, 33.3605),
        ("Ledra Street", 35.1722, 33.3615),
        ("Famagusta Gate", 35.1726, 33.3727),
        ("Archbishop's Palace", 35.1730, 33.3667),
        ("Cyprus Museum", 35.1713, 33.3561),
        ("Solomou Square", 35.1696, 33.3589),
        ("Makariou Avenue", 35.1650, 33.3590),
        ("Agioi Omologites", 35.1600, 33.3570),
        ("Lykavittos", 35.1590, 33.3660),
        ("Akropoli", 35.1520, 33.3700),
        ("Engomi", 35.1620, 33.3400),
        ("Strovolos", 35.1450, 33.3470),
        ("Aglantzia", 35.1500, 33.3900),
    ),
    default_trip=((35.1620, 33.3400), (35.1726, 33.3727)),  # Engomi to Famagusta Gate
    default_tour=(
        (35.1703, 33.3605),  # Eleftheria Square
        (35.1620, 33.3400),
        (35.1450, 33.3470),
        (35.1520, 33.3700),
        (35.1726, 33.3727),
        (35.1600, 33.3570),
        (35.1500, 33.3900),
        (35.1713, 33.3561),
    ),
    demand_centres=(
        ("Solomou Square", "station", 35.1696, 33.3589, 1.0),
        ("Makariou Avenue", "office", 35.1650, 33.3590, 1.2),
        ("Engomi", "office", 35.1620, 33.3400, 0.8),
        ("Strovolos", "office", 35.1450, 33.3470, 0.8),
        ("Ledra Street", "nightlife", 35.1722, 33.3615, 1.2),
        ("Famagusta Gate", "nightlife", 35.1726, 33.3727, 0.8),
        ("Akropoli", "nightlife", 35.1520, 33.3700, 0.5),
    ),
    drive_on_left=True,
    eta_trips=8_000,
    dispatch_days=3,
    notes=("Traffic follows London's weekday pattern; it has not been checked for Nicosia.",),
)

NEW_YORK = City(
    key="new-york",
    name="Manhattan",
    bbox=(40.700, -74.020, 40.772, -73.945),
    landmarks=(
        ("Times Square", 40.7580, -73.9855),
        ("Grand Central", 40.7527, -73.9772),
        ("Penn Station", 40.7506, -73.9935),
        ("Empire State Building", 40.7484, -73.9857),
        ("Rockefeller Center", 40.7587, -73.9787),
        ("Bryant Park", 40.7536, -73.9832),
        ("Columbus Circle", 40.7681, -73.9819),
        ("Hudson Yards", 40.7536, -74.0019),
        ("Chelsea", 40.7465, -74.0014),
        ("Flatiron", 40.7411, -73.9897),
        ("Union Square", 40.7359, -73.9911),
        ("Washington Square", 40.7308, -73.9973),
        ("East Village", 40.7265, -73.9815),
        ("SoHo", 40.7233, -74.0030),
        ("Lower East Side", 40.7150, -73.9843),
        ("Chinatown", 40.7158, -73.9970),
        ("City Hall", 40.7128, -74.0060),
        ("World Trade Center", 40.7127, -74.0134),
        ("Wall Street", 40.7069, -74.0113),
        ("Long Island City", 40.7447, -73.9485),
        ("Williamsburg", 40.7081, -73.9571),
    ),
    default_trip=((40.7506, -73.9935), (40.7069, -74.0113)),  # Penn Station to Wall Street
    default_tour=(
        (40.7359, -73.9911),  # Union Square
        (40.7580, -73.9855),
        (40.7527, -73.9772),
        (40.7465, -74.0014),
        (40.7233, -74.0030),
        (40.7069, -74.0113),
        (40.7150, -73.9843),
        (40.7265, -73.9815),
        (40.7681, -73.9819),
    ),
    demand_centres=(
        ("Penn Station", "station", 40.7506, -73.9935, 1.3),
        ("Grand Central", "station", 40.7527, -73.9772, 1.3),
        ("Port Authority", "station", 40.7569, -73.9903, 0.9),
        ("Midtown", "office", 40.7587, -73.9787, 1.4),
        ("Financial District", "office", 40.7069, -74.0113, 1.2),
        ("Hudson Yards", "office", 40.7536, -74.0019, 0.7),
        ("Times Square", "nightlife", 40.7580, -73.9855, 1.1),
        ("East Village", "nightlife", 40.7265, -73.9815, 1.1),
        ("Lower East Side", "nightlife", 40.7150, -73.9843, 0.9),
        ("Meatpacking District", "nightlife", 40.7409, -74.0080, 0.8),
    ),
    drive_on_left=False,
    eta_trips=8_000,
    dispatch_days=3,
    notes=(
        "Traffic follows London's weekday pattern; it has not been checked for New York.",
        "Turn costs are mirrored for driving on the right.",
    ),
)

CITIES: dict[str, City] = {c.key: c for c in (LONDON, NICOSIA, NEW_YORK)}
DEFAULT_CITY = LONDON.key


def get(key: str) -> City:
    try:
        return CITIES[key]
    except KeyError:
        raise ValueError(f"unknown city {key!r}; choose from {', '.join(CITIES)}") from None
