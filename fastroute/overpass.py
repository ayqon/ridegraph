"""Overpass API query for drivable roads.

Kept free of third-party imports so ``python -m fastroute fetch`` works on a
bare Python install.
"""

from __future__ import annotations

from .speeds import ROAD_CLASSES


def overpass_query(south: float, west: float, north: float, east: float) -> str:
    """Overpass QL fetching every drivable way in a bounding box plus its nodes."""
    classes = "|".join(ROAD_CLASSES)
    bbox = f"{south},{west},{north},{east}"
    return (
        "[out:json][timeout:300];"
        f'way["highway"~"^({classes})$"]["area"!="yes"]({bbox});'
        "(._;>;);"
        "out body;"
    )
