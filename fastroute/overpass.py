"""Download drivable roads from the Overpass API.

Kept free of third-party imports so ``python -m fastroute fetch`` works on a
bare Python install.

The public Overpass servers are shared and frequently overloaded, answering
502/503/504 or 429 (rate limited) under load. Downloads therefore retry with
backoff and fall back across independent mirrors before giving up.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence

from .speeds import ROAD_CLASSES

OVERPASS_URLS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
USER_AGENT = "fastroute (github.com/ayqon/fastroute)"
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}


class OverpassError(RuntimeError):
    """Every server and retry failed."""


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


def download(
    query: str,
    urls: Sequence[str] = OVERPASS_URLS,
    *,
    attempts_per_server: int = 2,
    backoff_s: float = 15.0,
    timeout_s: float = 600.0,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
    opener: Callable[..., object] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Run ``query`` against each server in turn until one returns valid data."""
    body = urllib.parse.urlencode({"data": query}).encode()
    failures: list[str] = []
    for url in urls:
        host = urllib.parse.urlparse(url).netloc
        for attempt in range(1, attempts_per_server + 1):
            request = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT})
            log(f"  {host}: attempt {attempt}/{attempts_per_server} ...")
            try:
                with opener(request, timeout=timeout_s) as response:  # type: ignore[attr-defined]
                    raw = response.read()
                _check_payload(raw)
                return raw
            except urllib.error.HTTPError as exc:
                reason = f"HTTP {exc.code}"
                retryable = exc.code in _RETRYABLE_HTTP
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = f"network error ({getattr(exc, 'reason', exc)})"
                retryable = True
            except ValueError as exc:  # malformed or partial response
                reason = str(exc)
                retryable = True
            failures.append(f"{host}: {reason}")
            log(f"  {host}: {reason}")
            if not retryable:
                break
            if attempt < attempts_per_server:
                wait = backoff_s * attempt
                log(f"  waiting {wait:.0f}s before retrying")
                sleep(wait)
    raise OverpassError(
        "All Overpass servers failed. They are shared public servers and are often "
        "busy; try again in a few minutes.\n  " + "\n  ".join(failures)
    )


def _check_payload(raw: bytes) -> None:
    """Reject responses that are not a complete Overpass result."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("response was not valid JSON") from exc
    remark = str(payload.get("remark", ""))
    # Overpass reports query timeouts and memory exhaustion inside a 200 response.
    if "runtime error" in remark.lower():
        raise ValueError(f"server aborted the query: {remark.strip()}")
    if not payload.get("elements"):
        raise ValueError("response contained no map data")
