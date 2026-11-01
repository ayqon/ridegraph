from __future__ import annotations

import io
import json
import urllib.error

import pytest

from fastroute.__main__ import main
from fastroute.overpass import OverpassError, download

GOOD = json.dumps({"elements": [{"type": "node", "id": 1, "lat": 51.5, "lon": -0.1}]}).encode()


class FakeServers:
    """Stands in for urlopen: replays a scripted outcome per call."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.hosts: list[str] = []

    def __call__(self, request, timeout):
        self.hosts.append(request.full_url.split("/")[2])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return io.BytesIO(outcome)


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {}, None)


def fetch(servers: FakeServers, urls=("https://a/api", "https://b/api")) -> bytes:
    return download("q", urls, opener=servers, sleep=lambda s: None, log=lambda m: None)


def test_returns_first_success() -> None:
    servers = FakeServers(GOOD)
    assert fetch(servers) == GOOD
    assert servers.hosts == ["a"]


def test_retries_busy_server_then_succeeds() -> None:
    servers = FakeServers(http_error(502), GOOD)
    assert fetch(servers) == GOOD
    assert servers.hosts == ["a", "a"]


def test_falls_back_to_mirror() -> None:
    servers = FakeServers(http_error(502), http_error(504), GOOD)
    assert fetch(servers) == GOOD
    assert servers.hosts == ["a", "a", "b"]


def test_does_not_retry_bad_request_on_same_server() -> None:
    servers = FakeServers(http_error(400), GOOD)
    assert fetch(servers) == GOOD
    assert servers.hosts == ["a", "b"]


def test_rejects_query_aborted_inside_200_response() -> None:
    aborted = json.dumps({"elements": [], "remark": "runtime error: Query timed out"}).encode()
    servers = FakeServers(aborted, GOOD)
    assert fetch(servers) == GOOD


def test_network_errors_are_retried() -> None:
    servers = FakeServers(urllib.error.URLError("connection reset"), GOOD)
    assert fetch(servers) == GOOD


def test_gives_up_with_summary() -> None:
    servers = FakeServers(*[http_error(503)] * 4)
    with pytest.raises(OverpassError, match="a: HTTP 503"):
        fetch(servers)


def test_missing_inputs_give_clear_messages(tmp_path) -> None:
    with pytest.raises(SystemExit, match="fastroute fetch"):
        main(["build", "--raw", str(tmp_path / "missing.json.gz")])
    with pytest.raises(SystemExit, match="fastroute build"):
        main(["contract", "--network", str(tmp_path / "missing.npz")])
