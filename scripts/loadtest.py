"""Load-test a running fastroute API with random London routes.

    python -m fastroute serve --network data/central-london.npz &
    python scripts/loadtest.py --requests 2000 --concurrency 16

Origins and destinations are random junctions of the network, so every
request is a fresh query; times of day are spread over the morning so the
quarter-hour cache does real work.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time

import httpx
import numpy as np


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--network", default="data/central-london.npz")
    p.add_argument("--requests", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--learned", action="store_true", help="request routes only (ETA if loaded)")
    args = p.parse_args()

    data = np.load(args.network)
    lat, lon = data["lat"], data["lon"]
    rng = random.Random(1)
    times = ["07:30", "07:45", "08:00", "08:15", "08:30", "08:45"]
    queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
    for _ in range(args.requests):
        a, b = rng.randrange(len(lat)), rng.randrange(len(lat))
        queue.put_nowait(
            {
                "from": f"{lat[a]:.6f},{lon[a]:.6f}",
                "to": f"{lat[b]:.6f},{lon[b]:.6f}",
                "at": rng.choice(times),
            }
        )
    latencies: list[float] = []
    errors = 0

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal errors
        while not queue.empty():
            params = queue.get_nowait()
            started = time.perf_counter()
            r = await client.get("/route", params=params)
            latencies.append((time.perf_counter() - started) * 1000)
            if r.status_code != 200:
                errors += 1

    async with httpx.AsyncClient(base_url=args.url, timeout=30) as client:
        await client.get("/route", params={"from": "51.5154,-0.1410", "to": "51.5033,-0.1145"})
        started = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(args.concurrency)))
        elapsed = time.perf_counter() - started

    q = statistics.quantiles(latencies, n=100)
    print(f"{len(latencies):,} requests, concurrency {args.concurrency}, {errors} errors")
    print(f"throughput {len(latencies) / elapsed:,.0f} requests/s")
    print(f"latency p50 {q[49]:.1f} ms, p95 {q[94]:.1f} ms, p99 {q[98]:.1f} ms")


if __name__ == "__main__":
    asyncio.run(main())
