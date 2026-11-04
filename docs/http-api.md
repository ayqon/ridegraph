# HTTP API

`python -m ridegraph serve` (or the Docker image) runs the engine as a JSON
API. Interactive OpenAPI documentation is served at `/docs` by the running
service.

## Endpoints

### `GET /route`

| Parameter | Meaning |
|---|---|
| `from`, `to` | `lat,lon` of each end; points more than 500 m from any road are rejected |
| `at` | Time of day, `HH:MM` (default `08:00`); served at the nearest quarter-hour |
| `weekend` | Use the weekend traffic curve |
| `alternatives` | Up to two extra routes, by the penalty method |
| `avoid_charge` | Add the quickest route that stays out of the Congestion Charge zone |

The response is a GeoJSON `FeatureCollection`, quickest route first. Each
feature carries `seconds`, `metres`, the `charge` it incurs at that time, and
`learned_eta` (with `window_seconds`, the calibrated 10th to 90th percentile
range) when the service was started with a model.

```json
{
  "type": "FeatureCollection",
  "time": "08:15",
  "query_ms": 5.6,
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "LineString", "coordinates": [[-0.141, 51.5154], "..."]},
      "properties": {
        "name": "quickest",
        "seconds": 876.9,
        "metres": 3590.1,
        "charge": {"enters_zone": true, "charged": true, "amount_gbp": 18.0},
        "learned_eta": {"seconds": 1498.6, "window_seconds": [1206.8, 1568.5]}
      }
    }
  ]
}
```

### `GET /reach`

`from`, `at`, `weekend`, `minutes` (comma-separated, 1 to 60) and `arriving`.
Returns the area reachable within each band in km² and the junctions inside.

### `POST /tour`

```json
{"stops": [[51.5174, -0.12], [51.5308, -0.1238], [51.5033, -0.1145]], "at": "13:00", "round_trip": true}
```

Returns the best order (indices into `stops`, depot first), which method
found it, and the total time under every method.

### `GET /health`

Network size, hierarchy size, whether a learned model is loaded, and how long
start-up took.

## Errors

Malformed coordinates, points off the map, impossible times and trips that
cannot avoid the zone return `422` with a sentence explaining why;
disconnected points return `404`.

## Performance

On central London with one worker on a 2-vCPU machine (the load generator on
the same machine): 14 ms median and 21 ms 95th-percentile latency per route
including the learned ETA, about 70 requests per second per worker, over
2,000 random requests across six quarter-hours. A Customizable CH query
alone takes about 5.6 ms in Python.
