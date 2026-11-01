# fastroute

Quickest-route search on a real road network, built from first principles.

Ride-hailing and maps apps answer "what is the fastest way from A to B?"
millions of times a second. This project rebuilds that core on OpenStreetMap
data for central London: the routing algorithms are implemented by hand, not
called from a library, and every one is tested against a reference
implementation.

| Algorithm | Nodes settled (mean) | vs Dijkstra | Query time (median) | vs Dijkstra |
|---|---:|---:|---:|---:|
| Dijkstra | 7,752 | 1.0x | 9.67 ms | 1.0x |
| Bidirectional Dijkstra | 5,222 | 1.5x | 9.13 ms | 0.9x |
| A\* | 3,476 | 2.2x | 6.09 ms | 1.2x |
| **Contraction Hierarchies** | **318** | **24.3x** | **2.42 ms** | **4.1x** |

<sub>100 random queries on a 14,400-node synthetic street grid
(`python -m fastroute bench --grid 120 --queries 100`). London results land
with the next milestone.</sub>

> **Status:** milestones 1 to 3 of 6 are done: road graph, travel-time model,
> Dijkstra, bidirectional search, A\* and Contraction Hierarchies. Traffic and
> the live map demo are next ([roadmap](#roadmap)).

## How it works

### 1. The map is a graph

Intersections become **nodes** and road segments become directed **edges**.
A two-way street is two edges; a one-way street or roundabout is one.

The pipeline (`fastroute/osm.py`) pulls drivable ways from OpenStreetMap,
drops what a car cannot use (footpaths, private roads, car park aisles),
applies `oneway`, roundabout and motorway direction rules, then keeps only the
largest strongly connected component. That last step matters: cutting a city
out of a bounding box leaves stubs you can drive into but never leave, and a
router that starts there would fail.

### 2. Edges cost seconds, not metres

The quickest route is not the shortest one. Each edge's weight is its travel
time (`fastroute/speeds.py`):

```
seconds = length / free-flow speed
```

Free-flow speed comes from the posted `maxspeed` when OSM has it, discounted to
what drivers actually sustain between junctions, or from a typical speed for
the road class otherwise. So 900 m of main road can beat 600 m of back street,
and the tests check that it does.

The raw map (`RoadNetwork`) and the weighted routing graph (`Graph`) are kept
separate, so swapping in a traffic model means rebuilding weights, not
re-downloading the map. Edges are stored in CSR (compressed sparse row) form,
the layout production engines use: all edges leaving node `u` sit
contiguously in `head[first_out[u]:first_out[u + 1]]`, with a mirrored
incoming index for searches that run backwards.

### 3. Four ways to search

**Dijkstra** (`dijkstra.py`) grows a ball of settled nodes outward from the
start in order of travel time; when the destination is settled, its time is
provably optimal. It is the reference every other algorithm must agree with,
and its weakness motivates the rest: the ball expands in every direction,
including away from the destination.

**Bidirectional Dijkstra** (`bidirectional.py`) searches from both ends at
once, so two half-size balls replace one big one. The subtle part is the
stopping rule: the first node both searches reach is not necessarily on the
best route, so the search tracks the best joined route seen and stops only
once the two frontiers together cannot beat it.

**A\*** (`astar.py`) adds a sense of direction by ranking nodes on time so far
plus an estimate of time remaining: straight-line distance at the network's
top speed. That estimate can never overshoot, so results stay optimal while
the search stretches towards the target. In a city the gain is bounded
because "top speed in a straight line" is very optimistic.

**Contraction Hierarchies** (`ch.py`) is what production routing engines rely
on. A one-off preprocessing step ranks nodes by importance and removes them
from least to most important. Whenever removing node `v` would break a
quickest route `u -> v -> w`, a **shortcut** edge `u -> w` is added, unless a
local *witness search* finds another route that is just as quick.

Afterwards, every quickest route can be found by searching only *upwards* in
the ranking from both ends, so each query explores a small cone instead of a
ball of the city. The implementation includes lazy priority updates during
ordering, bounded witness searches, stall-on-demand at query time, and
recursive unpacking of shortcuts back into real road segments.

### 4. Measuring fairly

Benchmarks report **nodes settled** alongside wall-clock time. Settled nodes
measure the work each algorithm does, independent of language; Python
timings mostly measure interpreter overhead per node, which is why
Contraction Hierarchies' 24x reduction in work shows up as a smaller time
gain here. Every benchmark query is also cross-checked against Dijkstra, so a
benchmark run doubles as a correctness check.

## Quick start

```bash
pip install -e ".[dev]"

python -m fastroute fetch      # download central London from OpenStreetMap
python -m fastroute build      # turn it into a routing graph
python -m fastroute contract   # one-off Contraction Hierarchies preprocessing
python -m fastroute info

# Oxford Circus to Waterloo (default algorithm: ch)
python -m fastroute route 51.5154,-0.1410 51.5033,-0.1196
python -m fastroute route 51.5154,-0.1410 51.5033,-0.1196 -a astar

python -m fastroute bench      # compare all four on random queries
```

On Windows PowerShell 5, run the commands one per line (it does not support
`&&`). `fetch` uses shared public OpenStreetMap servers that are sometimes
busy; it retries and falls back across mirrors automatically.

No map data handy? `python -m fastroute bench --grid 120` runs the comparison
on a synthetic street grid.

## Testing

```bash
pytest
```

Every algorithm is checked for exact agreement with Dijkstra on randomised
street grids with one-way roads, and Dijkstra itself is checked against
NetworkX. Targeted cases cover a faster road beating a shorter one, one-way
loops, parallel edges, unreachable targets, OSM tag parsing, component
pruning, hierarchy rank invariants, witness behaviour and save/load round
trips. CI runs lint, formatting and tests on Python 3.10, 3.12 and 3.13.

## Roadmap

- [x] **Road graph and Dijkstra:** OSM pipeline, travel-time model, CSR graph, tested baseline
- [x] **A\* and bidirectional search:** admissible straight-line heuristic, meet-in-the-middle stopping rule
- [x] **Contraction Hierarchies:** node ordering, witness search, shortcuts, upward query, path unpacking
- [ ] **Traffic:** time-of-day speed profiles and turn penalties
- [ ] **Live demo:** in-browser routing on a map, animating the nodes each algorithm explores
- [ ] **ETA model:** learning a correction to graph ETAs, the approach used in production ride-hailing

## Data

Road data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
available under the Open Database License.

## License

MIT
