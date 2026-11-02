# fastroute

Quickest-route search and learned arrival times on a real road network,
built from first principles.

**[Try the live map →](https://ayqon.github.io/fastroute/)** Click two points in
central London, pick a time of day, and watch how much of the city each
algorithm searches before it finds the route, then compare the router's ETA
with a machine-learned one.

Ride-hailing and maps apps answer "what is the fastest way from A to B, and
when will I get there?" millions of times a second. This project rebuilds both
halves on OpenStreetMap data for central London:

* **Routing.** Dijkstra, bidirectional search, A\* and Contraction
  Hierarchies, implemented by hand rather than called from a library, with
  rush-hour traffic and turn costs, every one tested against a reference.
* **Learned ETA.** The approach behind Uber's DeepETA: keep the routing
  engine, and train a model on observed trips to correct its estimate.
  Gradient-boosted trees cut the error on held-out London trips from 64% to
  8%, and run live in the browser.

| Algorithm | Nodes settled (mean) | vs Dijkstra | Query time (median) | vs Dijkstra |
|---|---:|---:|---:|---:|
| Dijkstra | 37,822 | 1.0x | 44.94 ms | 1.0x |
| Bidirectional Dijkstra | 24,840 | 1.5x | 39.44 ms | 1.0x |
| A\* | 23,705 | 1.6x | 40.62 ms | 0.9x |
| **Contraction Hierarchies** | **97** | **389.5x** | **1.21 ms** | **38.0x** |

<sub>200 random queries on central London (72,636 intersections, 127,286
directed road segments, Paddington to Whitechapel, King's Cross to Vauxhall),
every answer verified against Dijkstra. Reproduce with
`python -m fastroute bench`.</sub>

Oxford Circus to Waterloo, for example, comes out at 8 min 04 s over 3.45 km
with every algorithm. Contraction Hierarchies finds it after looking at 117
intersections; Dijkstra looks at 27,278.

| ETA model, on 4,000 held-out London trips | Mean abs. error | 90th pct. error | Within 10% |
|---|---:|---:|---:|
| Router alone | 64.3% | 74.0% | 0% |
| Router × one global factor | 18.8% | 39.5% | 34% |
| Ridge regression | 10.8% | 23.0% | 58% |
| Neural network (MLP, 64×64) | 8.8% | 18.5% | 68% |
| **Gradient-boosted trees** | **8.1%** | **16.9%** | **72%** |

<sub>Trained on 16,000 of 20,000 simulated trips; see
[the learned ETA](#5-the-learned-eta) for how the trips are generated and
why. Reproduce with `python -m fastroute eta-train`.</sub>

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

### 4. Traffic and turns

**Rush hour** (`traffic.py`). The same road has a different speed at 3am and
8am. Each edge's speed is scaled by a city-wide congestion curve for the hour
(separate weekday and weekend curves) and by how sensitive that class of road
is to congestion: arterials lose up to half their speed at the peak,
residential streets far less, because through-traffic avoids them. So at
8am the quickest route can leave the main road for back streets, and the
tests check that it does. The curves are illustrative, shaped after London's
familiar pattern rather than fitted to measured data; production systems use
historical and live per-segment speeds from vehicle GPS traces, which would
slot in here without touching the routing code.

**Turns** (`turns.py`). In a plain road graph, a node is an intersection, so
the search cannot tell turning right across oncoming traffic from carrying
straight on. The fix is an **edge-expanded graph**: each node is a directed
road segment ("I just drove u to v") and each edge is a manoeuvre ("then
take v to w"), costing the next segment's travel time plus a turn penalty.
UK defaults make right turns (crossing traffic) dearer than left turns, with
extra cost for sharp turns, U-turns and traffic lights parsed from OSM. A
bend in a single road is not a junction, so it costs nothing.

Every algorithm runs on the expanded graph unchanged, Contraction
Hierarchies included, since it is just another weighted graph. A `Router`
(`router.py`) builds the right graph for a scenario and maps routes back to
intersections.

```
$ python -m fastroute eta <from> <to> --turns
03:00 weekday, with turn costs    2 min 20 s   1.01 km  reference route
08:00 weekday, with turn costs    3 min 59 s   0.99 km  different route (30% shared)
13:00 weekday, with turn costs    3 min 06 s   1.01 km  same route
17:30 weekday, with turn costs    3 min 50 s   0.99 km  different route (30% shared)
```

<sub>Synthetic street grid; London output lands with the next data refresh.</sub>

### 5. The learned ETA

The router's estimate is only as good as its model of road speeds. It cannot
know that lights on a particular corridor hold you for half a minute at
8:30, that one junction always queues, or that the evening peak arrives later
than the timetable. Production ride-hailing systems therefore keep the
routing engine and learn the **residual**: given a route and a moment, how
far off will the engine be? `fastroute/eta.py` follows that design.

**Features** are things a production system knows when the trip is
requested: the router's ETA, route length, straight-line distance and
detour, the share of the route on main roads, B roads and side streets, the
number of left, right, sharp and U-turns and of traffic lights on the route,
time of day (as a point on a circle, so 23:59 sits next to 00:00), weekday or
weekend, and where the route starts, ends and passes through.

**The target** is `log(observed time / router ETA)`, so the model learns a
multiplicative correction and large and small trips weigh evenly.

**Observed trips.** Real trip logs for London are not public, so the
pipeline generates them, and it is open about it. A `World` is a hidden
ground truth that differs from the router the way a real city would:
congestion hotspots around a few busy junctions, a rush hour that peaks
later and harder than the router assumes, main roads that clog more than
modelled, waits at right turns and traffic lights that grow with traffic,
occasional incidents and day-to-day noise. 20,000 trips between random
points at random times are routed with Contraction Hierarchies and timed in
that world. Nothing in the features reads the world directly; the models
have to infer its effects from routes and outcomes.

**Results** are measured on trips the models never saw (table above). The
router alone is badly optimistic in central London, mostly because it knows
nothing about traffic lights, which the gradient-boosted model ranks as its
most useful feature, followed by route length and time of day. One global
correction factor helps, a linear model helps more, and both non-linear
models capture the interactions (lights cost more at peak; hotspots matter
only when the route passes through them). Gradient-boosted trees win and,
being a few hundred small trees, are also the easiest to ship.

**In the browser.** The trees export to JSON, and `web/js/eta.js` computes the
same features and walks the same trees. A test runs it under Node and
requires its features and predictions to match Python's to nine decimal
places.

### 6. The live map

The demo in `web/` runs entirely in the browser, with no server:

* **Same data, same answers.** `python -m fastroute export-web` writes the
  graph and one Contraction Hierarchy per time-of-day preset as compact,
  gzipped binary arrays. `web/js/engine.js` is a direct port of the four
  algorithms, and a test runs it under Node and requires exact agreement
  with the Python router on every preset.
* **The map is the graph.** Rather than a tile server's picture of London,
  the page draws the very network being searched, styled like a street
  atlas: roads coloured by class, the Thames, and a lettered grid so routes
  read as "5C to 9I".
* **Search you can see.** Every settled junction lights up its streets in
  the colour of the side that reached it, so Dijkstra visibly floods the
  city while Contraction Hierarchies touches a few hundred junctions
  scattered up the road hierarchy. All four methods run on every trip, with
  junctions examined and query time side by side.
* **Turn costs on demand.** A switch charges for turns and traffic lights.
  The browser rebuilds the edge-expanded turn graph from per-manoeuvre turn
  kinds in the export, numbered exactly as the Python one, and the trip
  summary lists the right and left turns, U-turns and lights on the route.
  Contraction Hierarchies are switched off in this mode: an edge-based
  hierarchy for London would add around 20 MB per time of day, so turn-aware
  hierarchies live in the Python tool instead.
* **Learned ETA beside the router's.** Every trip shows both estimates, and
  a model card shows each model's error on held-out trips and the features
  the model leans on.
* **Built and deployed by CI.** On each push, a workflow downloads central
  London from OpenStreetMap (cached between runs), builds the graph,
  contracts it for each preset, trains the ETA model and publishes `web/` to
  GitHub Pages.

### 7. What the London numbers show

* **Contraction Hierarchies work because real roads are hierarchical.** On a
  uniform synthetic grid the reduction in search work is 24x; on London it
  is 389x, with preprocessing adding fewer shortcuts (121k) than the map has
  road segments. Main roads naturally end up at the top of the ranking.
* **A\* barely helps in a city.** Its estimate assumes the fastest road
  speed in a straight line, but London's average route speed is about half
  of that, so the search still covers a wide ellipse. This is why production
  engines rely on preprocessing rather than on geometric heuristics.
* **Bidirectional search halves the work in theory but not the time in
  Python**, because juggling two frontiers costs interpreter overhead that
  eats most of the saving.

### 8. Measuring fairly

Benchmarks report **nodes settled** alongside wall-clock time. Settled nodes
measure the work each algorithm does, independent of language; Python
timings mostly measure interpreter overhead per node, which is why
Contraction Hierarchies' 24x reduction in work shows up as a smaller time
gain here. Every benchmark query is also cross-checked against Dijkstra, so a
benchmark run doubles as a correctness check.

## Quick start

```bash
pip install -e ".[dev]"     # or ".[ml]" for just the ETA model extras

python -m fastroute fetch      # download central London from OpenStreetMap
python -m fastroute build      # turn it into a routing graph
python -m fastroute contract   # one-off Contraction Hierarchies preprocessing
python -m fastroute info

# Oxford Circus to Waterloo (default algorithm: ch)
python -m fastroute route 51.5154,-0.1410 51.5033,-0.1196
python -m fastroute route 51.5154,-0.1410 51.5033,-0.1196 -a astar

python -m fastroute bench      # compare all four on random queries

# Rush hour, with turn costs and traffic lights
python -m fastroute route 51.5154,-0.1410 51.5033,-0.1196 --at 08:30 --turns -a astar
python -m fastroute eta 51.5154,-0.1410 51.5033,-0.1196 --turns

# Contraction Hierarchies need preprocessing per scenario
python -m fastroute contract --at 08:30 --turns
python -m fastroute bench --at 08:30 --turns

# Simulate trips, train and compare ETA models
python -m fastroute eta-train --trips 20000 --out data/eta-model.json
```

On Windows PowerShell 5, run the commands one per line (it does not support
`&&`). `fetch` uses shared public OpenStreetMap servers that are sometimes
busy; it retries and falls back across mirrors automatically.

To run the map locally after `fetch` and `build`:

```bash
python -m fastroute export-web
python -m http.server -d web 8000   # then open http://localhost:8000
```

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
pruning, hierarchy rank invariants, witness behaviour, save/load round
trips, rush-hour route changes, left versus right turn costs (and their swap
for right-hand traffic), traffic lights, free bends, U-turns, and the
turn-expanded graph matching the plain one when penalties are zero.

The ETA model has its own checks: features agree with the router's own trip
times and lengths, the simulated world is reproducible and slower than the
router at peak, the exported trees reproduce scikit-learn's predictions, and
the learned model beats both the router and a global factor. Two Node tests
hold the browser to the Python implementation: all routing algorithms, with
and without turn costs, and the ETA features and predictions.

CI runs lint, formatting and the full suite on Python 3.10, 3.12 and 3.13.

## Roadmap

- [x] **Road graph and Dijkstra:** OSM pipeline, travel-time model, CSR graph, tested baseline
- [x] **A\* and bidirectional search:** admissible straight-line heuristic, meet-in-the-middle stopping rule
- [x] **Contraction Hierarchies:** node ordering, witness search, shortcuts, upward query, path unpacking
- [x] **Traffic and turns:** time-of-day speed profiles, edge-expanded graph with turn and traffic-light costs
- [x] **Live map:** in-browser routing on a street-atlas render of the graph, animating each algorithm's search
- [x] **Learned ETA:** simulated observed trips, route features, ridge, MLP and gradient-boosted trees, shipped to the browser

## Data

Road data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
available under the Open Database License.

## License

MIT
