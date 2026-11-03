# fastroute

[![CI](https://github.com/ayqon/fastroute/actions/workflows/ci.yml/badge.svg)](https://github.com/ayqon/fastroute/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fayqon%2Ffastroute%2Fbadges%2Fcoverage.json)](https://github.com/ayqon/fastroute/actions/workflows/ci.yml)
[![Compared with OSRM](https://github.com/ayqon/fastroute/actions/workflows/osrm.yml/badge.svg)](https://github.com/ayqon/fastroute/actions/workflows/osrm.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![Typed with mypy](https://img.shields.io/badge/types-mypy-2a6db2)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-d7ff64)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Quickest routes, live closures, travel-time maps, delivery rounds, learned
arrival times and ride dispatch on London's real road network, built from
first principles.

**[Try the live map →](https://ayqon.github.io/fastroute/)** Click two points in
central London, pick a time of day, and watch how much of the city each
algorithm searches before it finds the route. Compare alternative routes and
the Congestion Charge, switch on live TfL closures, and set the router's ETA
beside a machine-learned one.

![The live map: Dijkstra floods the city while Contraction Hierarchies touches a few hundred junctions; alternative routes and the Congestion Charge; a travel-time map; a delivery round; ride dispatch with shared rides](docs/demo.gif)

The short version of what the numbers show is in [the write-up](docs/writeup.md).

Ride-hailing and maps apps answer "what is the fastest way from A to B, and
when will I get there?" millions of times a second. This project rebuilds both
halves on OpenStreetMap data for central London:

* **Routing.** Dijkstra, bidirectional search, A\* and Contraction
  Hierarchies, implemented by hand rather than called from a library, with
  rush-hour traffic and turn costs, every one tested against a reference.
* **Live closures in milliseconds.** Customizable Contraction Hierarchies
  separate the road layout from travel times, so TfL's live closures are
  folded in by re-customising in 20 ms in the browser instead of
  rebuilding. Alternative routes come from the penalty method, and any
  route can avoid the Congestion Charge zone (from Paddington to Aldgate
  at 8am that costs 14 minutes and saves £18).
* **Everyday tools on the same engine.** A travel-time map shades
  everywhere reachable in 10, 20 and 30 minutes from (or to) any point,
  at rush hour and at night: from Oxford Circus, 20 minutes at 8am covers
  46.6 km², 74% of what it covers at 3am. A delivery planner finds the best
  order for up to a dozen stops: eight drops around central London take
  1 h 07 min in the optimal order, 49% less than in the order entered.
* **Ride dispatch.** Every 30 seconds, price every free driver against every
  waiting rider with one many-to-many query and assign them optimally with
  the Hungarian algorithm; move idle drivers ahead of demand predicted by a
  forecasting model. Shared rides and surge pricing are simulated on the
  same requests, so their trade-offs can be measured rather than argued.
* **Learned ETA.** The approach behind Uber's DeepETA: keep the routing
  engine, and train a model on observed trips to correct its estimate.
  Gradient-boosted trees cut the error on held-out London trips from 28% to
  7%, give each trip a calibrated arrival window, and run live in the
  browser, where a map layer shows the congestion the model discovered. A
  graph neural network over every road, written in numpy, does slightly
  better again.

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
| Router alone | 27.5% | 38.5% | 2% |
| Router × one global factor | 10.1% | 20.3% | 58% |
| Ridge regression | 7.6% | 15.5% | 74% |
| Neural network (MLP, 64×64) | 7.3% | 14.4% | 75% |
| **Gradient-boosted trees** (shipped to the map) | **6.8%** | **13.8%** | **78%** |
| Graph neural network over every road | 6.6% | 13.4% | 80% |

* **Arrival window:** the 10th to 90th percentile range contains **81.0%** of
  held-out trips against an 80% target (77.4% before conformal calibration).
* **What it discovered:** its map of slow areas correlates at **r = 0.83**
  with congestion hotspots it was never shown.

<sub>Trained on 16,000 of 20,000 simulated trips; see
[the learned ETA](#7-the-learned-eta) for how the trips are generated and
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

### 5. Live closures and Customizable Contraction Hierarchies

Contraction Hierarchies are fast because they are built for one set of
travel times. A closure on Oxford Street makes them stale, and rebuilding
takes far longer than a user will wait. Production engines (OSRM's MLD, Bing
Maps) separate the road layout from the times. `fastroute/cch.py` implements
**Customizable CH** (Dibbelt, Strasser and Wagner, 2016):

1. **Once per map:** order junctions by *nested dissection* (cut the city
   along the direction that crosses the fewest roads, recurse on each half,
   rank the separator last), then contract in that order adding every
   shortcut, with no witness searches. On London: 77,124 roads become
   301,179 edges and 3.19 million *lower triangles*, in 8 s.
2. **Whenever times change:** load the current time of every road and sweep
   the triangles bottom-up, `u->w = min(u->w, u->v + v->w)`. That takes
   **50 ms** in numpy and **about 20 ms** in the browser.
3. **Queries:** the usual two upward searches, with shortcuts unpacked
   through the triangles.

The times come from **Transport for London's open disruption feed**
(`fastroute/live.py`). Each disrupted stretch is map-matched onto the graph
(both ends and the middle of an edge within 18 m of the line). Closed roads
get an infinite time, and open ones are slowed by severity. The map asks
TfL directly when you switch on *Live roadworks and closures*. If the feed
can't be reached from the browser, it falls back to a snapshot taken when CI
built the page.

```bash
python -m fastroute live                      # what is disrupted right now
python -m fastroute route 51.5130,-0.1590 51.5160,-0.1300 --at 13:00 --live
```

### 6. Alternative routes and the Congestion Charge

Map apps offer two or three routes because the quickest is not always the one
you want. `fastroute/alternatives.py` finds them with the **penalty method**:
take the quickest route, make every road on it 40% slower, search again, and
repeat. Each search is pushed off the corridor already used. A candidate is
kept only if it is

* at most 30% slower than the quickest (on real times, not penalised ones),
* at most 65% shared with any route already offered, and
* free of pointless detours: sampled stretches of a quarter of the trip must
  each be within 15% of the quickest way between their ends.

Closing roads is the same machinery with infinite penalties, which is how a
route **avoids the Congestion Charge zone** (`fastroute/charges.py`: the zone
polygon, £18 a day, 07:00 to 18:00 on weekdays and 12:00 to 18:00 at
weekends; the boundary roads are free). From Paddington to Aldgate at 8am:

| Route | Time | Charge |
|---|---:|---|
| Quickest | 36 min 32 s | £18 |
| Alternative 1 | 40 min 11 s | £18 |
| Alternative 2 | 45 min 27 s | £18 |
| Avoiding the zone, along Euston Road and City Road | 50 min 08 s | none |

So staying out of the zone costs 14 minutes on this trip, about £1.30 a minute
saved. The browser runs the same method (A\* on the penalised weights, under
100 ms on London) and a Node test holds it to Python.

```bash
python -m fastroute route 51.5265,-0.1820 51.5100,-0.0700 --at 08:00 --alternatives --avoid-charge
```

### 7. The learned ETA

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
simulated city is calibrated to central London: trips average about 21 km/h
at night, 14 at midday and 11 at the peaks. The router alone is optimistic
by about 28%, mostly because it knows nothing about traffic lights or about
when the peaks really arrive. Time of day is the model's most useful
signal, followed by where the route runs, weekday or weekend, length and
the number of sets of lights. One global factor helps, a linear model helps
more, and both non-linear models capture the interactions (lights cost more
at peak; hotspots matter only when the route passes through them).
Gradient-boosted trees win among the models that read one row per trip and,
being a few hundred small trees, are also the easiest to ship.

**A graph neural network** (`fastroute/gnn.py`) works on the roads instead
of on trip summaries, the approach DeepMind took for Google Maps. Every road
segment starts with its own features (class, length, lights, junction size,
position as sines and cosines), which are then averaged over the *line
graph* (segments that lead into one another) for 1, 2, 4 and 8 hops. That
is SIGN-style message passing, precomputed so it scales to all 127,000
segments. A two-layer network reads each segment's representation and the
time of day and predicts its slowdown; a trip's time is the sum of
`router seconds x slowdown` over its segments plus learned waits for each
turn and set of lights. It is written in numpy with hand-derived gradients
(checked against finite differences in the tests) and trained with Adam.

On held-out London trips it edges out the trees: 6.6% error against 6.8%,
80% of trips within 10% against 78%. The trees still run in the browser,
since the network needs the representation of every road (about 60 MB).
The network's per-road slowdowns at 08:30 correlate at r = 0.60 with the
hidden truth road by road, a harder test than the trees' area map below.

**Traffic lights in OpenStreetMap** are often tagged on each approach to a
junction and on nearby crossings, about 10 points per km of central London
route. Signal nodes within 60 m along a route therefore count as one set
(about 4 per km), both in the simulation and in the model's features.

**Arrival windows.** A single number hides how much trips vary, so two more
gradient-boosted models predict the 10th and 90th percentiles, and each trip
gets a range such as "usually 25 to 36 min". Quantile models alone tend to be
overconfident (77% coverage on London against an 80% target), so the window
is calibrated with **conformalized quantile regression**: part of the
training set is held back, the models' misses on it are measured, and the
window is widened by exactly the margin that gives 80% coverage on new trips.
On held-out London trips it lands at 81.0%, with misses split evenly above
and below.

**What the model learned.** The hidden hotspots are never shown to the model,
yet it should find them, since trips through them run slow. The map's
*Learned slowdown* layer asks the model how far off the router a typical
trip through each grid square would be at the chosen time, and shades the
square accordingly. Against the hidden hotspot field this map correlates at
r = 0.83 on London (0.91 on a synthetic grid): the model has located the
congestion from trip outcomes alone.

**In the browser.** The trees export to JSON, and `web/js/eta.js` computes the
same features and walks the same trees. A test runs it under Node and
requires its features and predictions to match Python's to nine decimal
places.

### 8. Travel-time maps and delivery rounds

**Travel-time maps** (`fastroute/reach.py`) answer "where can I get to in 20
minutes from here?", or, for someone choosing where to live, "from where can
I get *to* my office in 20 minutes?". It is a single-source search that
settles junctions in time order and stops at the budget, run on the reverse
graph for journeys into a point. Reachable area is counted over 250 m
squares. From Holborn, 10 minutes reaches 12.2 km² at 08:00 against
36.6 km² at 03:00: rush hour shrinks a 10-minute radius to a third.

**Delivery rounds** (`fastroute/tour.py`) find the best order to visit a
list of stops, the travelling salesman problem on one-way streets, where
A to B and B to A take different times:

1. One many-to-many Contraction Hierarchies query prices every stop against
   every other at the chosen time of day.
2. **Held-Karp** dynamic programming finds the provably best order, instant
   up to about a dozen stops.
3. Nearest-next-stop construction plus **2-opt and Or-opt** improvement give
   a fast heuristic for comparison and for larger jobs, re-costing whole
   tours on each move because reversing a stretch of an asymmetric tour
   changes its cost.

| Eight drops from Holborn, midday, round trip | Time |
|---|---:|
| In the order entered | 2 h 07 min |
| Nearest next stop | 1 h 20 min |
| Nearest next stop + 2-opt/Or-opt | 1 h 05 min |
| **Optimal (Held-Karp)** | **1 h 05 min** |

```bash
python -m fastroute reach 51.5174,-0.1200 --at 08:00
python -m fastroute tour 51.5174,-0.1200 51.5308,-0.1238 51.5033,-0.1145 ...
```

### 9. Ride dispatch and demand forecasting

Routing exists to answer a business question: which driver should take this
ride? `fastroute/dispatch.py` simulates a fleet over the 07:00 to 10:00
morning peak and compares six ways of answering it, on exactly the same
ride requests and the same starting fleet.

**Demand** (`fastroute/demand.py`) comes from places with their own rhythm:
main-line stations busy at commuting hours, the City and Holborn filling in
the morning and emptying in the evening, Soho, Covent Garden and Shoreditch
at night, plus requests from anywhere. Every day has its own level, its own
surprises at each hotspot, and a couple of unannounced events (a match, a
concert) that make one area surge for an hour or two.

**Pricing every pairing at once.** Every 30 seconds the dispatcher needs the
pickup time from every free driver to every waiting rider. Running one
query per pair would be hundreds of searches; bucket-based **many-to-many
Contraction Hierarchies** do it with one small upward search per driver and
per rider, then join them through buckets at the nodes they share.

**The methods compared:**

* *Nearest driver.* Riders, oldest first, each take the closest free driver.
* *Optimal matching.* The **Hungarian algorithm** (`fastroute/matching.py`)
  assigns the whole batch at once to minimise total pickup time, with each
  rider's cost reduced by how long they have already waited so that
  hard-to-reach riders are not left until they give up.
* *Optimal + repositioning.* Idle drivers also move towards neighbourhoods
  where expected demand over the next 15 minutes most exceeds the drivers
  already there, using either the historical average for that place and
  time, or an **ML demand forecast**.
* *+ shared rides.* The forecast method, and riders going the same way share
  a car. A waiting rider can **join a car already carrying one rider**: the
  car's position is the last junction it has reached on its route (every
  prefix of a quickest route is itself quickest, so a binary search over CH
  queries finds it), and every pickup and drop order is priced. Riders in
  the same batch can also pair up from the start. Nobody is held up by more
  than 40% of their own trip or six minutes, and the pairs that save most
  driving go first.
* *+ surge pricing.* The forecast method, and every two minutes each
  neighbourhood gets a price multiplier from how far waiting and expected
  riders outnumber nearby free drivers. Each rider accepts with probability
  `exp(-(multiplier - 1))`, and idle drivers are drawn towards surging
  areas. Surge here does not bring extra drivers online, so the comparison
  isolates its rationing effect.

**The forecast** is a gradient-boosted Poisson model predicting ride requests
per 500 m square for the next 15 minutes, from the time, the place and what
has happened so far today (the last 15 minutes and the last hour, nearby
and city-wide). On days it never saw:

| Forecast | Poisson deviance (lower is better) |
|---|---:|
| Same as the last 15 minutes | 3.35 |
| Historical average for the time | 0.77 |
| **Gradient-boosted forecast** | **0.39** |

**Results**, averaged over 6 simulated London mornings with 115 drivers:

| Method | Avg. wait | 90th pct. wait | Pickup drive | Gave up | Empty driving per driver |
|---|---:|---:|---:|---:|---:|
| Nearest driver | 7.3 min | 13.3 min | 5.8 min | 5.9% | 0 |
| Optimal matching | 6.6 min | 12.1 min | 5.4 min | 6.2% | 0 |
| Optimal + historical-average repositioning | 6.2 min | 11.6 min | 5.1 min | 4.9% | 16 min |
| **Optimal + ML-forecast repositioning** | **6.1 min** | **11.3 min** | **4.9 min** | 5.0% | **12 min** |
| + shared rides | 5.1 min | 9.6 min | 4.4 min | 1.4% | 14 min |
| + surge pricing | 5.2 min | 9.6 min | 4.7 min | 0.3% | 14 min |

Optimal matching alone cuts waits by about 10%, and repositioning ahead of
demand brings the total to 16%. The ML forecast halves the forecasting error
of the historical average, but in this dispatch loop that shows up mainly as
efficiency: it reaches the best waits while sending drivers on about a
quarter less empty repositioning, because it moves them only where demand is
actually building today. Better forecasts do not automatically mean
proportionally better decisions; the repositioning rule is where the next
gains would come from.

The two product levers pull in opposite directions:

| On top of the ML-forecast method | Riders served | Shared | Priced out | Fleet driving per ride served |
|---|---:|---:|---:|---:|
| Neither | 95.0% | none | none | 25.7 min |
| **Shared rides** | **98.6%** | 46% of rides | none | **22.1 min** |
| Surge pricing | 91.3% | none | 8.5% | 25.8 min |

**Shared rides** serve more people with less driving: 46% of rides are
shared, waits fall by 16%, those who give up drop from 5.0% to 1.4%, and the
fleet drives 14% less per rider served. The price is 2.1 extra minutes in
the car for a shared rider. **Surge pricing** also shortens waits (5.2 min,
almost nobody gives up), but by turning riders away: 8.5% decline the
price, so fewer riders are served than with no surge at all. Without the
extra drivers a higher price draws in, surge only rations a fixed fleet.
That is the honest reading of the simulation, and the reason real
platforms justify surge by its effect on supply.

### 10. The live map

The demo in `web/` runs entirely in the browser, with no server:

* **Same data, same answers.** `python -m fastroute export-web` writes the
  graph and one Contraction Hierarchy per time-of-day preset as compact,
  gzipped binary arrays. `web/js/engine.js` is a direct port of the
  algorithms, and a test runs it under Node and requires exact agreement
  with the Python router on every preset.
* **The map is the graph.** Rather than a tile server's picture of London,
  the page draws the very network being searched, styled like a street
  atlas: roads coloured by class, the Thames, and a lettered grid so routes
  read as "5C to 9I".
* **Search you can see.** Every settled junction lights up its streets in
  the colour of the side that reached it, so Dijkstra visibly floods the
  city while Contraction Hierarchies touches a few hundred junctions
  scattered up the road hierarchy. All five methods (Customizable CH
  included) run on every trip, with junctions examined and query time side
  by side.
* **Routes, charges and closures.** Each trip lists the quickest route and
  up to two alternatives with their times and whether they enter the
  Congestion Charge zone, which is drawn on the map; a switch adds the
  quickest route that stays out. *Live roadworks and closures* asks TfL's
  feed directly, draws the closed streets and re-customises the
  hierarchy, while plain CH steps aside because its shortcuts are stale.
* **Turn costs on demand.** A switch charges for turns and traffic lights.
  The browser rebuilds the edge-expanded turn graph from per-manoeuvre turn
  kinds in the export, numbered exactly as the Python one, and the trip
  summary lists the right and left turns, U-turns and lights on the route.
  Contraction Hierarchies are switched off in this mode: an edge-based
  hierarchy for London would add around 20 MB per time of day, so turn-aware
  hierarchies live in the Python tool instead.
* **Learned ETA beside the router's.** Every trip shows both estimates and a
  calibrated arrival window; a model card gives each model's error on
  held-out trips and the features the model leans on, and a map layer shades
  the city by the slowdown the model learned for the chosen time.
* **Four modes.** *Plan a route*, *Travel-time map* (click anywhere; leaving
  or arriving; each band compared with 3am), *Deliveries* (click to add up
  to 12 stops; every ordering method compared, the best drawn and
  numbered) and *Dispatch rides*. The browser runs the same algorithms as
  Python, held to them by Node tests.
* **Dispatch you can watch.** A *Dispatch rides* mode replays a simulated
  morning: cars coloured by what they are doing (orange when carrying two
  riders) move along their real routes, riders wait and are collected, surge
  squares show their multipliers, and every method's running averages update
  side by side. Only the decisions are stored; the browser
  rebuilds every route with the same hierarchy, and a test holds each leg to
  the duration the simulation recorded.
* **Built and deployed by CI.** On each push, a workflow downloads central
  London from OpenStreetMap (cached between runs), snapshots TfL's
  disruptions, builds the graph, contracts it for each preset, trains the
  ETA models, simulates dispatch and publishes `web/` to GitHub Pages.

### 11. What the London numbers show

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

### 12. Measuring fairly

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

# Alternatives, the Congestion Charge, and live TfL closures
python -m fastroute route 51.5265,-0.1820 51.5100,-0.0700 --at 08:00 --alternatives --avoid-charge
python -m fastroute live
python -m fastroute route 51.5130,-0.1590 51.5160,-0.1300 --at 13:00 --live

# Where can I get to in 20 minutes? Best order for a round of deliveries
python -m fastroute reach 51.5174,-0.1200 --at 08:00
python -m fastroute tour 51.5174,-0.1200 51.5308,-0.1238 51.5033,-0.1145 51.5154,-0.1410

# Rush hour, with turn costs and traffic lights
python -m fastroute route 51.5154,-0.1410 51.5033,-0.1196 --at 08:30 --turns -a astar
python -m fastroute eta 51.5154,-0.1410 51.5033,-0.1196 --turns

# Contraction Hierarchies need preprocessing per scenario
python -m fastroute contract --at 08:30 --turns
python -m fastroute bench --at 08:30 --turns

# Simulate trips, train and compare ETA models
python -m fastroute eta-train --trips 20000 --out data/eta-model.json

# Simulate ride dispatch on the morning peak and compare methods
python -m fastroute dispatch --drivers 115 --days 6
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
the learned model beats both the router and a global factor, the calibrated
window reaches its target coverage, the conformal margin fixes a
deliberately overconfident interval, and the learned slowdown map lines up
with the hidden hotspots. Two Node tests
hold the browser to the Python implementation: all routing algorithms, with
and without turn costs, and the ETA features and predictions.

Dispatch has its own: the Hungarian algorithm against brute force on random
matrices, many-to-many travel times against Dijkstra for every pair, demand
following its daily rhythm and staying reproducible, forecast features never
reading the slot they predict, the forecast beating both baselines, every
policy running on identical requests and never exceeding the pickup limit,
and a Node test holding the browser's replayed routes to the simulation.
Shared rides keep every rider within the detour limits and every driver's
legs back to back; surge only ever prices out riders under the surge policy.

The newer pieces are held the same way. Customizable CH must equal Dijkstra
after any re-customisation, including random closures; TfL feeds are parsed
in JSON and WKT forms and closures land on the right roads; alternative
routes start with the quickest, stay within the stretch limit and keep out
of the zone when asked; travel-time maps and delivery rounds match Dijkstra
and brute force; the graph neural network's hand-written gradients match
finite differences. Node tests hold the browser's alternatives, charge zone,
Customizable CH, disruption matching, travel-time maps and tours to Python.

**Property-based tests** (Hypothesis) generate random road networks, with
one-way traps, parallel roads, unreachable targets and near-zero lengths,
and require every search method, Customizable CH under arbitrary new times
and closures included, to match Dijkstra; quickest times to obey the
triangle inequality; and the Hungarian algorithm and Held-Karp to match
brute force.

CI runs lint, formatting, type checking (mypy) and the full suite with
coverage (92% of lines) on Python 3.10, 3.12 and 3.13.

## Roadmap

- [x] **Road graph and Dijkstra:** OSM pipeline, travel-time model, CSR graph, tested baseline
- [x] **A\* and bidirectional search:** admissible straight-line heuristic, meet-in-the-middle stopping rule
- [x] **Contraction Hierarchies:** node ordering, witness search, shortcuts, upward query, path unpacking
- [x] **Traffic and turns:** time-of-day speed profiles, edge-expanded graph with turn and traffic-light costs
- [x] **Live map:** in-browser routing on a street-atlas render of the graph, animating each algorithm's search
- [x] **Live closures:** TfL disruption feed, map matching, Customizable CH with nested dissection and triangle customization
- [x] **Alternative routes and the Congestion Charge:** penalty method with stretch, overlap and detour checks; zone avoidance
- [x] **Travel-time maps and delivery rounds:** bounded one-to-all from or to a point, Held-Karp and 2-opt/Or-opt on asymmetric matrices
- [x] **Ride dispatch and demand forecasting:** many-to-many CH, Hungarian matching, ML demand forecast, repositioning, simulated and replayed
- [x] **Learned ETA:** simulated observed trips, route features, ridge, MLP and gradient-boosted trees, calibrated arrival windows, a learned congestion map, shipped to the browser
- [x] **Graph neural network ETA:** line-graph message passing over every road, per-segment slowdowns, hand-derived gradients
- [x] **Shared rides and surge pricing:** en-route joining with detour limits, batch pairing, price multipliers with rider elasticity

## Data

Road data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
available under the Open Database License.

Live road disruptions: Powered by TfL Open Data. Contains OS data © Crown
copyright and database rights 2016 and Geomni UK Map data © and database
rights 2019.

## License

MIT
