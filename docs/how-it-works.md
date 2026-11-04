# How it works

## Three cities, one engine

Nothing in the engine is London-specific. `fastroute/cities.py` gives each
city its area, the places a person would name on the map, where ride demand
concentrates and which side of the road traffic drives on; the same
pipeline downloads, builds, contracts, trains and simulates each one, and
the map switches between them, keeping the view inside the chosen city.

| City | Area | Extras |
|---|---|---|
| **Central London** | Paddington to Whitechapel, King's Cross to Vauxhall | Congestion Charge, live TfL closures, checked against OSRM and published speeds |
| **Nicosia** | The south of the city, stopping short of the buffer zone so no route crosses the Green Line | Drives on the left, like London |
| **Manhattan** | Battery to Central Park South, with the bridges to Brooklyn and Queens | Drives on the right, so turn costs are mirrored |

The time-of-day traffic pattern is London's everywhere; only London's has
been checked against real figures.

## 1. The map is a graph

Intersections become **nodes** and road segments become directed **edges**.
A two-way street is two edges; a one-way street or roundabout is one.

The pipeline (`fastroute/osm.py`) pulls drivable ways from OpenStreetMap,
drops what a car cannot use (footpaths, private roads, car park aisles),
applies `oneway`, roundabout and motorway direction rules, then keeps only the
largest strongly connected component. That last step matters: cutting a city
out of a bounding box leaves stubs you can drive into but never leave, and a
router that starts there would fail.

## 2. Edges cost seconds, not metres

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

## 3. Four ways to search

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

## 4. Traffic and turns

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

## 5. Live closures and Customizable Contraction Hierarchies

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

## 6. Alternative routes and the Congestion Charge

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

## 7. The learned ETA

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

**Results** are measured on 4,000 trips the models never saw:

| Model | Mean abs. error | 90th pct. error | Within 10% |
|---|---:|---:|---:|
| Router alone | 27.5% | 38.5% | 2% |
| Router times one global factor | 10.1% | 20.3% | 58% |
| Ridge regression | 7.6% | 15.5% | 74% |
| Neural network (MLP, 64x64) | 7.3% | 14.4% | 75% |
| Gradient-boosted trees (shipped to the map) | 6.8% | 13.8% | 78% |
| Graph neural network over every road | 6.6% | 13.4% | 80% |

The
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

**How realistic is the simulated city?** `python -m fastroute validate`
times random weekday trips (about 6 km) in it and sets the speeds beside
published figures:

| Weekday time | Simulated trips | Router's estimate |
|---|---:|---:|
| 03:00 | 20.9 km/h | 25.4 km/h |
| 08:00 | 11.1 km/h | 16.2 km/h |
| 13:00 | 13.7 km/h | 19.6 km/h |
| 17:30 | 10.9 km/h | 16.8 km/h |

| Published figure | Source | Speed |
|---|---|---:|
| City of London, local A roads | DfT 2022 (CGN0503d) | 12.4 km/h |
| Camden, local A roads | DfT 2022 (CGN0503d) | 13.0 km/h |
| London city centre, all day | TomTom Traffic Index 2023 | 16.1 km/h |
| London city centre, peak hours | TomTom Traffic Index 2023 | 13.7 km/h |

Midday in the simulated city (13.7 km/h) sits right among the official
figures for the boroughs it covers, and the router's own daytime estimate
(16 to 20 km/h) is close to TomTom's all-day figure, which is what a router
that knows nothing about lights or queues should look like. The simulated
rush hours run about 20% slower than TomTom's peak figure, so if anything
the models are trained and tested on a harsher London than the real one.
The definitions differ (DfT counts A roads only, TomTom whole trips in its
own city-centre area), so these are a range to sit within, not a target to
fit.

**In the browser.** The trees export to JSON, and `web/js/eta.js` computes the
same features and walks the same trees. A test runs it under Node and
requires its features and predictions to match Python's to nine decimal
places.

## 8. Travel-time maps and delivery rounds

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

## 9. Ride dispatch and demand forecasting

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

## 10. The live map

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

## 11. An HTTP API

The map needs no server, but a product would. `fastroute/api.py` serves the
engine as a FastAPI application with OpenAPI docs at `/docs`:

| Endpoint | Answers |
|---|---|
| `GET /route?from=lat,lon&to=lat,lon&at=08:15` | The quickest route as GeoJSON, with distance, Congestion Charge and the learned ETA with its arrival window; `alternatives=2` and `avoid_charge=true` add more routes |
| `GET /reach?from=lat,lon&minutes=10,20,30&arriving=false` | Area reachable within each time |
| `POST /tour` | Best visiting order for up to 13 stops, every method compared |
| `GET /health` | What is loaded |

**Any minute of the day.** The browser offers four preset times because each
plain hierarchy is built for one set of travel times. The API answers for
any time by keeping one Customizable CH and loading each quarter-hour's
times into it on first use (tens of milliseconds), with recent quarter-hours
cached, including the plain-list views the search loop reads.

**Measured** with `scripts/loadtest.py` on central London, one worker, on a
2-vCPU machine that also ran the load generator: **14 ms** median and 21 ms
95th-percentile latency for a route with its learned ETA, about **70
requests a second** per worker, no errors over 2,000 random requests spread
over six quarter-hours. Start-up, including building the hierarchy, takes 9 s.
More throughput is more workers.

```bash
pip install -e ".[api,ml]"
python -m fastroute serve --eta-model web/data/eta-model.json
curl "http://127.0.0.1:8000/route?from=51.5154,-0.1410&to=51.5033,-0.1145&at=08:15"

docker build -t fastroute . && docker run -p 8000:8000 fastroute
```

A CI workflow builds the image and checks the running container answers.

## 12. What the London numbers show

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

## 13. Measuring fairly

Benchmarks report **nodes settled** alongside wall-clock time. Settled nodes
measure the work each algorithm does, independent of language; Python
timings mostly measure interpreter overhead per node, which is why
Contraction Hierarchies' 24x reduction in work shows up as a smaller time
gain here. Every benchmark query is also cross-checked against Dijkstra, so a
benchmark run doubles as a correctness check.

**Against OSRM.** Agreeing with Dijkstra proves the searches are right, but
they all share this package's map parsing and speed model. OSRM shares
neither: it parses OpenStreetMap with its own C++ code and routes with
Multi-Level Dijkstra. The *Compare with OSRM* workflow (`fastroute/osrm.py`)
gives OSRM v5.27 the ways this package treats as drivable, loads fastroute's
free-flow speed for every one of the 127,286 road segments as a *segment
speed file*, switches off OSRM's turn and traffic-light penalties (the
free-flow router has none) and asks both engines the same 1,000 random
trips:

| fastroute vs OSRM, 1,000 random central London trips | |
|---|---:|
| Travel time within 1% of OSRM's | 83.7% |
| Travel time within 5% of OSRM's | 96.9% |
| Median time difference | 0.23% |
| Same route (95%+ of junctions shared) | 75.0% |
| Median share of junctions in common | 99.7% |
| Median query, fastroute CH in Python | 1.02 ms |
| Median query, OSRM over local HTTP | 1.56 ms |

What remains comes from the parts that are genuinely independent. OSRM
snaps a coordinate to the nearest point on a road rather than to a
junction, stores each segment's time in tenths of a second, and applies its
own rules about which ways a car may use. Where the routes differ but the
times agree, the two engines have found different routes that are equally
quick. The query times are not a like-for-like contest (OSRM's include an
HTTP round trip), but they show the Python hierarchy is in the same league
as a production C++ engine on a city of this size.
