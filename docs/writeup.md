# Rebuilding a ride-hailing routing stack on London's roads

*What I learned implementing quickest routes, live closures, learned arrival
times and ride dispatch from first principles, and what the numbers say.*

Every ride-hailing or maps app answers the same two questions millions of
times a second: what is the quickest way from A to B, and when will I get
there? Behind them sits a third, the one the business runs on: which driver
should take this ride? I built all three from scratch on OpenStreetMap's map
of central London (72,637 junctions, 127,286 road segments) to see where the
real difficulty lies. Everything is in [the repository](../README.md), and
[the live map](https://ayqon.github.io/fastroute/) runs the same algorithms
in the browser.

## 1. Fast routes come from preprocessing, not cleverness at query time

Dijkstra's algorithm settles 37,822 junctions on an average London query. A\*,
the textbook improvement, still settles 23,705: its straight-line estimate
assumes the fastest road speed, and London's routes run at about half of
it. **Contraction Hierarchies** settle 97, a 389-fold reduction, because
they move the work to preprocessing: junctions are ranked by importance and
shortcuts added so every query climbs from each end towards main roads and
meets in the middle. The gain is far larger on London than on a uniform
grid (24-fold), because real roads are hierarchical. That is the whole
idea.

To check the engine against something it did not write itself, CI runs OSRM,
the open-source engine behind many commercial maps, on the same roads and
speeds. On 1,000 random trips the travel times agree within 1% for 84% of
trips (median difference 0.23%), and the routes share a median 99.7% of
their junctions. The differences left come from independent map parsing
and from how each engine snaps a coordinate to the road.

## 2. Live traffic needs the layout and the times kept apart

A Contraction Hierarchy is built for one set of travel times. A closure on
Oxford Street makes it stale, and rebuilding takes seconds. **Customizable
Contraction Hierarchies** split the work: the road layout is ordered once by
nested dissection (cut the city along the line that crosses fewest roads,
recurse, rank the cut last) and contracted with every possible shortcut.
New times are then loaded by sweeping 3.2 million "lower triangles"
bottom-up, which takes **50 ms in numpy and about 20 ms in the browser**. The
map fetches Transport for London's live closures, matches them onto the
graph, re-customises and routes around them. Nothing about the query
changes; only the cost of keeping it current does.

## 3. A good alternative route is a constrained search problem

Showing a second route sounds trivial until it is a pointless detour or 95%
the same road. The penalty method (make the roads on each route found 40%
slower, search again) finds candidates; the rules decide which to keep: at
most 30% slower, at most 65% shared, and no stretch more than 15% slower
than the quickest way between its ends. Closing every road into the
Congestion Charge zone is the same machinery with infinite penalties. From
Paddington to Aldgate at 8am, staying out of the zone costs 14 minutes and
saves £18.

## 4. Machine learning should correct the router, not replace it

Uber's DeepETA keeps the routing engine and learns its error. I did the same
on simulated trips through a city calibrated to London (about 21 km/h at
night, 11 at the peaks) that hides congestion hotspots, a later rush hour
and waits at lights and turns. On 4,000 held-out trips:

| Model | Mean error | Within 10% |
|---|---:|---:|
| Router alone | 27.5% | 2% |
| Router x one global factor | 10.1% | 58% |
| Gradient-boosted trees on route features | 6.8% | 78% |
| Graph neural network over every road | 6.6% | 80% |

Three things stood out. Time of day and the number of sets of traffic lights
matter more than distance. Arrival *windows* need calibration: quantile
models alone covered 77% of trips against an 80% target, and conformal
calibration brought that to 81%. And the models genuinely discover what they
are not shown: the trees' map of slow areas correlates at r = 0.83 with the
hidden hotspots. The graph neural network, written in numpy with gradients
derived by hand, learns a slowdown for every road segment from its place in
the network and is slightly more accurate, but the trees are what ships to
the browser: a few hundred small trees against a representation of every
road.

## 5. In dispatch, the decision rule matters more than the forecast

The dispatch simulator replays the same morning of ride requests and the
same fleet under each policy. Every 30 seconds one many-to-many query
prices every free driver against every waiting rider.

| Policy (6 simulated mornings, 115 drivers) | Avg. wait | Gave up | Riders served | Driving per ride |
|---|---:|---:|---:|---:|
| Nearest free driver | 7.3 min | 5.9% | 94.1% | 23.8 min |
| Optimal matching (Hungarian algorithm) | 6.6 min | 6.2% | 93.8% | 23.4 min |
| + repositioning on the ML demand forecast | 6.1 min | 5.0% | 95.0% | 25.7 min |
| + shared rides | 5.1 min | 1.4% | 98.6% | 22.1 min |
| + surge pricing (instead of sharing) | 5.2 min | 0.3% | 91.3% | 25.8 min |

Optimal matching beats "nearest driver" by about 10% on waits, and moving idle
drivers ahead of predicted demand takes the gain to 16%. The demand model
halves the forecasting error of a historical average, yet its dispatch gain
over that average is small; it mostly shows up as a quarter less empty
repositioning. Better predictions do not automatically make proportionally
better decisions.

The two product levers were the most interesting result. **Shared rides**
serve more people with less driving: almost half of rides are shared,
fewer riders give up, and the fleet drives 14% less per rider served, for
about two extra minutes in the car. **Surge pricing** also shortens waits,
but by turning 8.5% of riders away, so fewer are served than with no surge
at all. With a fixed fleet, surge only rations; its case rests on the extra
drivers a higher price brings out, which is exactly what a simulation like
this has to model before it can say anything about surge.

## What I would build next

Turn-aware Customizable CH (the browser drops hierarchies when turn costs
are on), a repositioning rule learned by reinforcement rather than a
hand-set score, and real trip data in place of the simulated city, which is
the one thing a personal project cannot get.
