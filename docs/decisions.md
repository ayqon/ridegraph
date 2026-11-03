# Design decisions

Each entry records a choice, what else was on the table, and why it went
this way. The costs are listed too, because every one of these has some.

## Implement the algorithms instead of calling a library

* **Alternatives:** NetworkX, OSRM, GraphHopper.
* **Why:** the point is to understand where routing speed comes from, and to be able to change anything (turn costs, live times, the browser port).
* **Cost:** more code to trust, which is why every method is tested against Dijkstra, property-tested on random graphs, and compared with OSRM.

## Seconds, not metres, on every edge

* **Alternatives:** distance with a speed applied afterwards.
* **Why:** "quickest" depends on speed, which depends on road class, limits and time of day. Putting seconds on the edges makes every algorithm optimise the right thing with no special cases.
* **Cost:** a new set of weights per time of day, and preprocessing per set.

## Flat CSR arrays and Python lists in hot loops

* **Alternatives:** an object per node, NetworkX graphs, numpy indexing in loops.
* **Why:** CSR is compact, exports to the browser byte for byte, and Python lists are several times faster than numpy scalar access inside a heap loop.
* **Cost:** less readable than objects; documented at each array.

## Both Contraction Hierarchies and Customizable CH

* **Alternatives:** CH only, or CCH only.
* **Why:** CH with witness searches gives the smallest, fastest hierarchy for a fixed set of times (97 junctions settled per query). CCH is larger but its times can be replaced in milliseconds, which is what live closures need. The map uses CH by default and steps over to CCH when live times are on.
* **Cost:** two hierarchies to build and test.

## Geometric nested dissection for CCH

* **Alternatives:** METIS, KaHIP, inertial flow.
* **Why:** no native dependency, and on London a median cut in the best of four directions gives 3.9 times the original edges and 3.2 million triangles, small enough to customise in 20 ms in the browser.
* **Cost:** a graph partitioner would give fewer shortcuts; the gap is acceptable at city scale.

## A browser port instead of a server

* **Alternatives:** a routing API that the page calls.
* **Why:** a static site costs nothing to host, cannot go down, and lets every search be animated junction by junction.
* **Cost:** two implementations; Node tests hold the JavaScript to Python on the same exported data.

## Simulated trips for the learned ETA

* **Alternatives:** real trip data, which is not available to a personal project.
* **Why:** a hidden world with hotspots, a later rush hour and waits at lights and turns, calibrated to London's average speeds, lets the models be scored honestly against a truth they never see, including whether they rediscover the hotspots.
* **Cost:** results describe a believable London, not the real one. The writing says so wherever numbers appear.

## Ship the trees, report the graph neural network

* **Alternatives:** ship the GNN.
* **Why:** the GNN is slightly more accurate (6.6% against 6.8%) but needs a learned representation of every road, about 60 MB, against a few hundred small trees.
* **Cost:** the browser shows the second-best model; the model card says so.

## Calibrate the arrival window with conformal prediction

* **Alternatives:** raw quantile models.
* **Why:** raw 10th and 90th percentile models covered 77% of trips against an 80% target. Conformalized quantile regression widens the window by exactly the margin that restores coverage on held-out data (81%).
* **Cost:** a slice of training data held back for calibration.

## Compare dispatch policies on identical requests

* **Alternatives:** separate random runs per policy.
* **Why:** the same riders, the same fleet and the same willingness to pay make differences between policies attributable to the policy alone.
* **Cost:** one simulated morning per policy per day, run in CI.

## Report surge pricing without supply response

* **Alternatives:** model extra drivers coming online when prices rise.
* **Why:** any supply curve would be an assumption that decides the answer. The simulation isolates rationing and says plainly that this is all it shows.
* **Cost:** surge looks worse than its real-world case, which rests on supply. 