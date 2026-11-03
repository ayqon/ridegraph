# Changelog

All notable changes to fastroute. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/).

## [1.0.0] - 2026-11-03

First public release.

### Routing
- Road graph from OpenStreetMap: drivable ways, one-way rules, largest
  strongly connected component, travel times from road class and limits.
- Dijkstra, bidirectional Dijkstra, A\* and Contraction Hierarchies, all
  tested against Dijkstra; Contraction Hierarchies settle 389 times fewer
  junctions than Dijkstra on central London.
- Time-of-day traffic profiles and turn costs through an edge-expanded graph.
- Customizable Contraction Hierarchies with nested dissection; new travel
  times are loaded in about 50 ms in numpy and 20 ms in the browser.
- Live closures from Transport for London's disruption feed, map-matched
  onto the graph.
- Alternative routes by the penalty method, and Congestion Charge zone
  avoidance.
- Travel-time maps (leaving or arriving) and delivery rounds (Held-Karp,
  2-opt and Or-opt on asymmetric matrices).
- A head-to-head with OSRM in CI: travel times within 1% on 84% of 1,000
  random London trips.

### Machine learning
- Learned ETA on simulated trips: ridge, MLP and gradient-boosted trees,
  with conformally calibrated arrival windows and a learned congestion map.
- Graph neural network over every road segment, in numpy with hand-derived
  gradients.
- Poisson gradient-boosted demand forecast per 500 m square.

### Dispatch
- Many-to-many Contraction Hierarchies and the Hungarian algorithm.
- Six policies on identical requests: nearest driver, optimal matching,
  repositioning on historical averages or on the ML forecast, shared rides
  and surge pricing.

### Live map
- In-browser port of every algorithm on a canvas street atlas, held to
  Python by Node tests; five modes including a dispatch replay.

### Engineering
- Property-based tests, mypy, 92% line coverage, CI on Python 3.10, 3.12 and
  3.13, documentation site, automated build and deployment.

[1.0.0]: https://github.com/ayqon/fastroute/releases/tag/v1.0.0
