# Changelog

All notable changes to Ridegraph. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- Turn restrictions from OpenStreetMap ("no right turn", "only straight on"),
  obeyed by turn-aware routing in Python and in the browser.
- Delivery stops and the travel-time origin can be dragged on the map; a
  click moves the nearer route pin once both are placed.

### Fixed
- Route pins no longer reappear in other tabs when the map is moved.

## [1.0.0] - 2026-11-05

First public release.

### Cities
- Central London, Nicosia and Manhattan from the same code; each city supplies
  its area, landmarks, demand centres and which side of the road traffic
  drives on. Turn costs are mirrored for right-hand traffic.

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
  Python by Node tests; route planning, travel-time maps, delivery rounds and
  a dispatch replay, with a city switcher.

### HTTP API
- FastAPI service for routes, travel-time areas and delivery rounds at any
  time of day on Customizable Contraction Hierarchies, with GeoJSON
  responses and a Docker image.

### Validation
- Simulated trip speeds by time of day set beside Department for Transport
  and TomTom figures for central London.

### Engineering
- Property-based tests, mypy, 92% line coverage, CI on Python 3.10, 3.12 and
  3.13, documentation site (including a page on road rules and what the
  router does not model yet), automated build and deployment.

[Unreleased]: https://github.com/ayqon/ridegraph/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ayqon/ridegraph/releases/tag/v1.0.0
