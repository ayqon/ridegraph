# fastroute

Quickest-route search on a real road network, built from first principles.

Ride-hailing and maps apps answer "what is the fastest way from A to B?"
millions of times a second. This project rebuilds that core on OpenStreetMap
data for central London: the routing algorithms are implemented by hand, not
called from a library, and every one is tested against a reference
implementation.

> **Status:** milestone 1 of 6. The road graph, travel-time model and
> Dijkstra baseline are done. A\*, Contraction Hierarchies, traffic and the
> live map demo are next ([roadmap](#roadmap)).

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
separate, so swapping in a traffic model later means rebuilding weights, not
re-downloading the map.

### 3. Compact storage

Edges are stored in CSR (compressed sparse row) form, the layout production
routing engines use: all edges leaving node `u` sit contiguously in
`head[first_out[u]:first_out[u + 1]]`.

### 4. Dijkstra: the baseline

`fastroute/dijkstra.py` grows a ball of settled nodes outward from the start in
order of travel time, so when the destination is settled its time is provably
optimal. It uses a binary heap with lazy deletion and only allocates state for
nodes it actually reaches.

Its weakness is the reason the rest of this project exists: the ball expands
in every direction, including away from the destination. Each algorithm is
measured by **nodes settled**, the honest measure of search effort, and later
milestones aim to cut that number by orders of magnitude.

## Quick start

```bash
pip install -e ".[dev]"

python -m fastroute fetch     # download central London from OpenStreetMap
python -m fastroute build     # turn it into a routing graph
python -m fastroute info

# Oxford Circus to Waterloo
python -m fastroute route 51.5154,-0.1410 51.5033,-0.1196
```

## Testing

```bash
pytest
```

Correctness is checked against NetworkX on randomised street grids with
one-way roads, alongside targeted cases: a faster road beating a shorter one,
one-way loops, parallel edges, unreachable targets, OSM tag parsing and
component pruning.

## Roadmap

- [x] **Road graph and Dijkstra:** OSM pipeline, travel-time model, CSR graph, tested baseline
- [ ] **A\* search:** straight-line distance at top speed as an admissible heuristic
- [ ] **Contraction Hierarchies:** preprocessing with shortcut edges, bidirectional query
- [ ] **Traffic:** time-of-day speed profiles and turn penalties
- [ ] **Live demo:** in-browser routing on a map, animating the nodes each algorithm explores
- [ ] **ETA model:** learning a correction to graph ETAs, the approach used in production ride-hailing

## Data

Road data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
available under the Open Database License.

## License

MIT
