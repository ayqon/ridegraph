# Road rules and driving behaviour

A route is only useful if a driver can legally follow it, and an arrival time
is only useful if it reflects how people actually drive. This page sets out
the rules and behaviours that matter for a ride-hailing or delivery router in
a city like London, what Ridegraph models today, what it does not model yet,
and how each would fit into the engine.

Two kinds of rule are kept apart throughout. **Legal rules** decide whether a
route is allowed at all, so they belong in the road graph: a banned turn is
removed, not made expensive. **Driving behaviour** decides how long a legal
route takes or how pleasant it is, so it belongs in the costs: a right turn
across traffic is allowed, but it takes longer.

| Rule | Kind | In Ridegraph today |
|---|---|---|
| One-way streets, roads closed to cars | Legal | Yes |
| Speed limits | Legal | Yes, from OpenStreetMap where tagged |
| Live closures and roadworks | Legal | Yes, from TfL's feed (London) |
| Congestion Charge | Legal (a charge) | Yes, priced and avoidable |
| Turn restrictions | Legal | Yes, when turn costs are on |
| Bus lanes and bus gates | Legal | Not yet |
| Time-limited restrictions | Legal | Not yet |
| ULEZ | Legal (a charge) | Not yet |
| Where a car may stop | Legal | Not yet |
| Turning across traffic | Behaviour | Yes, as a time cost |
| Waiting at traffic lights | Behaviour | Yes, as one average wait |
| Kerbside pickup | Behaviour | Not yet |
| Time spent at each stop | Behaviour | Not yet |

## Legal rules

### Turn restrictions

Signs such as "No right turn", "No U-turn" and "Only straight on" are
recorded in OpenStreetMap as restriction relations: a road you arrive on, the
junction (or short stretch of road) you pass through, and a road you may not,
or must, leave by. Central London has thousands of them.

Ridegraph downloads these with the roads and obeys them whenever turn costs
are switched on. Each restriction is reduced to the node a driver comes
from, the junction and the node they may not continue to (`osm.py`), and
an "only straight on" sign becomes a ban on every other way out, including
turning back.

They fit naturally into the turn-aware graph Ridegraph builds for turn costs
(`turns.py`). In that graph every road segment is a node and every possible
move from one segment to the next is an edge, so a banned turn is simply an
edge that is never created. The browser keeps the same edges, so its
numbering matches Python's, but gives forbidden ones an infinite cost.

Two limits remain. A plain junction-to-junction graph cannot express a turn
ban at all, so the fastest search methods, which run on that graph, ignore
them; switch on turn costs for legal routes. And restrictions that run
through a stretch of road rather than a single junction ("via ways") are
rare and not read yet; they need the same idea applied to short chains of
segments.

### Bus lanes and bus gates

Many London streets have lanes, or whole short sections, reserved for buses
during some or all of the day. London's licensed black cabs may use most bus
lanes; private hire cars, the category Uber drivers fall into, may not. The
same journey can therefore have different quickest routes, and different
times, depending on the vehicle.

The clean way to model this is with vehicle profiles: one road graph, and a
small table saying which roads and lanes each vehicle type may use and when.
OpenStreetMap tags bus lanes and bus-only roads (`busway`, `bus:lanes`,
`access=no` with `psv=yes`), so a taxi profile and a private hire profile
could be compared on the same trip.

### Time-limited restrictions

Some rules depend on the clock. School streets close to traffic around the
start and end of the school day, some low-traffic neighbourhoods are filtered
only at certain times, and "no entry except for access" roads are legal only
if your trip starts or ends on them.

Ridegraph already prices travel by time of day, and its Customizable
Contraction Hierarchies can load a new set of travel times in about 20 ms, so
a time-limited closure can be applied in the same way as a live closure: set
the affected roads to impassable for the hours the rule applies and
re-customise. Access-only roads need a different treatment, allowing them
only as the first or last part of a route.

### Charges and zones

The Congestion Charge is modelled: routes can be priced, and a route that
stays out of the zone can be requested. The Ultra Low Emission Zone (ULEZ)
now covers the whole of Greater London and charges older, more polluting
vehicles each day they drive in it. Most ride-hailing cars meet its standard,
so it is better treated as a property of the vehicle profile than of the
route: a non-compliant vehicle pays it on any London trip.

### Speed limits

Much of central London now has a 20 mph limit: the City of London, boroughs
such as Camden and Islington, and the main TfL roads within the Congestion
Charge zone. Ridegraph reads the posted limit from OpenStreetMap where it is
tagged and assumes traffic flows somewhat below it; where a road has no
limit tagged, it falls back on a typical speed for that class of road. The
estimate is only as good as the tagging, so the fallback matters most on
minor streets.

### Where a car may stop

A pickup point is only useful if a driver may legally stop there. Red routes
(major roads marked with red lines and managed by TfL) restrict stopping, as
do double red lines, bus stops and zig-zag markings near crossings. A good
pickup point is a short walk from the rider on a street where stopping is
allowed, not the exact point they tapped.

Ridegraph currently snaps a pickup to the nearest junction. Choosing legal
stopping points needs kerb data that OpenStreetMap only partly records, so
this is the hardest item here to do well.

## Driving behaviour (cost, not law)

### Turning across traffic

Turning across oncoming traffic, right in the UK and left in most of the
world, means waiting for a gap. It is slower, more likely to cause a
collision and wastes fuel while idling. UPS is well known for planning its
delivery routes to avoid these turns where it can.

Ridegraph's turn costs already reflect this: in the UK a right turn costs
more than a left, a sharp turn more again, and the costs swap sides for
cities that drive on the right (Manhattan). A delivery-van profile could go
further and make right turns so expensive that they are only used when there
is no reasonable alternative.

### Kerbside pickup

When a rider is waiting on one side of a busy road, a car that arrives on the
other side forces them to cross, or forces the driver into a U-turn. Uber and
other ride-hailing apps try to send the driver to the rider's side of the
road. In routing research this is known as the curb approach: a stop is not
just a place but a place and a side, and the vehicle must arrive travelling
in the direction that puts the kerb next to it.

To model it, a pickup becomes a point on a directed road segment rather than
a junction. In the UK the rider should be on the car's left, so only one
direction of travel along that segment is acceptable. The search then has to
reach that segment from the right end, which can add a minute or two to the
pickup but saves the rider a crossing. The dispatch simulator could then
report how much extra driving kerbside pickups cost the fleet, and what it
saves riders.

### Waiting at traffic lights

Ridegraph adds the same average wait, 12 seconds, at every set of traffic
lights. Real waits depend on the signal's cycle length, how much of the
cycle is green for your direction, and how busy the junction is. Webster's
formula, a classic result from traffic engineering, estimates the average
delay from exactly those three things, and rises steeply as a junction
approaches capacity.

Signal timings are not public for most junctions, but the formula would
still let the wait grow at peak times and on busier roads instead of staying
flat. The learned arrival-time model already corrects part of this from
data: the number of traffic lights on a route is one of its most useful
features.

### Time spent at each stop

Every pickup and drop-off takes time: pulling in, the rider getting in or
out, luggage, payment. For a single ride it is small; for a delivery round
with a dozen stops, or a shared ride that picks up twice, it adds up. A fixed
allowance of one to two minutes per stop is a common starting point. Neither
the dispatch simulator nor the delivery planner includes it yet: pickups and
drop-offs are instant, so their totals are driving time only.

## Order of work

1. Kerbside pickup, because it is the most distinctive ride-hailing problem
   here and changes what the dispatch results mean.
2. Vehicle profiles (taxi, private hire, delivery van), which bring bus
   lanes, ULEZ and stronger turn avoidance in together.
3. Time-limited restrictions, using the same mechanism as live closures.
4. Signal delays and stop times, which refine the arrival-time estimates.
