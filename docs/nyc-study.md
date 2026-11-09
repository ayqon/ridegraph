# Arrival times on real New York taxi trips

Everything else in Ridegraph's arrival-time work is trained and tested on a
simulated city. This page reports the first test on real trips: yellow taxi
journeys in Manhattan, published by the New York City Taxi and Limousine
Commission (TLC). The question is simple: how far off is the router on real
trips, and how much does the learned correction help on a month it has
never seen?

## Data

TLC publishes every yellow taxi trip with its pickup and drop-off times, its
distance and the taxi zone it started and ended in. Zones are the finest
location the public data gives; Manhattan has about 70 of them, typically a
few hundred metres to a kilometre across.

| | |
|---|---|
| Train months | January, February and March 2025 |
| Test month | April 2025, never seen in training |
| Zones on the map | 50 (zones cut by the edge of the map are left out) |
| Trip records per month | 3.5 to 4.1 million (all of New York) |
| Usable per month | 1.8 to 2.1 million |
| Sampled per month | 40,000 |

A trip is usable if it starts and ends in different zones on the map, lasts
between one minute and three hours, and its recorded distance and duration
imply a speed between 1 and 80 km/h. Each trip is routed between a
representative junction of its pickup zone and one of its drop-off zone,
using the router's time-of-day speeds, and its recorded duration is the
target.

## Results on the test month

| Model | Mean abs. error | Mean abs. % error | 90th pct. % error | Within 10% |
|---|---:|---:|---:|---:|
| Zone-pair median, no routing | 209 s | 30.7% | 60.2% | 23% |
| Router alone | 362 s | 43.4% | 68.9% | 7% |
| Router x one overall factor | 242 s | 36.0% | 71.6% | 20% |
| Ridge regression | 220 s | 32.7% | 63.2% | 21% |
| Neural network (MLP) | 196 s | 28.4% | 55.1% | 25% |
| Gradient-boosted trees | 192 s | 28.1% | 55.1% | 25% |
| Gradient-boosted trees + zone history | 192 s | 28.2% | 55.1% | 25% |
| Graph neural network | 200 s | 29.1% | 56.9% | 24% |

The arrival window (10th to 90th percentile) contained 77.8% of test trips
against an 80% target, and is wide: 88% of the trip on average.

By recorded trip distance:

| Distance | Trips | Router alone | Gradient-boosted trees |
|---|---:|---:|---:|
| Under 2 km | 15,461 | 44.7% | 38.0% |
| 2 to 4 km | 15,841 | 45.2% | 23.4% |
| 4 to 7 km | 6,496 | 39.9% | 19.2% |
| Over 7 km | 2,202 | 32.6% | 18.8% |

## What this shows

**Real trips are much harder than the simulated city.** On simulated London
trips the best model was within 7% on average; on real Manhattan trips it
is 28%. The simulated city's only noise is what it was built with; real
trips add traffic incidents, double-parked vehicles, pickups that take
longer than expected, and drivers who do not follow the router's route.

**The routing adds less than it seems.** The fairest baseline needs no
routing at all: predict each trip as the median of earlier trips between
the same two zones in the same part of the day. That alone is 30.7% off,
against 43.4% for the router and 28.1% for the best model. So most of the
learned model's gain over the router is knowledge any taxi company already
has from its past trips; the route itself adds about 17 seconds a trip on
average. That is a real but modest improvement, and the honest headline of
this study.

**Adding the history to the model changes nothing.** Giving the trees the
zone-pair median as an extra input (taken, for each training month, from
the other months only, so no trip sees its own time) scores 28.2%, the same
as without it. The route features already carry what the history knows:
which zones a trip joins, how long the route is and when it runs.

**The learned correction still pays for itself against the router.** It cuts the router's error
by more than a third overall, and by roughly half on trips of 4 to 7 km.
Part of the router's error is a simple bias that one overall factor
removes, but only part: time of day, where the trip runs and its length all
matter.

**Zone-level locations set a floor on short trips.** On trips under 2 km the
model is still 38% off, against 19% on trips over 4 km. A trip's real start
and end can be several hundred metres from its zones' representative
junctions, which is a large share of a short trip and a small one of a long
trip. Some of the remaining error is not the model's to remove.

**The graph neural network is no better than the trees here.** On simulated
trips it was slightly ahead; on real ones it is slightly behind. With
zone-level ends, learning a slowdown for every road segment has less to
work with than it has in a city where every route is exact.

**The window is honest but not quite calibrated.** Conformal calibration on
earlier months barely moved coverage (77.0% to 77.8%), which suggests the
test month differs a little from the training months. Calibrating on the most recent weeks before the test month is the
obvious next thing to try.

## Limits and next steps

- **More months and other years,** to see whether the gap between training
  and test months is seasonal.
- **Dispatch on real demand** (issue #5): replaying real request times and
  zones through the dispatch policies.

## Reproducing

The study runs in GitHub Actions (the *New York study* workflow), which
downloads the TLC files and runs:

```bash
pip install -e ".[ml,nyc]"
python -m ridegraph fetch --city new-york
python -m ridegraph build --city new-york
python -m ridegraph nyc-eta \
  --zones data/nyc/taxi_zones.zip \
  --train data/nyc/yellow_tripdata_2025-01.parquet \
          data/nyc/yellow_tripdata_2025-02.parquet \
          data/nyc/yellow_tripdata_2025-03.parquet \
  --test data/nyc/yellow_tripdata_2025-04.parquet \
  --per-month 40000 --gnn-epochs 5 --out results/nyc-eta.json
```

Trip records and taxi zones: New York City Taxi and Limousine Commission,
[TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
