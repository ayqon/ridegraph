"""A graph neural network ETA: learn a slowdown for every road, from trips alone.

The gradient-boosted model (``eta.py``) sees a route as one row of summary
features. A graph neural network instead works on the roads themselves, the
way DeepMind's model for Google Maps does: it learns a representation of
every road segment from the segment and its neighbourhood in the network,
predicts how much slower than the router each segment really is at a given
time, and adds the segments of a route up.

Architecture (numpy only, gradients written out by hand):

* **Message passing, precomputed.** Each road segment starts with its own
  features: road class, length, traffic lights, junction size, and its
  position encoded with sines and cosines. Features are then averaged over
  the *line graph* (segments are neighbours when one leads into the other)
  for 1, 2, 4 and 8 hops with symmetric normalisation, and all hops are
  concatenated. Precomputing the propagation is the SIGN approach (Frasca et
  al., 2020): it scales to every road in a city, and a learned network on
  top decides which hop to trust for each road.
* **A segment network.** A two-layer MLP reads a segment's representation
  together with the time of day and predicts ``g``, the log of its slowdown.
* **A route readout.** Predicted time is the sum over the route's segments
  of ``router seconds x exp(g)``, plus a learned wait for each kind of turn
  and each set of traffic lights, also by time of day.

The loss is the squared error of log(predicted / observed), trained with
Adam on minibatches of trips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .graph import Graph, RoadNetwork

HOPS = (1, 2, 4, 8)
FREQS = (0.5, 1.0, 2.0, 4.0)  # cycles per km for the position encoding
HIDDEN = 48


def segment_features(
    network: RoadNetwork, graph: Graph, group: np.ndarray, x_km, y_km
) -> np.ndarray:
    """Own features of every segment (CSR edge order)."""
    head = np.asarray(graph.head)
    first = np.asarray(graph.first_out)
    tail = np.repeat(np.arange(network.num_nodes), np.diff(first))
    mid_x = (x_km[tail] + x_km[head]) / 2
    mid_y = (y_km[tail] + y_km[head]) / 2
    degree = np.diff(first)
    signals = np.asarray(network.traffic_signals, dtype=np.float64)
    cols = [
        (group == 0).astype(float),
        (group == 1).astype(float),
        (group == 2).astype(float),
        np.log1p(np.asarray(graph.length_m)),
        signals[head],
        np.log1p(degree[head]),
        mid_x / 4,
        mid_y / 4,
    ]
    for f in FREQS:
        for v in (mid_x, mid_y):
            cols.append(np.sin(2 * math.pi * f * v))
            cols.append(np.cos(2 * math.pi * f * v))
    return np.stack(cols, axis=1)


def line_graph(graph: Graph):
    """Normalised adjacency of segments: e follows f when f ends where e starts."""
    from scipy import sparse

    head = np.asarray(graph.head)
    first = np.asarray(graph.first_out)
    m = len(head)
    counts = np.diff(first)
    # Successors of e are the segments leaving head[e].
    succ_count = counts[head]
    rows = np.repeat(np.arange(m), succ_count)
    starts = first[head]
    before = np.cumsum(succ_count) - succ_count
    offsets = np.arange(succ_count.sum()) - np.repeat(before, succ_count)
    cols = np.repeat(starts, succ_count) + offsets
    a = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, m)).tocsr()
    a = a + a.T + sparse.identity(m, format="csr")
    a.data[:] = 1.0
    deg = np.asarray(a.sum(axis=1)).ravel()
    d = sparse.diags(1 / np.sqrt(deg))
    return (d @ a @ d).tocsr()


def propagate(features: np.ndarray, adjacency) -> np.ndarray:
    """Concatenate the features averaged over 0, 1, 2, 4 and 8 hops."""
    out = [features]
    current = features
    done = 0
    for hops in HOPS:
        for _ in range(hops - done):
            current = adjacency @ current
        done = hops
        out.append(current)
    return np.concatenate(out, axis=1)


def time_features(hour: np.ndarray, weekend: np.ndarray, level: np.ndarray) -> np.ndarray:
    a = 2 * np.pi * hour / 24
    return np.stack(
        [np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a), weekend.astype(float), level], axis=1
    )


@dataclass
class TripBatch:
    """Trips as segment occurrences: which trip, which segment, router seconds."""

    occ_trip: np.ndarray
    occ_edge: np.ndarray
    occ_seconds: np.ndarray
    counts: np.ndarray  # per trip: left, right, sharp, U-turns, light sets
    tau: np.ndarray  # per trip time features
    observed: np.ndarray


@dataclass
class SegmentGNN:
    H: np.ndarray  # propagated segment representations
    mean: np.ndarray
    std: np.ndarray
    params: dict[str, np.ndarray] = field(default_factory=dict)
    history: list[float] = field(default_factory=list)

    @classmethod
    def create(cls, H: np.ndarray, n_tau: int, n_counts: int, seed: int = 0) -> SegmentGNN:
        rng = np.random.default_rng(seed)
        mean, std = H.mean(axis=0), H.std(axis=0) + 1e-6
        d_in = H.shape[1] + n_tau
        params = {
            "W1": rng.normal(0, math.sqrt(2 / d_in), (d_in, HIDDEN)),
            "b1": np.zeros(HIDDEN),
            "W2": rng.normal(0, math.sqrt(1 / HIDDEN), (HIDDEN, HIDDEN)),
            "b2": np.zeros(HIDDEN),
            "w3": np.zeros(HIDDEN),
            "b3": np.zeros(1),
            "C": np.zeros((n_tau + 1, n_counts)),  # waits per turn kind, by time of day
        }
        params["C"][0, :] = math.log(math.e**5 - 1)  # softplus -> about 5 s to start
        return cls(H, mean, std, params)

    def _forward(self, b: TripBatch):
        p = self.params
        Hn = (self.H[b.occ_edge] - self.mean) / self.std
        Z = np.concatenate([Hn, b.tau[b.occ_trip]], axis=1)
        a1 = Z @ p["W1"] + p["b1"]
        h1 = np.maximum(a1, 0)
        a2 = h1 @ p["W2"] + p["b2"]
        h2 = np.maximum(a2, 0)
        g = np.clip(h2 @ p["w3"] + p["b3"][0], -3, 3)
        seg = b.occ_seconds * np.exp(g)
        n = len(b.observed)
        drive = np.bincount(b.occ_trip, weights=seg, minlength=n)
        tau1 = np.concatenate([np.ones((n, 1)), b.tau], axis=1)
        cz = tau1 @ p["C"]
        wait_each = np.logaddexp(0, cz)  # softplus
        waits = (b.counts * wait_each).sum(axis=1)
        total = drive + waits
        return Z, a1, h1, a2, h2, g, seg, tau1, cz, total

    def predict(self, b: TripBatch) -> np.ndarray:
        return self._forward(b)[-1]

    def segment_log_slowdown(self, edges: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """g for given segments at one time of day."""
        p = self.params
        Hn = (self.H[edges] - self.mean) / self.std
        Z = np.concatenate([Hn, np.repeat(tau[None, :], len(edges), axis=0)], axis=1)
        h1 = np.maximum(Z @ p["W1"] + p["b1"], 0)
        h2 = np.maximum(h1 @ p["W2"] + p["b2"], 0)
        return np.clip(h2 @ p["w3"] + p["b3"][0], -3, 3)

    def step(self, b: TripBatch, opt: dict, lr: float) -> float:
        p = self.params
        Z, a1, h1, a2, h2, g, seg, tau1, cz, total = self._forward(b)
        n = len(b.observed)
        r = np.log(total) - np.log(b.observed)
        loss = float(np.mean(r**2))
        d_total = 2 * r / (n * total)
        # Waits.
        sig = 1 / (1 + np.exp(-cz))
        grads = {"C": tau1.T @ (d_total[:, None] * b.counts * sig)}
        # Segments.
        d_seg = d_total[b.occ_trip]
        d_g = d_seg * seg * ((g > -3) & (g < 3))
        grads["w3"] = h2.T @ d_g
        grads["b3"] = np.array([d_g.sum()])
        d_h2 = np.outer(d_g, p["w3"]) * (a2 > 0)
        grads["W2"] = h1.T @ d_h2
        grads["b2"] = d_h2.sum(axis=0)
        d_h1 = (d_h2 @ p["W2"].T) * (a1 > 0)
        grads["W1"] = Z.T @ d_h1
        grads["b1"] = d_h1.sum(axis=0)
        # Adam.
        opt["t"] += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, gk in grads.items():
            gk = gk + (1e-5 * p[k] if k in ("W1", "W2") else 0)
            m = opt.setdefault(("m", k), np.zeros_like(gk))
            v = opt.setdefault(("v", k), np.zeros_like(gk))
            m[:] = b1 * m + (1 - b1) * gk
            v[:] = b2 * v + (1 - b2) * gk * gk
            mh = m / (1 - b1 ** opt["t"])
            vh = v / (1 - b2 ** opt["t"])
            p[k] -= lr * mh / (np.sqrt(vh) + eps)
        return loss


def batch(trips: list, idx: np.ndarray) -> TripBatch:
    """Gather selected trips (edges, router seconds, counts, time features, observed)."""
    edges = [trips.edges[i] for i in idx]
    secs = [trips.seconds[i] for i in idx]
    lengths = np.array([len(e) for e in edges])
    return TripBatch(
        occ_trip=np.repeat(np.arange(len(idx)), lengths),
        occ_edge=np.concatenate(edges),
        occ_seconds=np.concatenate(secs),
        counts=trips.counts[idx],
        tau=trips.tau[idx],
        observed=trips.observed[idx],
    )


def train_gnn(
    H: np.ndarray,
    trips,
    train_idx: np.ndarray,
    *,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 3e-3,
    seed: int = 0,
    log=lambda m: None,
) -> SegmentGNN:
    model = SegmentGNN.create(H, trips.tau.shape[1], trips.counts.shape[1], seed)
    rng = np.random.default_rng(seed)
    opt: dict = {"t": 0}
    for epoch in range(epochs):
        order = rng.permutation(train_idx)
        losses = []
        rate = lr * (0.5 * (1 + math.cos(math.pi * epoch / epochs)))  # cosine decay
        for s in range(0, len(order), batch_size):
            losses.append(model.step(batch(trips, order[s : s + batch_size]), opt, rate))
        model.history.append(float(np.mean(losses)))
        if epoch % 5 == 4:
            log(f"    epoch {epoch + 1}: training loss {model.history[-1]:.4f}")
    return model
