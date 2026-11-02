// Best order to visit several stops (mirrors fastroute/tour.py).
// Stop 0 is the depot; the matrix is asymmetric travel time in seconds.

export const MAX_EXACT_STOPS = 13;

export function tourCost(m, order, returnToStart) {
  let total = 0;
  for (let i = 1; i < order.length; i++) total += m[order[i - 1]][order[i]];
  if (returnToStart && order.length > 1) total += m[order[order.length - 1]][order[0]];
  return total;
}

export function nearestNeighbour(m) {
  const order = [0];
  const left = new Set(m.map((_, i) => i).slice(1));
  while (left.size) {
    const here = order[order.length - 1];
    let best = -1;
    for (const j of left) if (best < 0 || m[here][j] < m[here][best]) best = j;
    order.push(best);
    left.delete(best);
  }
  return order;
}

export function improve(m, start, returnToStart) {
  let best = start.slice();
  let bestCost = tourCost(m, best, returnToStart);
  const n = best.length;
  let improved = true;
  while (improved) {
    improved = false;
    for (let i = 1; i < n - 1; i++) {
      for (let k = i + 1; k < n; k++) {
        const cand = [...best.slice(0, i), ...best.slice(i, k + 1).reverse(), ...best.slice(k + 1)];
        const c = tourCost(m, cand, returnToStart);
        if (c < bestCost - 1e-9) {
          best = cand;
          bestCost = c;
          improved = true;
        }
      }
    }
    for (const length of [1, 2, 3]) {
      for (let i = 1; i <= n - length; i++) {
        const run = best.slice(i, i + length);
        const rest = [...best.slice(0, i), ...best.slice(i + length)];
        for (let j = 1; j <= rest.length; j++) {
          const cand = [...rest.slice(0, j), ...run, ...rest.slice(j)];
          const c = tourCost(m, cand, returnToStart);
          if (c < bestCost - 1e-9) {
            best = cand;
            bestCost = c;
            improved = true;
          }
        }
      }
    }
  }
  return best;
}

export function heldKarp(m, returnToStart) {
  const n = m.length;
  if (n <= 2) return m.map((_, i) => i);
  if (n > MAX_EXACT_STOPS) throw new Error(`exact search is limited to ${MAX_EXACT_STOPS} stops`);
  const k = n - 1;
  const full = (1 << k) - 1;
  const cost = new Float64Array((1 << k) * k).fill(Infinity);
  const parent = new Int32Array((1 << k) * k).fill(-1);
  for (let j = 0; j < k; j++) cost[(1 << j) * k + j] = m[0][j + 1];
  for (let subset = 1; subset <= full; subset++) {
    for (let j = 0; j < k; j++) {
      const here = cost[subset * k + j];
      if (!(subset & (1 << j)) || !Number.isFinite(here)) continue;
      for (let q = 0; q < k; q++) {
        if (subset & (1 << q)) continue;
        const next = subset | (1 << q);
        const c = here + m[j + 1][q + 1];
        if (c < cost[next * k + q]) {
          cost[next * k + q] = c;
          parent[next * k + q] = j;
        }
      }
    }
  }
  let end = 0;
  let bestEnd = Infinity;
  for (let j = 0; j < k; j++) {
    const c = cost[full * k + j] + (returnToStart ? m[j + 1][0] : 0);
    if (c < bestEnd) {
      bestEnd = c;
      end = j;
    }
  }
  const order = [];
  let subset = full;
  let j = end;
  while (j >= 0) {
    order.push(j + 1);
    const prev = parent[subset * k + j];
    subset &= ~(1 << j);
    j = prev;
  }
  return [0, ...order.reverse()];
}

/** Every method's order and total, as fastroute.tour.plan does. */
export function planTour(m, returnToStart) {
  const orders = {
    "As entered": m.map((_, i) => i),
    "Nearest next stop": nearestNeighbour(m),
  };
  orders["Improved (2-opt + Or-opt)"] = improve(m, orders["Nearest next stop"], returnToStart);
  if (m.length <= MAX_EXACT_STOPS) orders["Optimal (Held-Karp)"] = heldKarp(m, returnToStart);
  const seconds = Object.fromEntries(Object.entries(orders).map(([k, o]) => [k, tourCost(m, o, returnToStart)]));
  return { orders, seconds };
}
