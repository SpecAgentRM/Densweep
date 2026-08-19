"""
Densweep — quickstart  /  szybki start
══════════════════════════════════════
Run:  python examples/quickstart.py

EN: Auto-select HDBSCAN's min_cluster_size with the smart-stopper, three ways.
PL: Automatyczny dobór min_cluster_size dla HDBSCAN ze smart-stopperem, na 3 sposoby.
"""
import os
import sys

import numpy as np

# Allow running straight from a checkout (no install needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from densweep import (
    Densweep, smart_sweep, hdbscan_cpu, make_mcs_range,
    StopperConfig, WINNING_CONFIG, adjusted_rand_score,
)


def toy_data(n=900, k=8, seed=0):
    """UMAP-embedding-like data: tight, well-separated 2D blobs."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-13, 13, size=(k, 2))
    sizes = rng.multinomial(n, np.ones(k) / k)
    X, y = [], []
    for ci, (c, s) in enumerate(zip(centers, sizes)):
        X.append(rng.normal(c, rng.uniform(0.35, 0.6), size=(s, 2)))
        y += [ci] * s
    return np.vstack(X), np.array(y)


def main():
    X, y_true = toy_data()
    print(f"data: {X.shape[0]} points, {len(set(y_true))} true clusters\n")
    print("winning stopper config:", WINNING_CONFIG, "\n")

    # ── 1. High-level estimator ───────────────────────────────────────────
    model = Densweep(algorithm="hdbscan").fit(X)
    print("[1] Densweep estimator")
    print(f"    best min_cluster_size : {model.best_min_cluster_size_}")
    print(f"    clusters found        : {model.result_.n_evaluated and len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)}")
    print(f"    sweep iterations      : {model.result_.n_evaluated} / {model.result_.n_total}"
          f"  ({model.result_.speedup:.1f}x fewer)")
    print(f"    stopped early         : {model.result_.stopped_early} ({model.result_.stop_reason})")
    print(f"    ARI vs ground truth   : {adjusted_rand_score(y_true, model.labels_):.3f}\n")

    # ── 2. Functional API + telemetry ─────────────────────────────────────
    res = smart_sweep(X, hdbscan_cpu, make_mcs_range(2, 200), mode="active")
    print("[2] smart_sweep (functional)")
    print(f"    best_sweep_value={res.best_sweep_value}  best_composite={res.best_composite:.3f}")
    peak = res.best_signals
    print(f"    @peak: k={peak.n_clusters}  noise={peak.noise_ratio:.2f}  "
          f"silhouette={peak.silhouette:.3f}  persistence_sum={peak.persistence_sum:.3f}\n")

    # ── 3. Custom configuration ───────────────────────────────────────────
    cfg = StopperConfig(warmup=10, patience=25)  # more patient → safer, slower
    res2 = smart_sweep(X, hdbscan_cpu, make_mcs_range(2, 200), stopper_cfg=cfg)
    print("[3] custom StopperConfig(patience=25)")
    print(f"    iterations={res2.n_evaluated}  speedup={res2.speedup:.1f}x  "
          f"best_mcs={int(res2.best_sweep_value)}")


if __name__ == "__main__":
    main()
