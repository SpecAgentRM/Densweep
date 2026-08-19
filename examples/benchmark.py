"""
Densweep — benchmark  /  benchmark
══════════════════════════════════
Reproduces the Phase-3 style comparison: the smart-stopper vs an exhaustive
oracle full-sweep and several baselines, on UMAP-embedding-like data.

Run:  python examples/benchmark.py

EN: reports mean ARI, mean sweep iterations, speedup and quality retention.
PL: raportuje średnie ARI, średnią liczbę iteracji, przyspieszenie i zachowaną jakość.

Each (dataset, min_cluster_size) is clustered once and cached, so every method
is scored from the same clusterings — the comparison is purely about *which*
parameter each strategy selects and *how many* it has to try.
"""
import os
import sys
import time

import numpy as np

# Allow running straight from a checkout (no install needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from densweep import (
    smart_sweep, hdbscan_cpu, make_mcs_range, StopperConfig, WINNING_CONFIG,
    adjusted_rand_score, signals_from_result,
)
from densweep.metrics import CompositeScorer


def umap_like(seed, k, n, std):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-13, 13, size=(k, 2))
    sizes = rng.multinomial(n, np.ones(k) / k)
    X, y = [], []
    for ci, (c, s) in enumerate(zip(centers, sizes)):
        X.append(rng.normal(c, rng.uniform(std * 0.7, std), size=(max(s, 1), 2)))
        y += [ci] * max(s, 1)
    return np.vstack(X).astype(np.float64), np.array(y)


DATASETS = [
    # (name, n_clusters, n_samples, cluster_std, seed)
    ("set_a", 6, 900, 0.50, 101),
    ("set_b", 8, 1000, 0.45, 202),
    ("set_c", 10, 1100, 0.50, 303),
    ("set_d", 5, 800, 0.60, 404),
    ("set_e", 7, 950, 0.55, 505),
]


def main():
    cfg = StopperConfig()
    print("Winning stopper config:", WINNING_CONFIG)
    print("=" * 78)
    rows = []
    for name, k, n, std, seed in DATASETS:
        X, y = umap_like(seed, k, n, std)
        vals = make_mcs_range(2, min(200, n // 4))

        # cache one clustering per candidate
        t0 = time.time()
        labels = {v: hdbscan_cpu(X, v).labels for v in vals}
        cache_s = time.time() - t0
        aris = np.array([adjusted_rand_score(y, labels[v]) for v in vals])

        # oracle (knows the truth): best ARI over the full sweep
        oi = int(np.argmax(aris))
        oracle_ari, oracle_iter = aris[oi], len(vals)

        # smart-stopper (unsupervised, early stop)
        res = smart_sweep(X, hdbscan_cpu, vals, mode="active", stopper_cfg=cfg)
        smart_ari = aris[vals.index(int(res.best_sweep_value))]

        # baselines, all unsupervised, all run the FULL sweep (no early stop)
        sil = np.array([_sig(labels[v], X, v).silhouette for v in vals])
        sil_ari = aris[int(np.argmax(sil))]
        rng = np.random.default_rng(0)
        rand_idx = rng.choice(len(vals), size=min(35, len(vals)), replace=False)
        rand_ari = aris[rand_idx[np.argmax([aris[i] for i in rand_idx])]]  # best of a random subset

        rows.append(dict(
            name=name, k=k, n=n, oracle_ari=oracle_ari, oracle_iter=oracle_iter,
            smart_ari=smart_ari, smart_iter=res.n_evaluated,
            sil_ari=sil_ari, rand_ari=rand_ari,
            speedup=oracle_iter / res.n_evaluated,
            quality=smart_ari / oracle_ari if oracle_ari > 0 else 1.0,
            reason=res.stop_reason, cache_s=cache_s,
        ))
        print(f"{name}: oracle ARI={oracle_ari:.3f}@{vals[oi]} ({oracle_iter} it) | "
              f"smart ARI={smart_ari:.3f}@{int(res.best_sweep_value)} ({res.n_evaluated} it, "
              f"{rows[-1]['speedup']:.1f}x, {rows[-1]['quality']*100:.0f}%, {res.stop_reason}) | "
              f"silhouette-only ARI={sil_ari:.3f}")

    print("=" * 78)
    sp = np.mean([r["speedup"] for r in rows])
    ql = np.mean([r["quality"] for r in rows])
    print(f"SMART-STOPPER  : mean speedup {sp:.1f}x   mean quality retained {ql*100:.1f}%")
    print(f"  mean iters    smart={np.mean([r['smart_iter'] for r in rows]):.1f}  "
          f"oracle={np.mean([r['oracle_iter'] for r in rows]):.1f}")
    print(f"  mean ARI      smart={np.mean([r['smart_ari'] for r in rows]):.3f}  "
          f"oracle={np.mean([r['oracle_ari'] for r in rows]):.3f}  "
          f"silhouette-only={np.mean([r['sil_ari'] for r in rows]):.3f}  "
          f"random={np.mean([r['rand_ari'] for r in rows]):.3f}")
    print("\nNote: the DGX-Spark tuning run reported 6.5x speedup at 97.5% quality\n"
          "on UMAP-embedded real datasets (MNIST/Fashion/PenDigits/USPS/Olivetti).")


def _sig(labels, X, v):
    class _R:  # tiny shim so we can reuse signals_from_result
        pass
    r = _R(); r.labels = labels
    r.probabilities = (labels >= 0).astype(float)
    r.cluster_persistence = None
    return signals_from_result(r, v, X=X, compute_silhouette=True, compute_dbcv=False)


if __name__ == "__main__":
    main()
