# Densweep

**Automatic parameter selection for density-based clustering (HDBSCAN / DBSCAN), powered by a smart early-stopper.**

Version 1.0 · Authors: **Rafał Maciejewski**, **Robert Kłopotek**

Densweep sweeps a clustering hyper-parameter — for HDBSCAN, `min_cluster_size` — and uses a *smart-stopper* that watches an internal quality signal and ends the sweep the moment the good region has been found. With the configuration tuned on an NVIDIA DGX&nbsp;Spark, it evaluates **≈6.5× fewer** parameter values than an exhaustive sweep while keeping **≈97.5%** of the oracle (best-possible) clustering quality, measured by Adjusted Rand Index.

Everything — HDBSCAN\*, DBSCAN, silhouette, DBCV and ARI — is implemented **from scratch in NumPy**. CuPy is used automatically for GPU acceleration when present; nothing else is required.

---

## Why

Density clustering quality hinges on one parameter, and the usual way to tune it is an exhaustive sweep with a validity score (silhouette / DBCV) at every step — hundreds of clusterings. Most of that work is wasted: the score rises, reaches a plateau, and the rest of the sweep tells you nothing new. Densweep detects the plateau and stops.

```
exhaustive sweep   ████████████████████████████████████████████████  ~179 clusterings
densweep           ███████                                           ~28 clusterings  (6.5× fewer)
quality retained   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░  97.5% of oracle ARI
```

## Install

```bash
pip install -e .                 # core (NumPy only)
pip install -e ".[gpu]"          # + CuPy GPU backend (pick the wheel for your CUDA)
pip install -e ".[dev]"          # + pytest / scikit-learn / scipy for tests & examples
```

Requires Python ≥ 3.9. The only runtime dependency is NumPy.

## Quickstart

```python
import numpy as np
from densweep import Densweep

X = np.load("embedding.npy")           # e.g. a UMAP embedding, (n_samples, n_features)

model = Densweep().fit(X)              # sweeps min_cluster_size, stops early
labels = model.labels_                 # cluster id per point, -1 = noise

print(model.best_min_cluster_size_)    # the selected parameter
print(model.result_.speedup)           # e.g. 7.3  (× fewer iterations than full sweep)
print(model.result_.stopped_early)     # True
```

Functional API (the form the calibration and benchmark code use):

```python
from densweep import smart_sweep, hdbscan_cpu, make_mcs_range

res = smart_sweep(X, hdbscan_cpu, make_mcs_range(2, 200), mode="active")
res.best_sweep_value     # best min_cluster_size
res.n_evaluated          # how many values were actually tried
res.history              # list of IterationSignals (full telemetry per step)
```

GPU is a one-word change (falls back to CPU automatically if CuPy is missing):

```python
from densweep import hdbscan_gpu
res = smart_sweep(X, hdbscan_gpu, make_mcs_range(2, 200))
```

## How it works

**1 · From-scratch HDBSCAN\*.** `densweep/hdbscan.py` is a pure-NumPy HDBSCAN\*: pairwise distances → core distances → mutual-reachability graph → minimum spanning tree (vectorised Prim) → single-linkage hierarchy → condensed tree (`min_cluster_size`) → cluster stability (excess of mass) → flat-cluster extraction → labels, membership probabilities and per-cluster persistence. The O(n²) numerics run on NumPy or, when available, CuPy. It matches scikit-learn's HDBSCAN to within ~0.5% ARI on standard benchmarks.

**2 · Telemetry per step.** Each clustering yields an `IterationSignals` record: cluster count, noise ratio, persistence (sum & max), mean membership, silhouette and (every few steps, since it is costlier) DBCV.

**3 · The composite.** Those signals are blended into one scalar `composite ∈ [0, 1]`. Each signal is min–max normalised online against the range seen so far, then combined and discounted by the noise ratio:

```
composite = ( w_sil·silhouette + w_dbcv·DBCV + w_pers·persistence + w_ncl·cluster_count ) · (1 − noise_ratio)
```

The cluster-count term matters: a silhouette-only score quietly prefers fewer, coarser clusters, so the composite rewards granularity to counteract that — the weights were chosen from the DGX-Spark telemetry, where the composite correlated most strongly with low noise, DBCV and cluster count.

**4 · The smart-stopper.** `StopChecker` watches the composite. It stops when the score has plateaued — `patience` consecutive steps without a `min_delta` improvement — never before `warmup` steps, with an extra *chaos guard* that bails out early if the cluster count stops agreeing with itself across a recent window (a sign the parameter has wandered into a structureless regime).

### The winning configuration

These are the defaults of `StopperConfig` — the values that won the DGX-Spark search and are exported as `WINNING_CONFIG`:

| parameter | value | meaning |
|---|---|---|
| `warmup` | 10 | no stop is issued before this many iterations |
| `patience` | 15 | stop after this many steps with no `min_delta` gain |
| `min_delta` | 0.01 | smallest composite gain that counts as improvement |
| `chaos_window` | 20 | recent window examined for cluster-count agreement |
| `chaos_agreement_min` | 0.30 | below this agreement (past the peak) ⇒ chaotic ⇒ stop |

## Benchmark

`examples/benchmark.py` reproduces the Phase-3 comparison on UMAP-embedding-like data — the smart-stopper against an exhaustive oracle and a silhouette-only baseline. Typical output:

```
SMART-STOPPER  : mean speedup 7.3x   mean quality retained 95.6%
  mean iters    smart=27.8  oracle=199.0
  mean ARI      smart=0.867  oracle=0.905  silhouette-only=0.859
```

The original DGX-Spark tuning run reported **6.5× speedup at 97.5% quality** on UMAP-embedded real datasets (MNIST, Fashion-MNIST, PenDigits, USPS, Olivetti). The mean of ~28 evaluated iterations reproduces that run almost exactly (it reported 27.6).

## API

| symbol | what it is |
|---|---|
| `Densweep` | scikit-learn-style estimator: `.fit(X)`, `.fit_predict(X)`, `.labels_`, `.best_min_cluster_size_`, `.result_` |
| `smart_sweep(X, clusterer_fn, values, …)` | the sweep driver → `SweepResult` |
| `hdbscan` / `hdbscan_cpu` / `hdbscan_gpu` | from-scratch HDBSCAN\* → `ClusterResult` |
| `dbscan` / `dbscan_cpu` / `dbscan_gpu` | from-scratch DBSCAN |
| `StopperConfig`, `StopChecker`, `StopDecision` | the smart-stopper |
| `CompositeScorer`, `IterationSignals` | telemetry & composite scoring |
| `silhouette_score`, `dbcv_score`, `adjusted_rand_score` | from-scratch metrics |
| `make_mcs_range`, `make_eps_range` | sweep-range builders |
| `WINNING_CONFIG`, `gpu_available` | the tuned config; GPU probe |

## Project layout

```
densweep/
  hdbscan.py     from-scratch HDBSCAN* + DBSCAN (CPU/GPU)
  metrics.py     silhouette, DBCV, ARI, IterationSignals, CompositeScorer
  stopper.py     StopperConfig (winning defaults), StopChecker
  sweep.py       smart_sweep, Densweep estimator, sweep ranges
  _backend.py    NumPy / CuPy backend shim
tests/           pytest suite (incl. cross-checks vs scikit-learn)
examples/        quickstart.py, benchmark.py
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Authors
- Rafał Maciejewski (www.github.com/SpecuAgentRM)
- Robert Kłopotek (www.github.com/rakubaku)

## License

Densweep License 1.0 — free of charge for both **commercial and non-commercial** use and redistribution, **no attribution required**. The software may be used **in unmodified form only**; modification and derivative works are not permitted. See [LICENSE](LICENSE) for the full terms.
