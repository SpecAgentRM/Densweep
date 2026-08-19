"""
Densweep
════════
Automatic parameter selection for density-based clustering (HDBSCAN / DBSCAN).

Densweep sweeps a clustering parameter (e.g. HDBSCAN's ``min_cluster_size``) and
uses a *smart-stopper* — early stopping on an internal composite quality signal
— to quit the sweep as soon as the good region is found. With the tuned
configuration shipped here it evaluates ≈6.5× fewer parameter values than an
exhaustive sweep while retaining ≈97.5% of the oracle clustering quality (ARI).

Everything (HDBSCAN, DBSCAN, silhouette, DBCV, ARI) is implemented from scratch
in NumPy; CuPy is used automatically for GPU acceleration when available.

Quick start
-----------
>>> from densweep import Densweep
>>> model = Densweep().fit(X)
>>> labels = model.labels_
>>> print(model.best_min_cluster_size_, model.result_.speedup)

Functional API
--------------
>>> from densweep import smart_sweep, hdbscan_cpu, make_mcs_range
>>> res = smart_sweep(X, hdbscan_cpu, make_mcs_range(2, 200))
>>> res.best_sweep_value, res.n_evaluated, res.stopped_early
"""
from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Rafał Maciejewski, Robert Kłopotek"

from .hdbscan import (
    ClusterResult,
    hdbscan,
    hdbscan_cpu,
    hdbscan_gpu,
    dbscan,
    dbscan_cpu,
    dbscan_gpu,
)
from .metrics import (
    IterationSignals,
    CompositeScorer,
    silhouette_score,
    dbcv_score,
    adjusted_rand_score,
    signals_from_result,
)
from .stopper import (
    StopperConfig,
    StopChecker,
    StopDecision,
    WINNING_CONFIG,
)
from .sweep import (
    Densweep,
    SweepResult,
    smart_sweep,
    make_mcs_range,
    make_eps_range,
    pick_clusterer,
)
from ._backend import gpu_available, to_numpy

__all__ = [
    "__version__",
    # estimator + driver
    "Densweep",
    "smart_sweep",
    "SweepResult",
    "make_mcs_range",
    "make_eps_range",
    "pick_clusterer",
    # clusterers
    "ClusterResult",
    "hdbscan",
    "hdbscan_cpu",
    "hdbscan_gpu",
    "dbscan",
    "dbscan_cpu",
    "dbscan_gpu",
    # stopper
    "StopperConfig",
    "StopChecker",
    "StopDecision",
    "WINNING_CONFIG",
    # metrics
    "IterationSignals",
    "CompositeScorer",
    "silhouette_score",
    "dbcv_score",
    "adjusted_rand_score",
    "signals_from_result",
    # backend
    "gpu_available",
    "to_numpy",
]
