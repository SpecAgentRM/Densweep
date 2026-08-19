"""
densweep.sweep
──────────────
The driver that ties everything together: walk a list of candidate parameter
values, cluster at each, score the result, and let the smart-stopper decide when
to quit early. Also the high-level :class:`Densweep` estimator and the sweep-range
builders.

    from densweep import Densweep
    model = Densweep().fit(X)        # auto-selects min_cluster_size, stops early
    labels = model.labels_

or the functional form used by the calibration/benchmark code:

    result = smart_sweep(X, hdbscan_cpu, make_mcs_range(2, 200))
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from .hdbscan import (
    ClusterResult, hdbscan_cpu, hdbscan_gpu, dbscan_cpu, dbscan_gpu,
)
from .metrics import IterationSignals, CompositeScorer, signals_from_result
from .stopper import StopperConfig, StopChecker


# ──────────────────────────────────────────────────────────────────────────
#  Sweep-range builders
# ──────────────────────────────────────────────────────────────────────────
def make_mcs_range(start: int = 2, end: int = 200, step: int = 1) -> List[int]:
    """Candidate ``min_cluster_size`` values for an HDBSCAN sweep."""
    start = max(2, int(start))
    return list(range(start, int(end) + 1, int(step)))


def make_eps_range(X, n_steps: int = 200, factor_min: float = 0.01,
                   factor_max: float = 4.0) -> List[float]:
    """Candidate ``eps`` values for a DBSCAN sweep, scaled to the data.

    The scale is the median nearest-neighbour distance; eps spans
    ``[factor_min, factor_max] * scale`` on a geometric grid.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    m = min(n, 1000)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=m, replace=False) if n > m else np.arange(n)
    sq = np.sum(X[idx] ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X[idx] @ X[idx].T, 0.0)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(d2.min(axis=1))
    scale = float(np.median(nn)) or 1.0
    return list(np.geomspace(factor_min * scale, factor_max * scale, int(n_steps)))


# ──────────────────────────────────────────────────────────────────────────
#  Clusterer selection
# ──────────────────────────────────────────────────────────────────────────
def pick_clusterer(algorithm: str = "hdbscan", device: str = "cpu",
                   min_samples: int = 5) -> Callable:
    """Return a ``clusterer_fn(X, value) -> ClusterResult`` for the sweep."""
    if algorithm == "hdbscan":
        base = hdbscan_gpu if device == "gpu" else hdbscan_cpu
        return lambda X, v: base(X, int(v))
    if algorithm == "dbscan":
        base = dbscan_gpu if device == "gpu" else dbscan_cpu
        return lambda X, v: base(X, float(v), min_samples)
    raise ValueError(f"unknown algorithm: {algorithm!r}")


# ──────────────────────────────────────────────────────────────────────────
#  Sweep result
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class SweepResult:
    history: List[IterationSignals]
    best_idx: int
    best_sweep_value: Optional[float]
    best_composite: float
    stopped_early: bool
    stop_reason: str
    n_evaluated: int
    n_total: int

    @property
    def speedup(self) -> float:
        """Iterations saved vs. the full sweep (the headline metric)."""
        return (self.n_total / self.n_evaluated) if self.n_evaluated else 1.0

    @property
    def best_signals(self) -> Optional[IterationSignals]:
        return self.history[self.best_idx] if self.history else None


# ──────────────────────────────────────────────────────────────────────────
#  The sweep driver
# ──────────────────────────────────────────────────────────────────────────
def smart_sweep(
    X,
    clusterer_fn: Callable,
    sweep_values: Sequence,
    *,
    algorithm: str = "hdbscan",
    mode: str = "active",
    compute_silhouette: bool = True,
    compute_dbcv: bool = True,
    dbcv_every: int = 10,
    stopper_cfg: Optional[StopperConfig] = None,
    scorer: Optional[CompositeScorer] = None,
    verbose: bool = False,
) -> SweepResult:
    """Run a (smart-stopped) parameter sweep.

    Parameters
    ----------
    X:
        Data matrix.
    clusterer_fn:
        Callable ``(X, value) -> ClusterResult`` (see :func:`pick_clusterer`).
    sweep_values:
        Ordered candidate parameter values to try.
    mode:
        ``"active"`` consults the stopper and may stop early; ``"passive"`` runs
        every value (used for offline calibration / oracle baselines).
    compute_dbcv / dbcv_every:
        DBCV is comparatively expensive, so it is computed every ``dbcv_every``
        iterations and forward-filled into the composite.
    stopper_cfg:
        Defaults to the winning :class:`StopperConfig`.
    """
    cfg = stopper_cfg or StopperConfig()
    checker = StopChecker(cfg)
    scorer = scorer or CompositeScorer(warmup=min(5, cfg.warmup))
    history: List[IterationSignals] = []
    stopped_early = False
    stop_reason = "full_sweep"
    n_total = len(list(sweep_values))

    for i, val in enumerate(sweep_values):
        t0 = time.time()
        result = clusterer_fn(X, val)
        dt = time.time() - t0

        do_dbcv = compute_dbcv and (i % max(1, dbcv_every) == 0)
        sig = signals_from_result(
            result, val, X=X,
            compute_silhouette=compute_silhouette,
            compute_dbcv=do_dbcv, iter_time=dt,
        )
        scorer.update(sig)
        history.append(sig)

        if verbose:
            print(f"  [{i:3d}] sv={val} k={sig.n_clusters} "
                  f"noise={sig.noise_ratio:.2f} comp={sig.composite:.4f}")

        if mode == "active":
            decision = checker.check(history)
            if decision.should_stop:
                stopped_early = True
                stop_reason = decision.reason
                break

    # Select the best parameter with unbiased full-range normalisation (the
    # causal per-step composite stored in `history` is kept for the stopper).
    final_comps = scorer.batch_score(history)
    best_idx = int(np.argmax(final_comps)) if len(final_comps) else 0
    best = history[best_idx] if history else None
    return SweepResult(
        history=history,
        best_idx=best_idx,
        best_sweep_value=(best.sweep_value if best else None),
        best_composite=(float(final_comps[best_idx]) if len(final_comps) else 0.0),
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        n_evaluated=len(history),
        n_total=n_total,
    )


# ──────────────────────────────────────────────────────────────────────────
#  High-level estimator
# ──────────────────────────────────────────────────────────────────────────
class Densweep:
    """Auto-tuned density clustering: sweep + smart-stop in one object.

    Example
    -------
    >>> model = Densweep(algorithm="hdbscan").fit(X)
    >>> model.labels_, model.best_min_cluster_size_, model.result_.speedup
    """

    def __init__(
        self,
        algorithm: str = "hdbscan",
        device: str = "cpu",
        min_samples: int = 5,
        sweep_values: Optional[Sequence] = None,
        mcs_range=(2, 200, 1),
        eps_steps: int = 200,
        stopper_cfg: Optional[StopperConfig] = None,
        compute_dbcv: bool = True,
        dbcv_every: int = 10,
        mode: str = "active",
    ):
        self.algorithm = algorithm
        self.device = device
        self.min_samples = min_samples
        self.sweep_values = sweep_values
        self.mcs_range = mcs_range
        self.eps_steps = eps_steps
        self.stopper_cfg = stopper_cfg or StopperConfig()
        self.compute_dbcv = compute_dbcv
        self.dbcv_every = dbcv_every
        self.mode = mode

        # populated by fit()
        self.result_: Optional[SweepResult] = None
        self.labels_: Optional[np.ndarray] = None
        self.probabilities_: Optional[np.ndarray] = None
        self.best_value_: Optional[float] = None
        self.best_min_cluster_size_: Optional[int] = None
        self.best_eps_: Optional[float] = None

    def _build_values(self, X):
        if self.sweep_values is not None:
            return list(self.sweep_values)
        if self.algorithm == "hdbscan":
            start, end, step = self.mcs_range
            end = min(end, max(2, X.shape[0] // 4))
            start = min(max(2, start), end)  # keep the range non-empty for small n
            return make_mcs_range(start, end, step)
        return make_eps_range(X, n_steps=self.eps_steps)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        clusterer = pick_clusterer(self.algorithm, self.device, self.min_samples)
        values = self._build_values(X)
        self.result_ = smart_sweep(
            X, clusterer, values,
            algorithm=self.algorithm, mode=self.mode,
            compute_dbcv=self.compute_dbcv, dbcv_every=self.dbcv_every,
            stopper_cfg=self.stopper_cfg,
        )
        self.best_value_ = self.result_.best_sweep_value
        if self.algorithm == "hdbscan":
            self.best_min_cluster_size_ = int(self.best_value_) if self.best_value_ else None
        else:
            self.best_eps_ = float(self.best_value_) if self.best_value_ else None

        # Realise the labelling at the chosen parameter. With no usable value
        # (e.g. an empty sweep) everything is noise.
        if self.best_value_ is None:
            self.labels_ = -np.ones(X.shape[0], dtype=np.int64)
            self.probabilities_ = np.zeros(X.shape[0])
        else:
            final = clusterer(X, self.best_value_)
            self.labels_ = final.labels
            self.probabilities_ = final.probabilities
        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X).labels_
