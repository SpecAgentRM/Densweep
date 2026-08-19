"""
densweep.metrics
────────────────
From-scratch, NumPy-only quality metrics and the per-iteration signal record
that the smart-stopper reasons over.

  * silhouette_score        – cohesion vs separation (sampled for big n)
  * dbcv_score              – density-based cluster validity (Moulavi et al.)
  * adjusted_rand_score     – external agreement, used by benchmarks/oracle
  * IterationSignals        – one row of sweep telemetry
  * CompositeScorer         – online blend of internal signals → composite

The *composite* is the single scalar the stopper tracks: an online min–max
normalised blend of the unsupervised quality signals (persistence, membership,
silhouette, DBCV) discounted by the noise ratio. It rises as the sweep finds
real structure and plateaus once the good region is reached — which is exactly
the trajectory the early-stopper exploits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
#  Internal validity metrics
# ──────────────────────────────────────────────────────────────────────────
def silhouette_score(X, labels, sample_size: int = 2000, random_state: int = 0) -> float:
    """Mean silhouette over non-noise points (``-1`` excluded).

    For ``n > sample_size`` a random subsample is scored for speed, matching how
    the original sweep kept per-iteration cost bounded.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    mask = labels != -1
    Xc, lc = X[mask], labels[mask]
    uniq = np.unique(lc)
    if len(uniq) < 2 or len(lc) < 2:
        return 0.0

    if len(lc) > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(lc), size=sample_size, replace=False)
        Xc, lc = Xc[idx], lc[idx]
        if len(np.unique(lc)) < 2:
            return 0.0

    n = len(Xc)
    D = _euclidean(Xc)
    sil = np.zeros(n)
    for c in np.unique(lc):
        in_c = lc == c
        size_c = in_c.sum()
        if size_c <= 1:
            sil[in_c] = 0.0
            continue
        # a(i): mean intra-cluster distance
        a = D[np.ix_(in_c, in_c)].sum(axis=1) / (size_c - 1)
        # b(i): smallest mean distance to another cluster
        b = np.full(size_c, np.inf)
        for o in np.unique(lc):
            if o == c:
                continue
            in_o = lc == o
            mean_o = D[np.ix_(in_c, in_o)].mean(axis=1)
            b = np.minimum(b, mean_o)
        denom = np.maximum(a, b)
        s = np.where(denom > 0, (b - a) / denom, 0.0)
        sil[in_c] = s
    return float(sil.mean())


def dbcv_score(X, labels) -> float:
    """Density-Based Clustering Validation (Moulavi et al., 2014) in ``[-1, 1]``.

    Combines within-cluster *density sparseness* (the longest edge of each
    cluster's internal mutual-reachability MST) with between-cluster *density
    separation*, weighted by cluster size. Noise points are ignored.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    d = X.shape[1]
    mask = labels != -1
    Xc, lc = X[mask], labels[mask]
    clusters = [c for c in np.unique(lc) if (lc == c).sum() >= 2]
    if len(clusters) < 1 or len(Xc) < 2:
        return 0.0

    D = _euclidean(Xc)
    np.fill_diagonal(D, np.inf)
    eps = 1e-12

    # all-points core distance within each point's own cluster
    apts = np.zeros(len(Xc))
    internal = {}  # cluster -> (member indices, internal MST max edge, mr matrix)
    for c in clusters:
        members = np.where(lc == c)[0]
        sub = D[np.ix_(members, members)]
        m = len(members)
        with np.errstate(divide="ignore"):
            inv = (1.0 / np.maximum(sub, eps)) ** d
        coredist = (inv.sum(axis=1) / (m - 1)) ** (-1.0 / d)
        apts[members] = coredist
        # mutual reachability inside the cluster
        mr = np.maximum(np.maximum(sub, coredist[:, None]), coredist[None, :])
        dsc = _mst_max_edge(mr)  # density sparseness of the cluster
        internal[c] = (members, dsc, coredist)

    total = len(Xc)
    score = 0.0
    for c in clusters:
        members, dsc, coredist_c = internal[c]
        # density separation to the nearest other cluster
        dspc = np.inf
        for o in clusters:
            if o == c:
                continue
            om, _, coredist_o = internal[o]
            cross = D[np.ix_(members, om)]
            mr_cross = np.maximum(
                np.maximum(cross, coredist_c[:, None]), coredist_o[None, :]
            )
            dspc = min(dspc, mr_cross.min())
        denom = max(dspc, dsc, eps)
        v_c = (dspc - dsc) / denom if np.isfinite(dspc) else 0.0
        score += (len(members) / total) * v_c
    return float(score)


def adjusted_rand_score(labels_true, labels_pred) -> float:
    """Adjusted Rand Index — external agreement of two labellings."""
    a = np.asarray(labels_true)
    b = np.asarray(labels_pred)
    n = a.shape[0]
    if n == 0:
        return 1.0
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    cont = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(cont, (a, b), 1)
    sum_comb = lambda x: (x * (x - 1) // 2).sum()
    sum_c = sum_comb(cont)
    sum_a = sum_comb(cont.sum(axis=1))
    sum_b = sum_comb(cont.sum(axis=0))
    total = n * (n - 1) // 2
    expected = sum_a * sum_b / total if total else 0.0
    maxi = 0.5 * (sum_a + sum_b)
    denom = maxi - expected
    if denom == 0:
        return 1.0
    return float((sum_c - expected) / denom)


# ──────────────────────────────────────────────────────────────────────────
#  Small numeric helpers
# ──────────────────────────────────────────────────────────────────────────
def _euclidean(X):
    sq = np.sum(X * X, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    return np.sqrt(d2)


def _mst_max_edge(W):
    """Largest edge of the MST of a dense symmetric weight matrix (Prim)."""
    n = W.shape[0]
    if n <= 1:
        return 0.0
    in_tree = np.zeros(n, dtype=bool)
    best = W[0].copy()
    best[0] = np.inf
    in_tree[0] = True
    max_edge = 0.0
    for _ in range(n - 1):
        j = int(np.argmin(best))
        w = best[j]
        if np.isfinite(w):
            max_edge = max(max_edge, float(w))
        in_tree[j] = True
        best[j] = np.inf
        cand = W[j]
        improve = (cand < best) & (~in_tree)
        best = np.where(improve, cand, best)
    return max_edge


# ──────────────────────────────────────────────────────────────────────────
#  Per-iteration telemetry
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class IterationSignals:
    """One row of sweep telemetry — the unit the stopper consumes."""

    sweep_value: float
    n_clusters: int
    noise_ratio: float
    persistence_sum: float
    persistence_max: float
    mean_membership: float
    silhouette: float
    dbcv: Optional[float] = None
    avg_size: float = 0.0
    iter_time: float = 0.0
    composite: float = 0.0


def signals_from_result(result, sweep_value, X=None, compute_silhouette=True,
                        compute_dbcv=False, iter_time=0.0) -> IterationSignals:
    """Build an :class:`IterationSignals` from a clustering ``ClusterResult``."""
    labels = np.asarray(result.labels)
    n = labels.shape[0]
    noise_ratio = float((labels == -1).mean()) if n else 1.0
    mask = labels != -1
    uniq = np.unique(labels[mask]) if mask.any() else np.array([])
    n_clusters = int(len(uniq))
    avg_size = float(mask.sum() / n_clusters) if n_clusters else 0.0

    persistence = result.cluster_persistence
    if persistence is not None and len(persistence):
        persistence_sum = float(np.sum(persistence))
        persistence_max = float(np.max(persistence))
    else:
        persistence_sum = persistence_max = 0.0

    probs = np.asarray(result.probabilities)
    mean_membership = float(probs[mask].mean()) if mask.any() else 0.0

    sil = 0.0
    if compute_silhouette and n_clusters >= 2 and X is not None:
        sil = silhouette_score(X, labels)
    dbcv = None
    if compute_dbcv and n_clusters >= 1 and X is not None:
        dbcv = dbcv_score(X, labels)

    return IterationSignals(
        sweep_value=float(sweep_value), n_clusters=n_clusters,
        noise_ratio=noise_ratio, persistence_sum=persistence_sum,
        persistence_max=persistence_max, mean_membership=mean_membership,
        silhouette=sil, dbcv=dbcv, avg_size=avg_size, iter_time=iter_time,
    )


# ──────────────────────────────────────────────────────────────────────────
#  Composite scorer
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class CompositeScorer:
    """Online blend of internal signals into a single ``composite`` score.

    Each contributing signal is min–max normalised against the running range
    observed so far in the sweep, so the composite is comparable across very
    different datasets, then blended and discounted by the noise ratio::

        composite = (w_sil·ŝil + w_dbcv·d̂bcv + w_pers·p̂ers + w_ncl·n̂cl)·(1−noise)

    The weights were chosen from the DGX-Spark telemetry, where the composite
    correlated most strongly with low noise, DBCV and *cluster count* — the last
    term is what stops the score from rewarding ever-coarser clusterings (a
    silhouette-only blend collapses everything into a few big clusters). The
    score is zeroed while ``n_clusters < 2`` or during the short normalisation
    warmup.
    """

    w_silhouette: float = 0.20
    w_dbcv: float = 0.35
    w_persistence: float = 0.10
    w_nclusters: float = 0.35
    warmup: int = 5  # iterations needed before the normalisation is trusted

    _mins: dict = field(default_factory=dict, repr=False)
    _maxs: dict = field(default_factory=dict, repr=False)
    _last_dbcv: Optional[float] = field(default=None, repr=False)
    _i: int = field(default=0, repr=False)

    def _norm(self, key, value):
        lo = self._mins.get(key)
        hi = self._maxs.get(key)
        self._mins[key] = value if lo is None else min(lo, value)
        self._maxs[key] = value if hi is None else max(hi, value)
        lo, hi = self._mins[key], self._maxs[key]
        return 0.0 if hi <= lo else (value - lo) / (hi - lo)

    def update(self, sig: IterationSignals) -> float:
        """Register ``sig``, set and return its ``composite`` (also stored on it)."""
        i = self._i
        self._i += 1

        if sig.dbcv is not None:
            self._last_dbcv = sig.dbcv
        sil01 = 0.5 * (np.clip(sig.silhouette, -1, 1) + 1.0)
        dbcv01 = None if self._last_dbcv is None else 0.5 * (np.clip(self._last_dbcv, -1, 1) + 1.0)

        n_sil = self._norm("sil", sil01)
        n_pers = self._norm("pers", sig.persistence_max)
        n_ncl = self._norm("ncl", float(sig.n_clusters))
        n_dbcv = self._norm("dbcv", dbcv01) if dbcv01 is not None else None

        terms = [(self.w_silhouette, n_sil),
                 (self.w_persistence, n_pers),
                 (self.w_nclusters, n_ncl)]
        if n_dbcv is not None:
            terms.append((self.w_dbcv, n_dbcv))
        wsum = sum(w for w, _ in terms)
        quality = sum(w * v for w, v in terms) / wsum if wsum else 0.0

        composite = quality * (1.0 - sig.noise_ratio)
        if sig.n_clusters < 2 or i < self.warmup:
            composite = 0.0

        sig.composite = float(composite)
        return sig.composite

    def batch_score(self, history) -> np.ndarray:
        """Re-score the whole ``history`` with full-range normalisation.

        The per-step :meth:`update` composite must be *causal* (it drives the
        stopper, which may only look backwards), so a signal that peaks late is
        under-credited online. For the final *selection* of the best parameter
        we have all the data, so we normalise each signal against its full
        observed range — removing the look-ahead bias of the online version.
        Returns one composite per history row; the stopper still uses the
        causal ``sig.composite``.
        """
        n = len(history)
        if n == 0:
            return np.array([])
        sil01 = np.array([0.5 * (np.clip(s.silhouette, -1, 1) + 1) for s in history])
        pers = np.array([s.persistence_max for s in history], dtype=float)
        ncl = np.array([float(s.n_clusters) for s in history])
        noise = np.array([s.noise_ratio for s in history], dtype=float)
        ncl_int = np.array([s.n_clusters for s in history])

        db = np.full(n, np.nan)
        last = None
        for i, s in enumerate(history):
            if s.dbcv is not None:
                last = s.dbcv
            if last is not None:
                db[i] = 0.5 * (np.clip(last, -1, 1) + 1)
        has_db = np.isfinite(db).any()

        def mm(a):
            finite = a[np.isfinite(a)]
            if finite.size == 0:
                return np.zeros_like(a)
            lo, hi = finite.min(), finite.max()
            return np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)

        n_sil, n_pers, n_ncl = mm(sil01), mm(pers), mm(ncl)
        n_db = mm(db) if has_db else None

        out = np.zeros(n)
        for i in range(n):
            terms = [(self.w_silhouette, n_sil[i]),
                     (self.w_persistence, n_pers[i]),
                     (self.w_nclusters, n_ncl[i])]
            if n_db is not None and np.isfinite(db[i]):
                terms.append((self.w_dbcv, n_db[i]))
            wsum = sum(w for w, _ in terms)
            q = sum(w * v for w, v in terms) / wsum if wsum else 0.0
            c = q * (1.0 - noise[i])
            if ncl_int[i] < 2:
                c = 0.0
            out[i] = c
        return out
