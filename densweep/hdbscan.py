"""
densweep.hdbscan
────────────────
A from-scratch implementation of HDBSCAN* (Campello, Moulavi & Sander, 2013)
in pure NumPy, plus a small DBSCAN, written so the heavy O(n^2) numerics can
run on either the CPU (NumPy) or the GPU (CuPy) through ``densweep._backend``.

Pipeline (HDBSCAN*):
    1. pairwise distances            ── _pairwise_distance
    2. core distances (k = min_samples)
    3. mutual reachability graph
    4. minimum spanning tree         ── Prim, vectorised
    5. single-linkage hierarchy      ── union-find
    6. condensed tree (min_cluster_size)
    7. cluster stability             ── excess of mass
    8. flat cluster extraction (EOM)
    9. labels, membership probabilities, cluster persistence

Nothing here depends on scikit-learn or the reference ``hdbscan`` package; those
are used only in the test-suite to cross-check correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._backend import get_xp, to_numpy


# ──────────────────────────────────────────────────────────────────────────
#  Result container
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ClusterResult:
    """Outcome of a single clustering call.

    Attributes
    ----------
    labels:
        Integer cluster id per point, ``-1`` for noise. ``numpy.ndarray``.
    probabilities:
        Membership strength in ``[0, 1]`` per point (0 for noise).
    cluster_persistence:
        Persistence (stability) score per surviving cluster, or ``None`` for
        algorithms without a hierarchy (DBSCAN).
    n_clusters:
        Number of non-noise clusters.
    """

    labels: np.ndarray
    probabilities: np.ndarray
    cluster_persistence: Optional[np.ndarray] = None
    n_clusters: int = 0

    def __post_init__(self):
        self.labels = np.asarray(self.labels)
        self.probabilities = np.asarray(self.probabilities)
        if self.n_clusters == 0:
            uniq = set(int(x) for x in self.labels.tolist())
            uniq.discard(-1)
            self.n_clusters = len(uniq)


# ──────────────────────────────────────────────────────────────────────────
#  Low-level numeric primitives (CPU/GPU via xp)
# ──────────────────────────────────────────────────────────────────────────
def _pairwise_distance(X, xp):
    """Dense Euclidean distance matrix, computed on the active backend."""
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b ; clip tiny negatives from round-off.
    sq = xp.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    d2 = xp.maximum(d2, 0.0)
    D = xp.sqrt(d2)
    # exact zero on the diagonal
    n = X.shape[0]
    D[xp.arange(n), xp.arange(n)] = 0.0
    return D


def _core_distances(D, min_samples, xp):
    """Core distance = distance to the ``min_samples``-th nearest neighbour.

    Neighbours are counted *including* the point itself, matching the
    scikit-learn / reference-``hdbscan`` convention where ``min_samples`` of 1
    makes every point its own core.
    """
    n = D.shape[0]
    k = min(max(int(min_samples), 1), n)
    # k-th smallest per row (0-indexed k-1). partition is O(n) per row.
    part = xp.partition(D, k - 1, axis=1)
    return part[:, k - 1]


def _mutual_reachability(D, core, xp):
    """MR(a,b) = max(core(a), core(b), d(a,b))."""
    mr = xp.maximum(D, core[:, None])
    mr = xp.maximum(mr, core[None, :])
    return mr


def _prim_mst(mr, xp):
    """Minimum spanning tree of a dense graph via Prim's algorithm.

    Returns an ``(n-1, 3)`` host array of ``[a, b, weight]`` edges. The inner
    relaxation is fully vectorised, so the whole MST is O(n^2) on the backend.
    """
    n = mr.shape[0]
    in_tree = xp.zeros(n, dtype=bool)
    best = mr[0].copy()
    parent = xp.zeros(n, dtype=xp.int64)
    in_tree[0] = True
    best[0] = xp.inf

    edges = xp.empty((n - 1, 3), dtype=xp.float64)
    for i in range(n - 1):
        j = int(xp.argmin(best))
        edges[i, 0] = parent[j]
        edges[i, 1] = j
        edges[i, 2] = best[j]
        in_tree[j] = True
        best[j] = xp.inf
        cand = mr[j]
        # Only relax nodes not yet in the tree, otherwise settled nodes would
        # re-enter the frontier and create self-loop edges.
        improve = (cand < best) & (~in_tree)
        best = xp.where(improve, cand, best)
        parent = xp.where(improve, j, parent)
    return to_numpy(edges)


# ──────────────────────────────────────────────────────────────────────────
#  Hierarchy → condensed tree → stability → flat clusters  (host / NumPy)
# ──────────────────────────────────────────────────────────────────────────
def _single_linkage(edges, n):
    """Build a SciPy-style linkage matrix from sorted MST edges.

    Returns ``L`` with shape ``(n-1, 4)``: ``[node_a, node_b, dist, size]``,
    where leaves are ``0..n-1`` and internal node ``i`` lives in row ``i-n``.
    """
    order = np.argsort(edges[:, 2], kind="stable")
    edges = edges[order]

    parent = np.full(2 * n - 1, -1, dtype=np.int64)
    size = np.ones(2 * n - 1, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != -1:
            root = parent[root]
        while parent[x] != -1:
            parent[x], x = root, parent[x]
        return root

    L = np.zeros((n - 1, 4), dtype=np.float64)
    next_label = n
    for i in range(n - 1):
        a = int(edges[i, 0])
        b = int(edges[i, 1])
        w = edges[i, 2]
        ra, rb = find(a), find(b)
        L[i, 0] = ra
        L[i, 1] = rb
        L[i, 2] = w
        L[i, 3] = size[ra] + size[rb]
        parent[ra] = next_label
        parent[rb] = next_label
        size[next_label] = size[ra] + size[rb]
        next_label += 1
    return L


def _subtree_nodes(L, root, n):
    """All node ids in the subtree rooted at ``root`` (iterative DFS)."""
    out = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        if node >= n:
            row = L[node - n]
            stack.append(int(row[0]))
            stack.append(int(row[1]))
    return out


def _condense_tree(L, min_cluster_size, n):
    """Condense the single-linkage hierarchy.

    Returns a list of edges ``(parent_cluster, child, lambda, child_size)``
    where ``child`` is either a cluster id (``>= n``) or a point (``< n``), and
    ``lambda = 1/distance``.
    """
    root = 2 * n - 2
    relabel = {root: n}
    next_label = n + 1
    ignore = set()
    result = []

    for node in _subtree_nodes(L, root, n):
        if node in ignore or node < n:
            continue
        row = L[node - n]
        left, right, dist = int(row[0]), int(row[1]), row[2]
        lam = (1.0 / dist) if dist > 0 else np.inf
        lc = int(L[left - n, 3]) if left >= n else 1
        rc = int(L[right - n, 3]) if right >= n else 1

        big_left = lc >= min_cluster_size
        big_right = rc >= min_cluster_size

        if big_left and big_right:
            # genuine split: two new sub-clusters
            relabel[left] = next_label
            next_label += 1
            result.append((relabel[node], relabel[left], lam, lc))
            relabel[right] = next_label
            next_label += 1
            result.append((relabel[node], relabel[right], lam, rc))
        elif not big_left and not big_right:
            # whole node dissolves: every point falls out here
            for sub in _subtree_nodes(L, node, n):
                if sub < n:
                    result.append((relabel[node], sub, lam, 1))
                ignore.add(sub)
        else:
            # one side persists with the parent's label, the other falls out
            falling = left if not big_left else right
            staying = right if not big_left else left
            relabel[staying] = relabel[node]
            for sub in _subtree_nodes(L, falling, n):
                if sub < n:
                    result.append((relabel[node], sub, lam, 1))
                ignore.add(sub)
    return result


def _finite_max_lambda(condensed):
    vals = [l for (_p, _c, l, _s) in condensed if np.isfinite(l)]
    return max(vals) if vals else 1.0


def _compute_stability(condensed, n):
    """Excess-of-mass stability per cluster, plus each cluster's birth lambda."""
    cap = _finite_max_lambda(condensed)
    cluster_ids = sorted({int(p) for (p, _c, _l, _s) in condensed})
    births = {}
    for p, c, l, _s in condensed:
        if c >= n:  # child is a cluster -> it is born at lambda l
            births[int(c)] = cap if not np.isfinite(l) else l
    if cluster_ids:
        births[cluster_ids[0]] = 0.0  # root: alive from lambda 0

    stability = {cid: 0.0 for cid in cluster_ids}
    for p, c, l, s in condensed:
        lam = cap if not np.isfinite(l) else l
        b = births.get(int(p), 0.0)
        stability[int(p)] += (lam - b) * s
    return stability, births, cap


def _extract_eom(condensed, stability, n, allow_single_cluster=False):
    """Excess-of-mass flat-cluster selection."""
    cluster_edges = [(int(p), int(c)) for (p, c, _l, _s) in condensed if c >= n]
    children_of = {}
    for p, c in cluster_edges:
        children_of.setdefault(p, []).append(c)

    stab = dict(stability)
    is_cluster = {cid: True for cid in stability}
    root = min(stability) if stability else None

    # Process bottom-up (largest ids first = deepest clusters first).
    for node in sorted(stability.keys(), reverse=True):
        kids = children_of.get(node, [])
        if not kids:
            continue  # leaf cluster keeps its own stability / selection
        subtotal = sum(stab[k] for k in kids)
        if node == root and not allow_single_cluster:
            # root never wins as a single flat cluster
            is_cluster[node] = False
            stab[node] = subtotal
            continue
        if subtotal > stab[node]:
            is_cluster[node] = False
            stab[node] = subtotal
        else:
            for desc in _cluster_descendants(children_of, node):
                is_cluster[desc] = False
    return [cid for cid, keep in is_cluster.items() if keep]


def _cluster_descendants(children_of, node):
    out = []
    stack = list(children_of.get(node, []))
    while stack:
        c = stack.pop()
        out.append(c)
        stack.extend(children_of.get(c, []))
    return out


def _labels_and_probs(condensed, selected, stability, births, cap, n):
    """Assign points to selected clusters with membership probabilities."""
    selected = sorted(selected)
    label_of = {c: i for i, c in enumerate(selected)}
    selected_set = set(selected)
    cluster_parent = {int(c): int(p) for (p, c, _l, _s) in condensed if c >= n}
    point_edges = [(int(p), int(c), l) for (p, c, l, _s) in condensed if c < n]

    # nearest selected ancestor (including self) of each cluster
    resolve_cache = {}

    def resolve(cl):
        if cl in resolve_cache:
            return resolve_cache[cl]
        cur = cl
        seen = []
        while cur is not None and cur not in selected_set:
            seen.append(cur)
            cur = cluster_parent.get(cur)
        for s in seen:
            resolve_cache[s] = cur
        resolve_cache[cl] = cur
        return cur

    home = {}
    for parent_cluster, pt, lam in point_edges:
        c = resolve(parent_cluster)
        lam = cap if not np.isfinite(lam) else lam
        home[pt] = (c, lam)

    max_lambda = {c: 0.0 for c in selected}
    for pt, (c, lam) in home.items():
        if c is not None and lam > max_lambda[c]:
            max_lambda[c] = lam

    labels = -np.ones(n, dtype=np.int64)
    probs = np.zeros(n, dtype=np.float64)
    for pt, (c, lam) in home.items():
        if c is None:
            continue
        labels[pt] = label_of[c]
        ml = max_lambda[c]
        probs[pt] = (lam / ml) if ml > 0 else 1.0

    # persistence in [0,1): how long the cluster lives past its birth
    persistence = np.zeros(len(selected), dtype=np.float64)
    for c in selected:
        ml = max_lambda[c]
        b = births.get(c, 0.0)
        persistence[label_of[c]] = ((ml - b) / ml) if ml > 0 else 0.0
    return labels, probs, persistence


# ──────────────────────────────────────────────────────────────────────────
#  Public HDBSCAN entry points
# ──────────────────────────────────────────────────────────────────────────
def hdbscan(
    X,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    device: str = "cpu",
) -> ClusterResult:
    """Cluster ``X`` with from-scratch HDBSCAN*.

    Parameters
    ----------
    X:
        ``(n_samples, n_features)`` array.
    min_cluster_size:
        Smallest grouping considered a cluster (the swept parameter).
    min_samples:
        Core-distance neighbourhood size; defaults to ``min_cluster_size``.
    cluster_selection_method:
        ``"eom"`` (excess of mass, default) or ``"leaf"``.
    allow_single_cluster:
        Permit the root to be selected as one flat cluster.
    device:
        ``"cpu"`` or ``"gpu"`` (GPU falls back to CPU when CuPy is absent).
    """
    xp = get_xp(device)
    X = xp.asarray(X, dtype=xp.float64)
    n = int(X.shape[0])
    mcs = max(int(min_cluster_size), 2)
    ms = mcs if min_samples is None else max(int(min_samples), 1)

    if n < mcs:
        z = np.zeros(n)
        return ClusterResult(-np.ones(n, dtype=np.int64), z, np.array([]), 0)

    D = _pairwise_distance(X, xp)
    core = _core_distances(D, ms, xp)
    mr = _mutual_reachability(D, core, xp)
    edges = _prim_mst(mr, xp)

    L = _single_linkage(edges, n)
    condensed = _condense_tree(L, mcs, n)
    if not condensed:
        z = np.zeros(n)
        return ClusterResult(-np.ones(n, dtype=np.int64), z, np.array([]), 0)

    stability, births, cap = _compute_stability(condensed, n)
    if cluster_selection_method == "leaf":
        selected = _leaf_clusters(condensed, stability, n, allow_single_cluster)
    else:
        selected = _extract_eom(condensed, stability, n, allow_single_cluster)

    labels, probs, persistence = _labels_and_probs(
        condensed, selected, stability, births, cap, n
    )
    return ClusterResult(labels, probs, persistence, len(selected))


def _leaf_clusters(condensed, stability, n, allow_single_cluster):
    """Leaf selection: every leaf of the condensed cluster tree is a cluster."""
    cluster_edges = [(int(p), int(c)) for (p, c, _l, _s) in condensed if c >= n]
    parents = {p for p, _c in cluster_edges}
    children = {c for _p, c in cluster_edges}
    all_clusters = parents | children
    leaves = [c for c in all_clusters if c not in parents]
    if not leaves and allow_single_cluster and stability:
        return [min(stability)]
    return leaves


def hdbscan_cpu(X, min_cluster_size, **kwargs) -> ClusterResult:
    """HDBSCAN on the CPU backend. Signature matches the sweep driver."""
    kwargs.pop("device", None)
    return hdbscan(X, min_cluster_size, device="cpu", **kwargs)


def hdbscan_gpu(X, min_cluster_size, **kwargs) -> ClusterResult:
    """HDBSCAN on the GPU backend (CuPy); transparently CPU if unavailable."""
    kwargs.pop("device", None)
    return hdbscan(X, min_cluster_size, device="gpu", **kwargs)


# ──────────────────────────────────────────────────────────────────────────
#  DBSCAN (from scratch) — kept because the sweep API supports eps sweeps too
# ──────────────────────────────────────────────────────────────────────────
def dbscan(X, eps: float, min_samples: int = 5, device: str = "cpu") -> ClusterResult:
    """Classic DBSCAN with a dense O(n^2) region query."""
    xp = get_xp(device)
    Xx = xp.asarray(X, dtype=xp.float64)
    n = int(Xx.shape[0])
    D = to_numpy(_pairwise_distance(Xx, xp))

    neighbors = D <= eps
    n_neighbors = neighbors.sum(axis=1)  # includes self
    is_core = n_neighbors >= min_samples

    labels = -np.ones(n, dtype=np.int64)
    cluster = 0
    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue
        labels[i] = cluster
        stack = list(np.nonzero(neighbors[i])[0])
        while stack:
            j = stack.pop()
            if labels[j] == -1:
                labels[j] = cluster
                if is_core[j]:
                    stack.extend(np.nonzero(neighbors[j])[0].tolist())
            elif labels[j] != cluster and is_core[j]:
                labels[j] = cluster
        cluster += 1

    probs = (labels >= 0).astype(np.float64)
    return ClusterResult(labels, probs, None, cluster)


def dbscan_cpu(X, eps, min_samples=5) -> ClusterResult:
    return dbscan(X, eps, min_samples, device="cpu")


def dbscan_gpu(X, eps, min_samples=5) -> ClusterResult:
    return dbscan(X, eps, min_samples, device="gpu")
