"""Shared fixtures / helpers for the Densweep test-suite."""
import numpy as np
import pytest

try:
    from sklearn.datasets import make_blobs
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False

requires_sklearn = pytest.mark.skipif(not HAVE_SKLEARN, reason="scikit-learn not installed")


def _fallback_blobs(n, k, std, seed):
    """Minimal blob generator so core tests run without scikit-learn."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-12, 12, size=(k, 2))
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    X, y = [], []
    for ci, (c, s) in enumerate(zip(centers, sizes)):
        X.append(rng.normal(c, std, size=(s, 2)))
        y += [ci] * s
    return np.vstack(X).astype(np.float64), np.array(y)


def blobs(n=400, k=4, std=0.55, seed=0):
    """Well-separated 2D blobs (UMAP-embedding-like)."""
    if HAVE_SKLEARN:
        rng = np.random.default_rng(seed)
        centers = rng.uniform(-12, 12, size=(k, 2))
        X, y = make_blobs(n_samples=n, centers=centers, cluster_std=std, random_state=seed)
        return X.astype(np.float64), y
    return _fallback_blobs(n, k, std, seed)


@pytest.fixture
def small_blobs():
    return blobs(n=300, k=4, std=0.5, seed=1)
