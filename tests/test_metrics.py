"""Metrics: silhouette, DBCV, ARI, and the composite scorer."""
import numpy as np
import pytest

from densweep import silhouette_score, dbcv_score, adjusted_rand_score
from densweep.metrics import IterationSignals, CompositeScorer
from conftest import blobs, requires_sklearn


def test_ari_bounds_and_identity():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_score(a, a) == pytest.approx(1.0)
    # invariant to label permutation
    b = np.array([2, 2, 0, 0, 1, 1])
    assert adjusted_rand_score(a, b) == pytest.approx(1.0)


@requires_sklearn
def test_silhouette_matches_sklearn():
    from sklearn.metrics import silhouette_score as sk_sil
    for seed in range(3):
        X, y = blobs(n=300, k=4, std=0.7, seed=seed)
        assert silhouette_score(X, y) == pytest.approx(sk_sil(X, y), abs=1e-6)


@requires_sklearn
def test_ari_matches_sklearn():
    from sklearn.metrics import adjusted_rand_score as sk_ari
    rng = np.random.default_rng(0)
    for _ in range(3):
        a = rng.integers(0, 5, 200)
        b = rng.integers(0, 5, 200)
        assert adjusted_rand_score(a, b) == pytest.approx(sk_ari(a, b), abs=1e-9)


def test_dbcv_separates_good_from_bad():
    Xg, yg = blobs(n=300, k=3, std=0.35, seed=0)
    Xb, yb = blobs(n=300, k=3, std=3.0, seed=0)
    assert dbcv_score(Xg, yg) > dbcv_score(Xb, yb)
    assert dbcv_score(Xg, yg) > 0


def _sig(ncl, noise, sil, pmax, dbcv=None, mem=0.9):
    return IterationSignals(
        sweep_value=ncl, n_clusters=ncl, noise_ratio=noise,
        persistence_sum=pmax, persistence_max=pmax, mean_membership=mem,
        silhouette=sil, dbcv=dbcv,
    )


def test_composite_zero_during_warmup_and_degenerate():
    sc = CompositeScorer(warmup=5)
    # degenerate single-cluster -> 0 regardless
    c = sc.update(_sig(ncl=1, noise=0.0, sil=0.9, pmax=0.8))
    assert c == 0.0
    # still inside warmup window
    for _ in range(3):
        c = sc.update(_sig(ncl=4, noise=0.1, sil=0.8, pmax=0.7, dbcv=0.6))
    assert c == 0.0


def test_composite_rewards_quality_over_noise():
    sc = CompositeScorer(warmup=0)
    # establish range
    for _ in range(3):
        sc.update(_sig(ncl=3, noise=0.5, sil=0.2, pmax=0.2, dbcv=0.1))
    clean = sc.update(_sig(ncl=6, noise=0.02, sil=0.9, pmax=0.9, dbcv=0.9))
    noisy = sc.update(_sig(ncl=6, noise=0.8, sil=0.9, pmax=0.9, dbcv=0.9))
    assert clean > noisy  # the noise gate bites


def test_composite_is_finite_and_bounded():
    sc = CompositeScorer(warmup=0)
    vals = []
    for i in range(20):
        vals.append(sc.update(_sig(ncl=2 + i % 5, noise=0.1 * (i % 4),
                                    sil=0.5, pmax=0.5, dbcv=0.5)))
    vals = np.array(vals)
    assert np.all(np.isfinite(vals))
    assert np.all((vals >= 0) & (vals <= 1))
