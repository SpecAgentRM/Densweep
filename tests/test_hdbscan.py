"""From-scratch HDBSCAN / DBSCAN correctness."""
import numpy as np
import pytest

from densweep import hdbscan, hdbscan_cpu, dbscan, ClusterResult, adjusted_rand_score
from conftest import blobs, requires_sklearn


def test_recovers_well_separated_blobs():
    X, y = blobs(n=400, k=5, std=0.45, seed=3)
    res = hdbscan(X, min_cluster_size=15)
    assert isinstance(res, ClusterResult)
    assert res.n_clusters == 5
    assert adjusted_rand_score(y, res.labels) > 0.95


def test_probabilities_and_persistence_shapes():
    X, y = blobs(n=300, k=4, std=0.5, seed=0)
    res = hdbscan(X, min_cluster_size=10)
    assert res.labels.shape == (300,)
    assert res.probabilities.shape == (300,)
    assert ((res.probabilities >= 0) & (res.probabilities <= 1)).all()
    # one persistence value per discovered cluster, all finite & non-negative
    assert res.cluster_persistence.shape[0] == res.n_clusters
    assert np.all(np.isfinite(res.cluster_persistence))
    assert np.all(res.cluster_persistence >= 0)


def test_noise_points_have_zero_probability():
    X, _ = blobs(n=300, k=3, std=0.4, seed=2)
    # add a few far-away outliers
    X = np.vstack([X, np.array([[100.0, 100.0], [-100.0, 80.0]])])
    res = hdbscan(X, min_cluster_size=20)
    noise = res.labels == -1
    if noise.any():
        assert np.allclose(res.probabilities[noise], 0.0)


def test_min_cluster_size_monotonicity():
    X, _ = blobs(n=500, k=6, std=0.5, seed=5)
    k_small = hdbscan(X, 5).n_clusters
    k_large = hdbscan(X, 120).n_clusters
    # coarser min_cluster_size should not produce *more* clusters
    assert k_large <= k_small


def test_tiny_input_is_all_noise():
    X = np.random.RandomState(0).randn(3, 2)
    res = hdbscan(X, min_cluster_size=10)
    assert (res.labels == -1).all()
    assert res.n_clusters == 0


def test_dbscan_basic():
    X, y = blobs(n=400, k=4, std=0.4, seed=0)
    res = dbscan(X, eps=1.0, min_samples=5)
    assert res.n_clusters >= 3
    assert adjusted_rand_score(y, res.labels) > 0.8


@requires_sklearn
def test_matches_sklearn_hdbscan():
    from sklearn.cluster import HDBSCAN as SKH
    agree = []
    for seed in range(4):
        X, _ = blobs(n=400, k=5, std=0.6, seed=seed)
        for mcs in (10, 25):
            ours = hdbscan_cpu(X, mcs).labels
            sk = SKH(min_cluster_size=mcs).fit(X).labels_
            agree.append(adjusted_rand_score(ours, sk))
    # near-identical partitions vs the reference implementation
    assert np.mean(agree) > 0.95
