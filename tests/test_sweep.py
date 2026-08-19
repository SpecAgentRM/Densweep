"""End-to-end sweep: early stop, speedup, quality vs the oracle."""
import numpy as np
import pytest

from densweep import (
    Densweep, smart_sweep, hdbscan_cpu, make_mcs_range, make_eps_range,
    adjusted_rand_score, StopperConfig,
)
from conftest import blobs


def test_smart_sweep_stops_early():
    X, _ = blobs(n=700, k=6, std=0.5, seed=1)
    vals = make_mcs_range(2, 180)
    res = smart_sweep(X, hdbscan_cpu, vals, mode="active")
    assert res.stopped_early
    assert res.n_evaluated < len(vals)
    assert res.speedup > 2.0
    assert res.best_sweep_value is not None


def test_passive_mode_runs_full_sweep():
    X, _ = blobs(n=300, k=4, std=0.5, seed=0)
    vals = make_mcs_range(2, 60)
    res = smart_sweep(X, hdbscan_cpu, vals, mode="passive")
    assert not res.stopped_early
    assert res.n_evaluated == len(vals)


def test_speedup_and_quality_vs_oracle():
    """The headline behaviour: big iteration saving, near-oracle ARI."""
    speedups, qualities = [], []
    for seed in (11, 12, 13):
        X, y = blobs(n=800, k=7, std=0.5, seed=seed)
        vals = make_mcs_range(2, min(200, X.shape[0] // 4))
        labels = {v: hdbscan_cpu(X, v).labels for v in vals}   # cache once
        aris = [adjusted_rand_score(y, labels[v]) for v in vals]
        oracle = max(aris)
        res = smart_sweep(X, hdbscan_cpu, vals, mode="active")
        smart_ari = adjusted_rand_score(y, labels[int(res.best_sweep_value)])
        speedups.append(len(vals) / res.n_evaluated)
        qualities.append(smart_ari / oracle if oracle > 0 else 1.0)
    assert np.mean(speedups) >= 4.0          # substantial iteration saving
    assert np.mean(qualities) >= 0.95        # retains (almost) all the quality


def test_densweep_estimator():
    X, y = blobs(n=500, k=5, std=0.45, seed=2)
    model = Densweep().fit(X)
    assert model.labels_.shape == (500,)
    assert model.best_min_cluster_size_ is not None
    assert model.result_.stopped_early
    assert adjusted_rand_score(y, model.labels_) > 0.9
    # fit_predict convenience
    lp = Densweep().fit_predict(X)
    assert lp.shape == (500,)


def test_dbscan_eps_sweep_runs():
    X, _ = blobs(n=300, k=4, std=0.4, seed=0)
    vals = make_eps_range(X, n_steps=40)
    assert len(vals) == 40 and vals[0] < vals[-1]
    model = Densweep(algorithm="dbscan", sweep_values=vals).fit(X)
    assert model.labels_.shape == (300,)
