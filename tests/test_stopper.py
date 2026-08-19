"""Smart-stopper logic: warmup, patience, chaos guard, winning config."""
import numpy as np
import pytest

from densweep import StopperConfig, StopChecker, WINNING_CONFIG
from densweep.metrics import IterationSignals


def _hist(composites, ncls=None):
    ncls = ncls if ncls is not None else [4] * len(composites)
    return [IterationSignals(sweep_value=i, n_clusters=ncls[i], noise_ratio=0.1,
                             persistence_sum=0.0, persistence_max=0.0,
                             mean_membership=0.0, silhouette=0.0, composite=c)
            for i, c in enumerate(composites)]


def test_winning_config_is_the_default():
    cfg = StopperConfig()
    assert cfg.warmup == WINNING_CONFIG["warmup"] == 10
    assert cfg.patience == WINNING_CONFIG["patience"] == 15
    assert cfg.min_delta == WINNING_CONFIG["min_delta"] == 0.01
    assert cfg.chaos_window == WINNING_CONFIG["chaos_window"] == 20
    assert cfg.chaos_agreement_min == WINNING_CONFIG["chaos_agreement_min"] == 0.30


def test_never_stops_during_warmup():
    chk = StopChecker(StopperConfig(warmup=10, patience=2))
    # flat composites would trip patience, but warmup must veto it
    for i in range(1, 9):
        d = chk.check(_hist([0.0] * i))
        assert not d.should_stop
        assert d.reason in ("warmup", "empty", "continue")


def test_patience_triggers_after_plateau():
    cfg = StopperConfig(warmup=5, patience=8, chaos_agreement_min=0.0)
    chk = StopChecker(cfg)
    # rise to a peak at index 6, then a long flat plateau
    comps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] + [0.6] * 20
    stop_at = None
    for i in range(1, len(comps) + 1):
        d = chk.check(comps_hist := _hist(comps[:i]))
        if d.should_stop:
            stop_at = i - 1
            assert d.reason == "patience"
            assert d.best_idx == 6
            break
    assert stop_at is not None
    # peak(6) + patience(8) ≈ stop
    assert 13 <= stop_at <= 16


def test_best_idx_tracks_argmax():
    cfg = StopperConfig(warmup=3, patience=100)
    chk = StopChecker(cfg)
    comps = [0.1, 0.5, 0.9, 0.4, 0.3, 0.2]
    d = chk.check(_hist(comps))
    assert d.best_idx == 2


def test_chaos_guard_stops_when_structure_unstable():
    # past the peak, cluster counts all different -> low agreement -> chaos stop
    cfg = StopperConfig(warmup=4, patience=50, chaos_window=10, chaos_agreement_min=0.5)
    chk = StopChecker(cfg)
    comps = [0.0, 0.2, 0.5, 0.9] + [0.1] * 12        # clear early peak, then decline
    ncls = [3, 3, 3, 3] + list(range(4, 16))          # wildly varying -> chaotic
    stopped = False
    for i in range(1, len(comps) + 1):
        d = chk.check(_hist(comps[:i], ncls[:i]))
        if d.should_stop and d.reason == "chaos":
            stopped = True
            break
    assert stopped


def test_empty_history():
    d = StopChecker().check([])
    assert not d.should_stop and d.reason == "empty"
