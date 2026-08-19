"""
densweep.stopper
────────────────
The smart-stopper: decide when a clustering parameter sweep has seen enough.

It watches the per-iteration ``composite`` quality signal and stops once that
signal has plateaued (classic patience-based early stopping), with an extra
guard that cuts the sweep short when the clustering has become *chaotic* — i.e.
the cluster count stops agreeing with itself across a recent window, a sign the
swept parameter has wandered into a structureless regime.

The defaults below are the configuration that won the DGX-Spark search,
delivering ≈6.5× fewer sweep iterations while retaining ≈97.5% of the
oracle full-sweep clustering quality (ARI):

    warmup=10, patience=15, min_delta=0.01,
    chaos_window=20, chaos_agreement_min=0.30
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence


# The single source of truth for the winning configuration.
WINNING_CONFIG = dict(
    warmup=10,
    patience=15,
    min_delta=0.01,
    chaos_window=20,
    chaos_agreement_min=0.30,
)


@dataclass
class StopperConfig:
    """Hyper-parameters of the smart-stopper (winning defaults baked in).

    Parameters
    ----------
    warmup:
        Minimum iterations before a stop may be issued.
    patience:
        Stop after this many consecutive iterations without a ``min_delta``
        improvement in the composite score.
    min_delta:
        Smallest composite gain that counts as an improvement.
    chaos_window:
        Number of most-recent iterations examined for cluster-count agreement.
    chaos_agreement_min:
        If the fraction of the recent window agreeing on the modal cluster
        count drops below this (and we are already past the peak), the sweep is
        deemed chaotic and stopped early.
    """

    warmup: int = 10
    patience: int = 15
    min_delta: float = 0.01
    chaos_window: int = 20
    chaos_agreement_min: float = 0.30

    def __post_init__(self):
        self.warmup = int(self.warmup)
        self.patience = int(self.patience)
        self.min_delta = float(self.min_delta)
        self.chaos_window = int(self.chaos_window)
        self.chaos_agreement_min = float(self.chaos_agreement_min)


@dataclass
class StopDecision:
    """Result of a single stop check."""

    should_stop: bool
    reason: str          # "warmup" | "continue" | "patience" | "chaos" | "empty"
    best_idx: int        # index of the best composite seen so far
    no_improve: int = 0  # consecutive non-improving iterations
    agreement: float = 1.0  # recent cluster-count agreement in [0, 1]


def _modal_fraction(values: Sequence) -> float:
    if not values:
        return 1.0
    counts = Counter(values)
    return counts.most_common(1)[0][1] / len(values)


class StopChecker:
    """Stateless stop oracle: feed it the history-so-far, get a decision.

    ``check`` recomputes everything from the supplied history slice, so the same
    checker can be reused and a sweep can be replayed deterministically (this is
    exactly how the offline calibration evaluated candidate configs).
    """

    def __init__(self, config: StopperConfig | None = None):
        self.config = config or StopperConfig()

    def check(self, history: List) -> StopDecision:
        cfg = self.config
        comps = [float(s.composite) for s in history]
        L = len(comps)
        if L == 0:
            return StopDecision(False, "empty", 0)

        # Best-so-far (argmax) is the config we'd actually keep.
        best_idx = max(range(L), key=lambda i: comps[i])

        # Patience bookkeeping using the min_delta improvement rule.
        running_best = float("-inf")
        no_improve = 0
        for c in comps:
            if c > running_best + cfg.min_delta:
                running_best = c
                no_improve = 0
            else:
                no_improve += 1

        # Recent cluster-count agreement (chaos detector).
        recent = history[max(0, L - cfg.chaos_window):]
        agreement = _modal_fraction([s.n_clusters for s in recent])

        # Never stop during warmup.
        if L - 1 < cfg.warmup:
            return StopDecision(False, "warmup", best_idx, no_improve, agreement)

        # Primary rule: plateau in the composite.
        if no_improve >= cfg.patience:
            return StopDecision(True, "patience", best_idx, no_improve, agreement)

        # Guard rule: past the peak and the structure has gone chaotic. This is
        # an *early* out — independent of patience — so an unstable tail does not
        # have to be sat through in full.
        past_peak = best_idx < L - 1 and no_improve >= 3
        if past_peak and agreement < cfg.chaos_agreement_min:
            return StopDecision(True, "chaos", best_idx, no_improve, agreement)

        return StopDecision(False, "continue", best_idx, no_improve, agreement)
