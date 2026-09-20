"""Regression gate (01_METHOD_SPEC.md #5): accept/rollback a batch of candidate
memory updates based on a fixed development replay subset.

`replay_score_fn` is injected so this module has no dependency on the
pipeline/dataset - it only knows how to compare two scalar scores. This
keeps it independently unit-testable with a fake scoring function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .gap_memory import GapMemory
from .norm_memory import NormMemory

ReplayScoreFn = Callable[[NormMemory, GapMemory], float]


@dataclass
class GateResult:
    accepted: bool
    score_before: float
    score_after: float
    norm_memory: NormMemory
    gap_memory: GapMemory
    reason: str = ""


def run_regression_gate(
    current_norm: NormMemory,
    current_gap: GapMemory,
    proposed_norm: NormMemory,
    proposed_gap: GapMemory,
    replay_score_fn: ReplayScoreFn,
    epsilon: float = 0.0,
) -> GateResult:
    """Compare `proposed_*` against `current_*` on the replay set.

    Rolls back (returns the `current_*` memories) if the proposed batch
    decreases replay accuracy by more than `epsilon` (01_METHOD_SPEC.md #5:
    "rollback updates if accuracy decreases by > epsilon, epsilon=0 by
    default"). Ties (score_after == score_before - epsilon) are accepted.
    """
    score_before = replay_score_fn(current_norm, current_gap)
    score_after = replay_score_fn(proposed_norm, proposed_gap)

    if score_after >= score_before - epsilon:
        return GateResult(
            accepted=True,
            score_before=score_before,
            score_after=score_after,
            norm_memory=proposed_norm,
            gap_memory=proposed_gap,
            reason="accepted: no regression beyond epsilon",
        )

    return GateResult(
        accepted=False,
        score_before=score_before,
        score_after=score_after,
        norm_memory=current_norm,
        gap_memory=current_gap,
        reason=f"rolled back: replay score dropped {score_before:.4f} -> {score_after:.4f}",
    )


def accept_without_gate(proposed_norm: NormMemory, proposed_gap: GapMemory) -> GateResult:
    """Used by the `no_regression_gate` ablation (05_EXPERIMENT_PLAN.md A4):
    every source-valid candidate batch is accepted unconditionally.
    """
    return GateResult(
        accepted=True,
        score_before=float("nan"),
        score_after=float("nan"),
        norm_memory=proposed_norm,
        gap_memory=proposed_gap,
        reason="accepted: regression gate disabled (ablation)",
    )
