"""Statistical tests (06_EVALUATION_AND_STATS_SPEC.md #6): exact McNemar for
paired accuracy comparisons, paired bootstrap CIs for effect sizes. No
significance testing on tiny subgroups (06_EVALUATION_AND_STATS_SPEC.md #7).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import binomtest


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """Exact two-sided McNemar test over paired per-case correctness.

    `correct_a[i]`/`correct_b[i]` must refer to the SAME case i for both
    methods (same case_id order).
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("correct_a and correct_b must be paired (same length, same case order)")

    n_01 = sum(1 for a, b in zip(correct_a, correct_b) if (not a) and b)  # A wrong, B right
    n_10 = sum(1 for a, b in zip(correct_a, correct_b) if a and (not b))  # A right, B wrong
    n_discordant = n_01 + n_10

    if n_discordant == 0:
        p_value = 1.0
    else:
        result = binomtest(min(n_01, n_10), n_discordant, 0.5, alternative="two-sided")
        p_value = result.pvalue

    acc_a = sum(correct_a) / len(correct_a) if correct_a else float("nan")
    acc_b = sum(correct_b) / len(correct_b) if correct_b else float("nan")

    return {
        "n": len(correct_a),
        "n_01_a_wrong_b_right": n_01,
        "n_10_a_right_b_wrong": n_10,
        "n_discordant": n_discordant,
        "p_value": p_value,
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "accuracy_diff_b_minus_a": acc_b - acc_a,
    }


def paired_bootstrap_ci(
    values_a: list[float],
    values_b: list[float],
    statistic_fn: Callable[[list[float]], float] | None = None,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 20260921,
) -> dict:
    """Paired bootstrap CI for mean(values_b) - mean(values_a) (default), or
    for `statistic_fn(diffs)` if provided (06_EVALUATION_AND_STATS_SPEC.md #6).

    Each resample draws case INDICES with replacement (paired resampling),
    consistent with pairing accuracy/recall by case_id.
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must be paired (same length, same case order)")
    if not values_a:
        return {"point_estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_resamples": n_resamples}

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)

    point_estimate = float(np.mean(b) - np.mean(a))
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if statistic_fn is not None:
            diffs[i] = statistic_fn((b[idx] - a[idx]).tolist())
        else:
            diffs[i] = np.mean(b[idx]) - np.mean(a[idx])

    alpha = 1 - confidence_level
    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "point_estimate": point_estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
    }
