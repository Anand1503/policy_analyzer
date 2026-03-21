"""
Statistical Testing — Significance testing for model evaluation.
Module 7: Paired t-test, bootstrap confidence intervals, p-value computation.

Design:
  - Purely computational (no DB, no external libraries)
  - Deterministic (fixed seed for bootstrap)
  - Implements tests from scratch (no scipy dependency)
"""

import math
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Paired t-Test
# ═══════════════════════════════════════════════════════════════

def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, Any]:
    """
    Paired two-sided t-test for comparing two models.

    Tests H0: mean(A) == mean(B) vs H1: mean(A) != mean(B).

    Args:
        scores_a: Per-sample scores from model A (e.g. per-clause F1)
        scores_b: Per-sample scores from model B

    Returns:
        t-statistic, p-value, significance assessment
    """
    n = len(scores_a)
    if n != len(scores_b):
        return {"error": "Score lists must have equal length"}
    if n < 2:
        return {"error": "Need at least 2 samples for t-test"}

    # Compute differences
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = math.sqrt(var_diff) if var_diff > 0 else 1e-10

    # t-statistic
    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Degrees of freedom
    df = n - 1

    # p-value approximation using t-distribution CDF
    p_value = _t_distribution_p_value(abs(t_stat), df) * 2  # Two-sided

    # Significance assessment
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    elif p_value < 0.10:
        significance = "marginally significant (p < 0.10)"
    else:
        significance = "not significant (p >= 0.10)"

    return {
        "t_statistic": round(t_stat, 4),
        "p_value": round(min(p_value, 1.0), 6),
        "degrees_of_freedom": df,
        "mean_difference": round(mean_diff, 4),
        "std_difference": round(std_diff, 4),
        "significance": significance,
        "reject_h0_at_005": p_value < 0.05,
    }


# ═══════════════════════════════════════════════════════════════
# Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════

def bootstrap_confidence_interval(
    scores: List[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute bootstrap confidence interval for the mean of scores.

    Uses deterministic seed for reproducibility.

    Args:
        scores: List of per-sample scores
        confidence_level: 0.90, 0.95, or 0.99
        n_bootstrap: Number of bootstrap iterations
        seed: Fixed seed for reproducibility

    Returns:
        Mean, CI lower/upper, standard error
    """
    n = len(scores)
    if n < 2:
        return {"error": "Need at least 2 samples for bootstrap"}

    # Deterministic LCG pseudo-random generator
    rng = _LCG(seed)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()

    # Percentile method
    alpha = 1.0 - confidence_level
    lower_idx = max(0, int(math.floor(alpha / 2 * n_bootstrap)))
    upper_idx = min(n_bootstrap - 1, int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1)

    mean_val = sum(scores) / n
    ci_lower = bootstrap_means[lower_idx]
    ci_upper = bootstrap_means[upper_idx]
    std_error = _std(bootstrap_means)

    return {
        "mean": round(mean_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "confidence_level": confidence_level,
        "std_error": round(std_error, 4),
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
    }


def bootstrap_f1_confidence(
    per_sample_f1: List[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute bootstrap confidence interval specifically for F1 scores.

    Args:
        per_sample_f1: Per-clause F1 scores
        confidence_level: Confidence level (default 0.95)

    Returns:
        F1 mean with confidence bounds
    """
    result = bootstrap_confidence_interval(
        per_sample_f1, confidence_level, n_bootstrap, seed,
    )
    if "error" in result:
        return result

    result["metric"] = "f1_score"
    result["interpretation"] = (
        f"F1 score is {result['mean']:.4f} "
        f"({confidence_level*100:.0f}% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}])"
    )
    return result


# ═══════════════════════════════════════════════════════════════
# Full Statistical Report
# ═══════════════════════════════════════════════════════════════

def run_statistical_tests(
    hybrid_scores: List[float],
    baseline_scores: List[float],
    metric_name: str = "F1",
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Run full statistical test suite comparing two models.

    Args:
        hybrid_scores: Per-sample scores from hybrid model
        baseline_scores: Per-sample scores from baseline
        metric_name: Name of the metric being compared

    Returns:
        Complete statistical report
    """
    return {
        "metric": metric_name,
        "n_samples": len(hybrid_scores),
        "hybrid_summary": {
            "mean": round(sum(hybrid_scores) / max(len(hybrid_scores), 1), 4),
            "std": round(_std(hybrid_scores), 4),
            "min": round(min(hybrid_scores) if hybrid_scores else 0, 4),
            "max": round(max(hybrid_scores) if hybrid_scores else 0, 4),
        },
        "baseline_summary": {
            "mean": round(sum(baseline_scores) / max(len(baseline_scores), 1), 4),
            "std": round(_std(baseline_scores), 4),
            "min": round(min(baseline_scores) if baseline_scores else 0, 4),
            "max": round(max(baseline_scores) if baseline_scores else 0, 4),
        },
        "paired_t_test": paired_t_test(hybrid_scores, baseline_scores),
        "hybrid_confidence_interval": bootstrap_confidence_interval(
            hybrid_scores, confidence_level,
        ),
        "baseline_confidence_interval": bootstrap_confidence_interval(
            baseline_scores, confidence_level,
        ),
    }


# ═══════════════════════════════════════════════════════════════
# Math Helpers
# ═══════════════════════════════════════════════════════════════

class _LCG:
    """Linear Congruential Generator — deterministic PRNG."""

    def __init__(self, seed: int):
        self.state = seed

    def next(self) -> int:
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state

    def randint(self, low: int, high: int) -> int:
        return low + (self.next() % (high - low + 1))


def _std(values: List[float]) -> float:
    """Standard deviation (sample)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _t_distribution_p_value(t: float, df: int) -> float:
    """
    Approximate one-sided p-value for t-distribution.
    Uses the approximation from Abramowitz & Stegun.
    Accurate for df >= 3.
    """
    if df <= 0:
        return 1.0

    # Approximate using normal distribution for large df
    if df > 100:
        return _normal_sf(t)

    # Hill's approximation for smaller df
    x = df / (df + t * t)
    if df == 1:
        p = 1.0 - math.atan(t) / (math.pi / 2)
        return max(0, min(1, p / 2))

    # Regularized incomplete beta function approximation
    a = df / 2.0
    b = 0.5
    p = _incomplete_beta(x, a, b) / 2.0
    return max(0, min(1, p))


def _normal_sf(x: float) -> float:
    """Survival function (1 - CDF) for standard normal."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """Approximation of regularized incomplete beta function."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Simple continued fraction approximation
    # Good enough for research validation (±0.01 accuracy)
    lnbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lnbeta) / a

    # Lentz's algorithm for continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 100):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= c * d

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-8:
            break

    return max(0, min(1, front * f))
