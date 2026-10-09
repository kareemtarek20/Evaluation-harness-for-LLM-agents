"""Small-sample statistics for repeated trials.

Why Fisher's exact test rather than a bootstrap here: a task run is a binomial
observation (pass or fail) and the comparison is a 2x2 table. With the 3-10
trials a suite can realistically afford, the normal approximation and bootstrap
intervals are both noisy about their own noise, while Fisher's test conditions
on the margins and gives an exact p-value from integer binomial coefficients -
no sampling, no seed, fully reproducible in CI. A z-test is kept only as a
fallback for tables too large to enumerate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["NoiseCheck", "fisher_exact_two_sided", "test_pass_counts"]

#: Above this many combined trials, enumerate no more and use the z-test.
MAX_EXACT_TRIALS = 200


@dataclass(frozen=True)
class NoiseCheck:
    """A pass/fail comparison and whether it clears the noise floor."""

    p_value: float
    method: str
    alpha: float

    @property
    def significant(self) -> bool:
        """True when the difference is unlikely to be trial noise."""
        return self.p_value < self.alpha

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dict."""
        return {
            "p_value": round(self.p_value, 6),
            "method": self.method,
            "alpha": self.alpha,
            "significant": self.significant,
        }


def _hypergeom_pmf(x: int, row1: int, col1: int, total: int) -> float:
    """P(X = x) for the number of successes in row 1 of the table."""
    return math.comb(row1, x) * math.comb(total - row1, col1 - x) / math.comb(total, col1)


def fisher_exact_two_sided(
    base_passes: int, base_fails: int, new_passes: int, new_fails: int
) -> float:
    """Two-sided Fisher exact p-value for the 2x2 table.

    The table is [[base_passes, base_fails], [new_passes, new_fails]]; the
    p-value sums every table at least as extreme (probability <= the observed
    one) under fixed margins.
    """
    total = base_passes + base_fails + new_passes + new_fails
    row1 = base_passes + base_fails
    col1 = base_passes + new_passes
    if total == 0 or row1 == 0 or col1 == 0 or row1 == total or col1 == total:
        return 1.0
    low = max(0, col1 - (total - row1))
    high = min(row1, col1)
    observed = _hypergeom_pmf(base_passes, row1, col1, total)
    p_value = 0.0
    for x in range(low, high + 1):
        probability = _hypergeom_pmf(x, row1, col1, total)
        if probability <= observed * (1 + 1e-9):
            p_value += probability
    return min(1.0, p_value)


def two_proportion_z(base_passes: int, base_trials: int, new_passes: int, new_trials: int) -> float:
    """Two-sided p-value from a pooled two-proportion z-test (large samples)."""
    pooled = (base_passes + new_passes) / (base_trials + new_trials)
    if pooled in (0.0, 1.0):
        return 1.0
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / base_trials + 1 / new_trials))
    if standard_error == 0:
        return 1.0
    statistic = (new_passes / new_trials - base_passes / base_trials) / standard_error
    return min(1.0, math.erfc(abs(statistic) / math.sqrt(2)))


def test_pass_counts(
    base_passes: int,
    base_trials: int,
    new_passes: int,
    new_trials: int,
    *,
    alpha: float = 0.05,
) -> NoiseCheck:
    """Compare two pass counts and say whether the move beats noise.

    Returns method ``"single-trial"`` with p=1.0 when neither side has
    repetitions, because one trial per task cannot estimate variance at all.
    """
    if base_trials <= 0 or new_trials <= 0:
        return NoiseCheck(1.0, "no-data", alpha)
    if base_trials == 1 and new_trials == 1:
        return NoiseCheck(1.0, "single-trial", alpha)
    total = base_trials + new_trials
    if total > MAX_EXACT_TRIALS:
        return NoiseCheck(
            two_proportion_z(base_passes, base_trials, new_passes, new_trials), "z-test", alpha
        )
    p_value = fisher_exact_two_sided(
        base_passes, base_trials - base_passes, new_passes, new_trials - new_passes
    )
    return NoiseCheck(p_value, "fisher-exact", alpha)


def is_flaky(passes: int, trials: int) -> bool:
    """True when a task with repetitions passes sometimes and fails others."""
    return trials > 1 and 0 < passes < trials
