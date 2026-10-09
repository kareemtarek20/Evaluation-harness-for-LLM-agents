"""Stage 6 tests: the noise model itself."""

import math

import pytest

from agenteval.stats import (
    MAX_EXACT_TRIALS,
    fisher_exact_two_sided,
    is_flaky,
    two_proportion_z,
)
from agenteval.stats import test_pass_counts as noise_check


def test_three_of_five_to_two_of_five_is_not_significant() -> None:
    check = noise_check(3, 5, 2, 5)
    assert check.method == "fisher-exact"
    assert check.p_value > 0.05
    assert not check.significant


def test_five_of_five_to_zero_of_five_is_significant() -> None:
    check = noise_check(5, 5, 0, 5)
    assert check.p_value < 0.05
    assert check.significant
    assert check.p_value == pytest.approx(fisher_exact_two_sided(5, 0, 0, 5))


def test_symmetric_small_moves_are_noise() -> None:
    assert not noise_check(5, 5, 4, 5).significant
    assert not noise_check(4, 5, 3, 5).significant
    assert not noise_check(0, 5, 0, 5).significant


def test_single_trial_cannot_estimate_noise() -> None:
    check = noise_check(1, 1, 0, 1)
    assert check.method == "single-trial"
    assert check.p_value == 1.0
    assert not check.significant


def test_missing_data_is_not_significant() -> None:
    assert noise_check(0, 0, 1, 3).method == "no-data"
    assert noise_check(2, 4, 0, 0).p_value == 1.0


def test_larger_samples_use_the_z_fallback() -> None:
    assert MAX_EXACT_TRIALS == 200
    check = noise_check(500, 1000, 440, 1000)
    assert check.method == "z-test"
    assert check.significant
    assert not noise_check(500, 1000, 480, 1000).significant


def test_fisher_math_matches_known_tables() -> None:
    # [[5,0],[0,5]]: only one table is as extreme as the observed one, per side.
    assert fisher_exact_two_sided(5, 0, 0, 5) == pytest.approx(2 / math.comb(10, 5))
    assert fisher_exact_two_sided(1, 1, 1, 1) == pytest.approx(1.0)
    assert fisher_exact_two_sided(2, 0, 0, 2) == pytest.approx(2 / 6)
    assert 0.0 <= fisher_exact_two_sided(3, 7, 7, 3) <= 1.0


def test_degenerate_margins_return_one() -> None:
    assert fisher_exact_two_sided(0, 0, 0, 0) == 1.0
    assert fisher_exact_two_sided(3, 0, 0, 0) == 1.0


def test_z_test_edge_cases() -> None:
    assert two_proportion_z(0, 100, 0, 100) == 1.0
    assert two_proportion_z(100, 100, 100, 100) == 1.0
    assert 0.0 <= two_proportion_z(60, 100, 40, 100) < 0.01


def test_flakiness_rule() -> None:
    assert is_flaky(3, 5)
    assert is_flaky(1, 2)
    assert not is_flaky(5, 5)
    assert not is_flaky(0, 5)
    assert not is_flaky(1, 1)
    assert not is_flaky(0, 1)
