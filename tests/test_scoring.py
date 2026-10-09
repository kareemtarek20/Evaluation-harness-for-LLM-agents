"""Stage 2 tests: every match type, including the numeric traps."""

import pytest

from agenteval.models import MatchType, Task
from agenteval.scoring import extract_numbers, last_number, score_answer


def make_task(expected: str, match: MatchType, **overrides: object) -> Task:
    payload: dict[str, object] = {
        "id": "t",
        "prompt": "q?",
        "expected": expected,
        "match": match.value,
        **overrides,
    }
    return Task.from_dict(payload)


def test_exact() -> None:
    task = make_task("1024", MatchType.EXACT)
    assert score_answer(task, "1024").ok
    assert score_answer(task, "  1024  ").ok
    assert not score_answer(task, "The answer is 1024").ok
    assert not score_answer(task, "1023").ok


def test_contains() -> None:
    task = make_task("7 business days", MatchType.CONTAINS)
    assert score_answer(task, "Refunds land in 7 business days.").ok
    assert score_answer(task, "refunds land in SEVEN business days").ok is False


def test_contains_any_pipe_separated() -> None:
    task = make_task("tuesday|thursday", MatchType.CONTAINS_ANY)
    assert score_answer(task, "Releases ship on Thursday.").ok
    assert not score_answer(task, "Releases ship on Monday.").ok
    assert score_answer(task, "Monday").ok is False


def test_numeric_basic_and_tolerance() -> None:
    task = make_task("225", MatchType.NUMERIC, tolerance=0.5)
    assert score_answer(task, "75 * 3 = 225").ok
    assert score_answer(task, "Total is 224.8 USD").ok
    assert not score_answer(task, "Total is 226 USD").ok
    tight = make_task("2.5", MatchType.NUMERIC)
    assert score_answer(tight, "2.5").ok
    assert not score_answer(tight, "2.51").ok


@pytest.mark.parametrize(
    ("answer", "want"),
    [("126.", 126.0), ("1,000", 1000.0), ("I paid $1,200.50.", 1200.5), ("-42", -42.0)],
)
def test_numeric_formats(answer: str, want: float) -> None:
    assert last_number(answer) == want
    assert score_answer(make_task(str(want), MatchType.NUMERIC), answer).ok


def test_numeric_uses_last_number_not_echoed_question() -> None:
    task = make_task("225", MatchType.NUMERIC)
    echo = "A trip costs 75 USD per day for 3 days?"
    assert not score_answer(task, echo).ok
    assert score_answer(task, f"{echo} The total is 225.").ok


def test_numeric_no_number_in_answer_fails_gracefully() -> None:
    task = make_task("225", MatchType.NUMERIC)
    check = score_answer(task, "I could not compute it")
    assert not check.ok
    assert "no number found" in check.detail


def test_number_extraction_details() -> None:
    assert extract_numbers("2,3") == [2.0, 3.0]
    assert extract_numbers("v1.2.3") == [1.2, 3.0]
    assert extract_numbers("1e3 and 1.5e2") == [1000.0, 150.0]
    assert last_number("no digits here") is None


def test_judge_match_requires_verdict() -> None:
    task = make_task("anything", MatchType.JUDGE, rubric="Names the cap and the currency")
    missing = score_answer(task, "The cap is 75 USD per day.")
    assert not missing.ok
    assert "judge verdict missing" in missing.detail
    assert score_answer(task, "whatever", "PASS").ok
    assert not score_answer(task, "whatever", "FAIL").ok


def test_empty_answer_fails_for_every_match_type() -> None:
    for match in MatchType:
        task = make_task("x", match)
        assert not score_answer(task, "", "PASS" if match is MatchType.JUDGE else None).ok
