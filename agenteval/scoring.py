"""Answer matching, tool-use checking, and single-label failure classification.

All functions here are pure: they take a task plus an ``AgentResult`` (or the
pieces of one) and return booleans or an enum. That is what makes the failure
modes testable without an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agenteval.models import AgentResult, FailureMode, MatchType, Task

__all__ = ["AnswerCheck", "MAX_STEPS_REASON", "check_tools", "classify", "score_answer"]

#: finish_reason set by an agent that ran out of its step budget.
MAX_STEPS_REASON = "max_steps"

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")
_GROUPED_COMMA_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_PLAIN_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_PASS_TOKENS = frozenset({"pass", "passed", "true", "yes", "ok", "1"})


@dataclass(frozen=True)
class AnswerCheck:
    """Outcome of comparing an answer to the expectation."""

    ok: bool
    detail: str

    def __bool__(self) -> bool:
        """Let callers treat the check as its boolean result."""
        return self.ok


def normalize_text(text: str) -> str:
    """Collapse whitespace so formatting differences do not change the verdict."""
    return " ".join(text.split())


def extract_numbers(text: str) -> list[float]:
    """Return every number in ``text`` in the order they appear.

    ``1,000`` is one thousands-separated number; ``2,3`` is not, so it is read
    as its plain trailing numbers. Exponents and decimals are supported.
    """
    found: list[float] = []
    for match in _NUMBER_RE.findall(text):
        token = match
        if "," in token:
            if not _GROUPED_COMMA_RE.match(token):
                inner = _PLAIN_NUMBER_RE.findall(token)
                found.extend(float(part) for part in inner)
                continue
            token = token.replace(",", "")
        try:
            found.append(float(token))
        except ValueError:
            continue
    return found


def last_number(text: str) -> float | None:
    """Return the last number in ``text``, or None when it holds none.

    Reading the *last* number is deliberate: agents often restate the question
    before answering, so the final number is the answer rather than an input.
    """
    numbers = extract_numbers(text)
    return numbers[-1] if numbers else None


def numbers_close(actual: float, expected: float, tolerance: float) -> bool:
    """Compare with an absolute tolerance that grows for large expectations."""
    relative = abs(expected) * 1e-6
    return abs(actual - expected) <= max(tolerance, relative)


def _judge_pass(verdict: str | None) -> bool:
    """Interpret a judge verdict string as a pass/fail."""
    if verdict is None:
        return False
    return verdict.strip().lower().rstrip(".") in _PASS_TOKENS


def score_answer(
    task: Task, answer: str, verdict: str | None = None, *, found_answer: bool = True
) -> AnswerCheck:
    """Compare ``answer`` to the task expectation for the task's match type.

    Args:
        task: the task carrying ``expected``, ``match`` and ``tolerance``.
        answer: the agent's final text.
        verdict: judge verdict for ``match == "judge"`` tasks.
        found_answer: False when the agent produced nothing to compare.
    """
    if not found_answer:
        return AnswerCheck(False, "no final answer")
    text = normalize_text(answer or "")
    expected = normalize_text(task.expected)
    if not text:
        return AnswerCheck(False, "empty answer")
    match task.match:
        case MatchType.EXACT:
            ok = text.casefold() == expected.casefold()
            return AnswerCheck(ok, f"exact: want {expected!r}, got {text!r}")
        case MatchType.CONTAINS:
            ok = expected.casefold() in text.casefold()
            return AnswerCheck(ok, f"contains {expected!r}: {'yes' if ok else 'no'}")
        case MatchType.CONTAINS_ANY:
            options = [part.strip() for part in task.expected.split("|") if part.strip()]
            hit = next((o for o in options if o.casefold() in text.casefold()), None)
            return AnswerCheck(hit is not None, f"contains_any {options}: matched {hit!r}")
        case MatchType.NUMERIC:
            want = last_number(task.expected)
            if want is None:
                return AnswerCheck(False, "numeric task has no number in expected")
            got = last_number(text)
            if got is None:
                return AnswerCheck(False, f"no number found in answer {text!r}")
            ok = numbers_close(got, want, task.tolerance)
            return AnswerCheck(ok, f"numeric: got {got}, want {want} (+/-{task.tolerance:g})")
        case MatchType.JUDGE:
            if verdict is None:
                return AnswerCheck(
                    False, "judge verdict missing: run with a judge (stage 7) for this task"
                )
            return AnswerCheck(_judge_pass(verdict), f"judge verdict: {verdict!r}")
    raise ValueError(f"unhandled match type {task.match!r}")


def check_tools(task: Task, result: AgentResult) -> tuple[bool, list[str], list[str]]:
    """Return ``(tools_ok, forbidden_used, missing_required)`` for a run.

    Required tools only need to appear at least once, in any order or repeat
    count; efficiency tasks therefore rely on ``max_steps`` instead.
    """
    used = result.tools_used()
    forbidden_used = sorted({name for name in task.forbidden_tools if name in used})
    missing_required = sorted({name for name in task.expected_tools if name not in used})
    return (not forbidden_used and not missing_required), forbidden_used, missing_required


def classify(
    task: Task,
    result: AgentResult,
    *,
    answer_ok: bool,
    forbidden_used: list[str],
    missing_required: list[str],
) -> FailureMode:
    """Return exactly one label for a trial, most severe first.

    A passing trial is ``FailureMode.OK``. ``tool_error`` only applies to a run
    that got the answer wrong: recovering from a tool error and still answering
    correctly is a pass.
    """
    hit_step_budget = result.finish_reason == MAX_STEPS_REASON or (
        not answer_ok and result.steps >= task.max_steps
    )
    checks: dict[FailureMode, bool] = {
        FailureMode.AGENT_ERROR: result.crashed,
        FailureMode.MAX_STEPS: hit_step_budget,
        FailureMode.FORBIDDEN_TOOL: bool(forbidden_used),
        FailureMode.MISSING_TOOL: bool(missing_required),
        FailureMode.TOOL_ERROR: any(call.is_error for call in result.tool_calls) and not answer_ok,
        FailureMode.WRONG_ANSWER: not answer_ok,
    }
    for mode in FailureMode.severity_order():
        if checks[mode]:
            return mode
    return FailureMode.OK


def score_trial(
    task: Task, result: AgentResult, *, verdict: str | None = None
) -> tuple[bool, bool, bool, FailureMode, list[str], list[str], str]:
    """One call to get every scoring signal for a (task, result) pair.

    Returns:
        ``(answer_ok, tools_ok, passed, failure_mode, forbidden_used,
        missing_required, detail)``.
    """
    answer_check = score_answer(task, result.answer, verdict, found_answer=not result.crashed)
    tools_ok, forbidden_used, missing_required = check_tools(task, result)
    mode = classify(
        task,
        result,
        answer_ok=answer_check.ok,
        forbidden_used=forbidden_used,
        missing_required=missing_required,
    )
    passed = mode is FailureMode.OK and answer_check.ok
    detail = answer_check.detail if not answer_check.ok else "ok"
    if forbidden_used:
        detail = f"{detail}; forbidden tools used: {', '.join(forbidden_used)}"
    if missing_required:
        detail = f"{detail}; missing required tools: {', '.join(missing_required)}"
    return answer_check.ok, tools_ok, passed, mode, forbidden_used, missing_required, detail
