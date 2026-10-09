"""Stage 2 tests: tool checks and one reachable label per failure mode."""

import pytest

from agenteval.models import AgentResult, FailureMode, MatchType, Task, ToolCall
from agenteval.scoring import MAX_STEPS_REASON, check_tools, classify, score_trial


def make_task(**overrides: object) -> Task:
    payload: dict[str, object] = {
        "id": "t",
        "prompt": "What is 2 to the 10th power?",
        "expected": "1024",
        "match": MatchType.NUMERIC.value,
        "expected_tools": ["calculator"],
        "forbidden_tools": [],
        "max_steps": 4,
        **overrides,
    }
    return Task.from_dict(payload)


def ok_result(**overrides: object) -> AgentResult:
    base: dict[str, object] = {
        "answer": "2 ** 10 = 1024",
        "tool_calls": [ToolCall(step=1, tool="calculator", output="1024")],
        "steps": 2,
    }
    base.update(overrides)
    return AgentResult(**base)  # type: ignore[arg-type]


def test_check_tools_pass_and_fail() -> None:
    task = make_task()
    assert check_tools(task, ok_result()) == (True, [], [])
    task = make_task(expected_tools=["calculator", "kb_search"], forbidden_tools=["kb_search"])
    result = ok_result(tool_calls=[ToolCall(tool="kb_search", output="x")])
    tools_ok, forbidden, missing = check_tools(task, result)
    assert not tools_ok
    assert forbidden == ["kb_search"]
    assert missing == ["calculator"]


def test_check_tools_ignores_repeat_calls() -> None:
    task = make_task()
    result = ok_result(
        tool_calls=[
            ToolCall(tool="calculator", output="1"),
            ToolCall(tool="calculator", output="2"),
        ]
    )
    assert check_tools(task, result)[0]


@pytest.mark.parametrize(
    ("overrides", "expected_mode"),
    [
        ({"answer": "1023"}, FailureMode.WRONG_ANSWER),
        ({"answer": "", "steps": 4}, FailureMode.MAX_STEPS),
        ({"answer": "", "steps": 2, "error": "ConnectionError: boom"}, FailureMode.AGENT_ERROR),
        ({"answer": "", "steps": 1, "error": "boom"}, FailureMode.AGENT_ERROR),
    ],
)
def test_single_signals(overrides: dict, expected_mode: FailureMode) -> None:
    result = ok_result(**overrides)
    task = make_task()
    answer_ok = result.answer.strip() == "1024"
    _tools_ok, forbidden, missing = check_tools(task, result)
    assert classify(task, result, answer_ok=answer_ok, forbidden_used=forbidden, missing_required=missing) == expected_mode  # noqa: E501


def test_tool_error_mode() -> None:
    task = make_task()
    result = ok_result(
        answer="I could not calculate that",
        tool_calls=[ToolCall(tool="calculator", output="error: unsafe expression", is_error=True)],
        steps=2,
    )
    assert classify(task, result, answer_ok=False, forbidden_used=[], missing_required=[]) is FailureMode.TOOL_ERROR  # noqa: E501


def test_missing_tool_mode() -> None:
    task = make_task()
    result = AgentResult(answer="1024", steps=1)
    assert classify(task, result, answer_ok=True, forbidden_used=[], missing_required=["calculator"]) is FailureMode.MISSING_TOOL  # noqa: E501


def test_forbidden_tool_mode_beats_wrong_answer() -> None:
    task = make_task(forbidden_tools=["kb_search"])
    result = ok_result(answer="1023", tool_calls=[ToolCall(tool="kb_search", output="x")])
    assert classify(task, result, answer_ok=False, forbidden_used=["kb_search"], missing_required=[]) is FailureMode.FORBIDDEN_TOOL  # noqa: E501


def test_recovered_tool_error_still_passes() -> None:
    task = make_task()
    result = ok_result(
        answer="1024",
        tool_calls=[
            ToolCall(step=1, tool="calculator", output="error: unsafe expression", is_error=True),
            ToolCall(step=2, tool="calculator", output="1024"),
        ],
        steps=3,
    )
    assert classify(task, result, answer_ok=True, forbidden_used=[], missing_required=[]) is FailureMode.OK  # noqa: E501


def test_severity_order_is_resolved_in_one_label() -> None:
    task = make_task(expected_tools=["calculator"], forbidden_tools=["kb_search"])
    result = AgentResult(
        answer="",
        tool_calls=[ToolCall(tool="kb_search", output="x", is_error=True)],
        steps=9,
        error="kaboom",
    )
    mode = classify(task, result, answer_ok=False, forbidden_used=["kb_search"], missing_required=["calculator"])  # noqa: E501
    assert mode is FailureMode.AGENT_ERROR


def test_max_steps_flag_from_finish_reason_beats_forbidden() -> None:
    task = make_task(forbidden_tools=["kb_search"])
    result = ok_result(
        answer="1023",
        tool_calls=[ToolCall(tool="kb_search", output="x")],
        steps=1,
        finish_reason=MAX_STEPS_REASON,
    )
    assert classify(task, result, answer_ok=False, forbidden_used=["kb_search"], missing_required=[]) is FailureMode.MAX_STEPS  # noqa: E501


def test_every_failure_mode_is_reachable() -> None:
    modes = set()
    for factory in (
        lambda: (make_task(), ok_result(answer="1023")),
        lambda: (make_task(), ok_result(answer="", steps=4)),
        lambda: (make_task(), AgentResult(answer="", error="boom", steps=1)),
        lambda: (make_task(forbidden_tools=["kb_search"]), ok_result(answer="1023", tool_calls=[ToolCall(tool="kb_search", output="x")])),  # noqa: E501
        lambda: (make_task(expected_tools=["kb_search"]), ok_result(answer="1023", tool_calls=[])),
        lambda: (
            make_task(),
            ok_result(
                answer="1023",
                tool_calls=[ToolCall(tool="calculator", output="e", is_error=True)],
            ),
        ),
    ):
        task, result = factory()
        answer_ok, _tools_ok, forbidden, missing = (
            False,
            *check_tools(task, result),
        )
        modes.add(
            classify(task, result, answer_ok=answer_ok, forbidden_used=forbidden, missing_required=missing)  # noqa: E501
        )
    assert modes == set(FailureMode.severity_order())


def test_score_trial_returns_all_signals() -> None:
    task = make_task()
    answer_ok, tools_ok, passed, mode, forbidden, missing, detail = score_trial(task, ok_result())
    assert (answer_ok, tools_ok, passed) == (True, True, True)
    assert mode is FailureMode.OK
    assert (forbidden, missing, detail) == ([], [], "ok")


def test_score_trial_judge_without_verdict_is_not_a_pass() -> None:
    task = make_task(expected="any", match=MatchType.JUDGE.value)
    _ok, _tools, passed, mode, *_rest = score_trial(task, ok_result())
    assert not passed
    assert mode is FailureMode.WRONG_ANSWER


def test_score_trial_with_verdict_passes() -> None:
    task = make_task(expected="any", match=MatchType.JUDGE.value)
    _ok, _tools, passed, mode, *_rest = score_trial(task, ok_result(), verdict="PASS")
    assert passed
    assert mode is FailureMode.OK
