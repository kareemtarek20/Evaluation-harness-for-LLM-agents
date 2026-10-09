"""Stage 1 tests: task/tool result models round-trip through JSON."""

import json

import pytest

from agenteval.models import (
    AgentResult,
    FailureMode,
    MatchType,
    Task,
    ToolCall,
    TrialResult,
    task_list,
)

SAMPLE = {
    "id": "calc_power",
    "prompt": "What is 2 to the 10th power?",
    "expected": "1024",
    "match": "numeric",
    "category": "math",
    "expected_tools": ["calculator"],
    "forbidden_tools": [],
    "max_steps": 4,
    "tolerance": 0.001,
}


def test_task_from_dict_round_trip() -> None:
    task = Task.from_dict(SAMPLE)
    assert task.match is MatchType.NUMERIC
    assert task.expected_tools == ("calculator",)
    assert task.rubric == ""
    assert task.tolerance == pytest.approx(0.001)
    assert json.loads(json.dumps(task.to_dict()))["match"] == "numeric"
    assert Task.from_dict(task.to_dict()) == task


def test_task_defaults() -> None:
    task = Task.from_dict({"id": "t", "prompt": "p", "expected": "e", "match": "exact"})
    assert task.category == "general"
    assert task.max_steps == 8
    assert task.forbidden_tools == ()


def test_task_rejects_missing_and_unknown_fields() -> None:
    missing = {k: v for k, v in SAMPLE.items() if k != "expected"}
    with pytest.raises(ValueError, match="missing expected"):
        Task.from_dict(missing)
    with pytest.raises(ValueError, match="unknown field"):
        Task.from_dict({**SAMPLE, "temperture": 0})
    with pytest.raises(ValueError, match="unknown match type"):
        Task.from_dict({**SAMPLE, "match": "fuzzy"})


def test_task_list_reports_offending_index() -> None:
    with pytest.raises(ValueError, match=r"task #1"):
        task_list([SAMPLE, {"id": "b"}])


def test_agent_result_tools_used_and_crash() -> None:
    result = AgentResult(
        answer="1024",
        tool_calls=[
            ToolCall(step=1, tool="calculator", input={"expression": "2**10"}, output="1024")
        ],
        steps=2,
    )
    assert result.tools_used() == ["calculator"]
    assert not result.crashed
    assert AgentResult(error="boom").crashed


def test_failure_modes_are_unique_and_ordered() -> None:
    order = FailureMode.severity_order()
    assert len(order) == len(set(order))
    assert order[0] is FailureMode.AGENT_ERROR
    assert order[-1] is FailureMode.WRONG_ANSWER
    assert FailureMode.OK not in order


def test_trial_result_round_trip_keeps_tool_calls() -> None:
    trial = TrialResult(
        task_id="calc_power",
        trial=0,
        agent="mock:v1",
        result=AgentResult(
            answer="1024",
            tool_calls=[ToolCall(step=1, tool="calculator", output="1024")],
            steps=1,
            input_tokens=10,
            output_tokens=4,
        ),
        passed=True,
        answer_ok=True,
        tools_ok=True,
        failure_mode=FailureMode.OK,
        steps=1,
        latency_ms=12.5,
        input_tokens=10,
        output_tokens=4,
        cost_usd=0.000125,
    )
    payload = json.loads(json.dumps(trial.to_dict()))
    restored = TrialResult.from_dict(payload)
    assert restored == trial
    assert restored.result.tool_calls[0].tool == "calculator"
    assert payload["failure_mode"] == "ok"


def test_trial_result_rounds_money_and_latency() -> None:
    trial = TrialResult(task_id="t", trial=1, latency_ms=12.34567, cost_usd=0.000123456)
    payload = trial.to_dict()
    assert payload["latency_ms"] == 12.346
    assert payload["cost_usd"] == 0.000123
