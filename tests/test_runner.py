"""Stage 3 tests: runner bookkeeping - trials, timing, cost, crash capture, artifacts."""

import json
from pathlib import Path
from typing import Any

import pytest

from agenteval.agents import MockAgent
from agenteval.models import (
    AgentResult,
    FailureMode,
    MatchType,
    Task,
    ToolCall,
    TrialResult,
)
from agenteval.pricing import PriceTable
from agenteval.runner import (
    SCHEMA_VERSION,
    load_run,
    run_suite,
    run_trials,
    save_run,
    trials_from_artifact,
)
from agenteval.tools import ToolBox


def suite() -> list[Task]:
    payloads = [
        {
            "id": "calc_power",
            "prompt": "What is 2 to the 10th power?",
            "expected": "1024",
            "match": "numeric",
            "expected_tools": ["calculator"],
            "category": "math",
            "max_steps": 4,
        },
        {
            "id": "kb_retention",
            "prompt": "How long are application logs kept?",
            "expected": "90",
            "match": "numeric",
            "expected_tools": ["kb_search"],
            "category": "retrieval",
            "max_steps": 4,
        },
    ]
    return [Task.from_dict(payload) for payload in payloads]


class CrashyAgent:
    name = "stub:crashy"

    def run(self, task: Task, toolbox: ToolBox) -> Any:
        msg = f"exploded on {task.id}"
        raise RuntimeError(msg)


class KbAgent:
    """Stub agent that always searches and pastes the tool output as its answer."""

    name = "stub:kb"

    def run(self, task: Task, toolbox: ToolBox) -> AgentResult:
        output, is_error = toolbox.call("kb_search", {"query": "cafeteria open hours"})
        return AgentResult(
            answer=output,
            tool_calls=[ToolCall(step=1, tool="kb_search", output=output, is_error=is_error)],
            steps=1,
        )


def test_run_trials_matrix_is_tasks_times_trials() -> None:
    trials = run_trials(suite(), MockAgent("v1"), trials=3)
    assert len(trials) == 6
    assert {trial.task_id for trial in trials} == {"calc_power", "kb_retention"}
    assert sorted(trial.trial for trial in trials) == [0, 0, 1, 1, 2, 2]
    assert all(trial.latency_ms >= 0 for trial in trials)


def test_mock_runs_are_deterministic_across_trials() -> None:
    trials = run_trials(suite(), MockAgent("v2"), trials=4)
    per_task: dict[str, set[str]] = {}
    for trial in trials:
        payload = trial.to_dict()
        payload.pop("latency_ms")
        payload.pop("trial")
        per_task.setdefault(trial.task_id, set()).add(json.dumps(payload, sort_keys=True))
    assert all(len(signatures) == 1 for signatures in per_task.values())


def test_agent_exception_becomes_agent_error_trial() -> None:
    trials = run_trials(suite(), CrashyAgent(), trials=1)
    assert len(trials) == 2
    assert all(trial.failure_mode is FailureMode.AGENT_ERROR for trial in trials)
    assert all(not trial.passed for trial in trials)
    assert "RuntimeError" in (trials[0].result.error or "")


def test_rejects_bad_trial_count() -> None:
    with pytest.raises(ValueError, match="trials must be"):
        run_trials(suite(), MockAgent("v1"), trials=0)


def test_cost_comes_from_configured_prices() -> None:
    prices = PriceTable(input_per_mtok=3.0, output_per_mtok=15.0, source="test")
    (trial,) = run_trials([suite()[0]], MockAgent("v1"), trials=1, prices=prices)
    expected = prices.cost_usd(trial.input_tokens, trial.output_tokens)
    assert trial.cost_usd == pytest.approx(expected)
    assert trial.cost_usd > 0
    free = PriceTable(0.0, 0.0)
    (free_trial,) = run_trials([suite()[0]], MockAgent("v1"), trials=1, prices=free)
    assert free_trial.cost_usd == 0.0


def test_price_table_lookup_and_overrides() -> None:
    haiku = PriceTable.for_model("claude-haiku-4-5")
    assert (haiku.input_per_mtok, haiku.output_per_mtok) == (1.0, 5.0)
    assert haiku.source == "bundled-table"
    unknown = PriceTable.for_model("gpt-not-listed")
    assert (unknown.input_per_mtok, unknown.output_per_mtok) == (0.0, 0.0)
    assert unknown.source == "unpriced-model"
    override = PriceTable.for_model("claude-sonnet-4-5", output_price=9.0)
    assert (override.input_per_mtok, override.output_per_mtok) == (3.0, 9.0)
    assert override.cost_usd(1_000_000, 1_000_000) == pytest.approx(12.0)


def test_judge_is_only_called_for_judge_tasks() -> None:
    seen: list[str] = []

    def judge(task: Task, answer: str) -> tuple[str, str]:
        seen.append(task.id)
        return "PASS", f"graded {answer!r}"

    tasks = suite() + [
        Task.from_dict(
            {
                "id": "open_question",
                "prompt": "Is the freeze reasonable?",
                "expected": "any",
                "match": MatchType.JUDGE.value,
                "rubric": "gives a reason",
            }
        )
    ]
    trials = run_trials(tasks, MockAgent("v2"), trials=1, judge=judge)
    assert seen == ["open_question"]
    graded = next(trial for trial in trials if trial.task_id == "open_question")
    assert graded.judge_verdict == "PASS"
    assert graded.passed


def test_run_suite_artifact_shape_and_round_trip(tmp_path: Path) -> None:
    artifact = run_suite(
        suite(),
        MockAgent("v1"),
        trials=2,
        prices=PriceTable(2.0, 8.0, source="test"),
        task_source="tasks/basic.json",
        label="mock-baseline",
        config={"system_prompt": "v1"},
    )
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["agent"] == "mock:v1"
    assert artifact["trials_per_task"] == 2
    assert len(artifact["results"]) == 4
    assert artifact["prices"]["input_per_mtok"] == 2.0
    assert artifact["run_id"]

    path = save_run(artifact, tmp_path / "runs" / "demo.json")
    assert path.exists()
    loaded = load_run(path)
    assert loaded == json.loads(path.read_text(encoding="utf-8"))
    trials = trials_from_artifact(loaded)
    assert all(isinstance(trial, TrialResult) for trial in trials)
    assert sum(trial.passed for trial in trials) == 2  # calc_power passes in both trials


def test_load_run_rejects_foreign_files(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not an agenteval run artifact"):
        load_run(other)


def test_knowledge_base_override_reaches_the_agent_toolbox() -> None:
    fixture = [
        {"id": "x", "title": "Cafeteria hours", "body": "Open from 7 to 15.", "tags": ["food"]}
    ]
    task = Task.from_dict(
        {
            "id": "kb_cafeteria",
            "prompt": "When is the cafeteria open?",
            "expected": "7 to 15",
            "match": "contains",
            "expected_tools": ["kb_search"],
        }
    )
    (trial,) = run_trials([task], KbAgent(), trials=1, knowledge_base=fixture)
    assert trial.passed, trial.result.tool_calls[0].output
    (default_trial,) = run_trials([task], KbAgent(), trials=1)
    assert not default_trial.passed
    assert "No matching records" in default_trial.result.tool_calls[0].output
