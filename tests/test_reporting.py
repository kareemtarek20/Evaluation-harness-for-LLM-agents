"""Stage 5 tests: summaries and the v1 -> v2 regression diff."""

import json
from pathlib import Path

from agenteval.agents import MockAgent
from agenteval.pricing import PriceTable
from agenteval.reporting import compare, load_summary, render_comparison, render_summary, summarize
from agenteval.runner import load_run, run_suite
from agenteval.taskset import load_task_file

ROOT = Path(__file__).resolve().parent.parent
BASIC = ROOT / "tasks" / "basic.json"
PRICES = PriceTable(input_per_mtok=3.0, output_per_mtok=15.0, source="test")


def artifacts() -> tuple[dict, dict]:
    base = run_suite(load_task_file(BASIC), MockAgent("v1"), trials=1, prices=PRICES)
    new = run_suite(load_task_file(BASIC), MockAgent("v2"), trials=1, prices=PRICES)
    return base, new


def test_summary_counts_and_rates() -> None:
    base, _new = artifacts()
    summary = summarize(base)
    assert summary.tasks == 10
    assert summary.trials == 10
    assert summary.passed == 1
    assert summary.success_rate == 0.1
    assert summary.answer_accuracy == 0.2
    assert summary.tool_accuracy == 0.2
    assert summary.trials_per_task == 1
    assert summary.cost_usd > 0


def test_summary_failure_modes_and_categories() -> None:
    base, _new = artifacts()
    summary = summarize(base)
    assert dict(summary.failure_modes) == {
        "missing_tool": 7,
        "forbidden_tool": 1,
        "wrong_answer": 1,
    }
    assert summary.per_category["retrieval"].passes == 0
    assert summary.per_category["retrieval"].trials == 4
    assert summary.per_category["math"].passes == 1
    assert summary.per_task["calc_power"].pass_rate == 1.0


def test_summary_json_is_serialisable() -> None:
    base, _new = artifacts()
    payload = summarize(base).to_dict()
    text = json.dumps(payload)
    assert payload["by_task"]["calc_power"]["passes"] == 1
    assert "failure_modes" in payload
    assert isinstance(text, str)


def test_compare_reports_calc_power_regressed_and_rest_fixed() -> None:
    base, new = artifacts()
    comparison = compare(base, new)
    assert [delta.task_id for delta in comparison.regressed] == ["calc_power"]
    assert comparison.regressed[0].base == "1/1"
    assert comparison.regressed[0].new == "0/1"
    assert {delta.task_id for delta in comparison.fixed} == {
        "calc_budget_split",
        "calc_hotel",
        "calc_meals",
        "hallucination_teleport",
        "kb_freeze",
        "kb_meals",
        "kb_rate_limit",
        "kb_refund",
        "no_tool_sick_days",
    }
    assert comparison.unchanged == []
    assert comparison.metric_deltas["success_rate"] > 0
    assert comparison.metric_deltas["avg_steps"] > 0
    assert comparison.metric_deltas["cost_usd"] > 0
    assert comparison.only_in_base == [] and comparison.only_in_new == []


def test_compare_identical_run_has_no_movement() -> None:
    base, _new = artifacts()
    comparison = compare(base, base)
    assert comparison.regressed == []
    assert comparison.fixed == []
    assert len(comparison.unchanged) == 10
    assert comparison.metric_deltas["success_rate"] == 0.0


def test_compare_handles_task_sets_that_differ() -> None:
    base, new = artifacts()
    trimmed = dict(new, results=[t for t in new["results"] if t["task_id"] != "calc_hotel"])
    comparison = compare(base, trimmed)
    assert comparison.only_in_base == ["calc_hotel"]
    assert all(delta.task_id != "calc_hotel" for delta in comparison.deltas)


def test_render_outputs_mention_the_numbers() -> None:
    base, new = artifacts()
    text = render_summary(summarize(base))
    assert "success rate   10.0%" in text
    assert "missing_tool   7" in text
    diff = render_comparison(compare(base, new))
    assert "- calc_power" in diff
    assert "+ kb_meals" in diff
    assert "regressed  1" in diff


def test_committed_demo_artifacts_match_the_expectation() -> None:
    """The files the README quotes must keep agreeing with these assertions."""
    base = load_run(ROOT / "runs" / "baseline-mock-v1.json")
    new = load_run(ROOT / "runs" / "new-mock-v2.json")
    assert summarize(base).passed == 1
    assert summarize(new).passed == 9
    assert [d.task_id for d in compare(base, new).regressed] == ["calc_power"]
    assert load_summary(ROOT / "runs" / "new-mock-v2.json").success_rate == 0.9
