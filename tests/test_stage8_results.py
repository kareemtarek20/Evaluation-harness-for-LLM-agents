"""Stage 8 guard: docs/results.md must keep agreeing with the committed artifacts.

The brief forbids numbers that were not measured. This test recomputes every
headline figure in that page from `runs/*.json` and fails if the document no
longer contains it, so the table cannot drift away from the data.
"""

from pathlib import Path

from agenteval.reporting import compare, summarize
from agenteval.runner import load_run

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")

ROWS = {
    "stage8-basic-v1-trials5": "basic (10 tasks) | A `mock:v1`",
    "stage8-basic-v2-trials5": "basic (10 tasks) | B `mock:v2`",
    "stage8-extended-v1-trials5": "extended (46 tasks) | A `mock:v1`",
    "stage8-extended-v2-trials5": "extended (46 tasks) | B `mock:v2`",
}


def row_text(name: str, prefix: str) -> str:
    """Rebuild one table row from its artifact."""
    summary = summarize(load_run(ROOT / "runs" / f"{name}.json"))
    modes = ", ".join(f"{mode} {count}" for mode, count in summary.failure_modes.most_common())
    flaky = ", ".join(summary.flaky_tasks) if summary.flaky_tasks else "none"
    return (
        f"| {prefix} | {summary.success_rate:.1%} ({summary.passed}/{summary.trials})"
        f" | {summary.answer_accuracy:.1%} | {summary.tool_accuracy:.1%}"
        f" | {summary.avg_steps:.2f} | {summary.cost_usd:.4f} | {flaky} | {modes} |"
    )


def test_every_result_row_matches_its_artifact() -> None:
    for name, prefix in ROWS.items():
        expected = row_text(name, prefix)
        assert expected in DOC, f"docs/results.md is out of date for {name}:\n{expected}"


def test_trial_counts_and_labels_are_the_ones_measured() -> None:
    for name, tasks, trials in (
        ("stage8-basic-v1-trials5", 10, 5),
        ("stage8-basic-v2-trials5", 10, 5),
        ("stage8-extended-v1-trials5", 46, 5),
        ("stage8-extended-v2-trials5", 46, 5),
    ):
        artifact = load_run(ROOT / "runs" / f"{name}.json")
        assert artifact["trials_per_task"] == trials
        summary = summarize(artifact)
        assert summary.tasks == tasks
        assert summary.trials == tasks * trials
        assert artifact["prices"]["source"] == "cli-override"
        assert "stage-8" in artifact["label"]


def test_quoted_deltas_match_the_comparisons() -> None:
    basic = compare(
        load_run(ROOT / "runs" / "stage8-basic-v1-trials5.json"),
        load_run(ROOT / "runs" / "stage8-basic-v2-trials5.json"),
    )
    extended = compare(
        load_run(ROOT / "runs" / "stage8-extended-v1-trials5.json"),
        load_run(ROOT / "runs" / "stage8-extended-v2-trials5.json"),
    )
    assert len(basic.fixed) == 9
    assert [d.task_id for d in basic.regressed] == ["calc_power"]
    assert basic.regressed[0].p_value < 0.01
    assert [d.task_id for d in extended.fixed] == ["tool_error_caret", "inject_roster"]
    assert extended.regressed == []
    assert f"+{basic.metric_deltas['success_rate'] * 100:.1f} pp" in DOC
    assert f"+{extended.metric_deltas['success_rate'] * 100:.1f} pp" in DOC
    assert f"+${basic.metric_deltas['cost_usd']:.4f}" in DOC
    assert f"-${abs(extended.metric_deltas['cost_usd']):.4f}" in DOC


def test_cited_failure_trials_exist_verbatim_in_the_artifacts() -> None:
    """The three quoted cases must still say what the document claims they say."""
    power = [
        trial
        for trial in load_run(ROOT / "runs" / "stage8-basic-v2-trials5.json")["results"]
        if trial["task_id"] == "calc_power"
    ][0]
    assert power["result"]["tool_calls"][0]["input"] == {"expression": "2 ^ 10"}
    assert power["result"]["tool_calls"][0]["is_error"] is True
    assert "BitXor" in power["result"]["tool_calls"][0]["output"]
    assert power["failure_mode"] == "tool_error"

    meals = [
        trial
        for trial in load_run(ROOT / "runs" / "stage8-extended-v2-trials5.json")["results"]
        if trial["task_id"] == "kb_intl_meals"
    ][0]
    assert meals["failure_mode"] == "missing_tool"

    weeks = [
        trial
        for trial in load_run(ROOT / "runs" / "stage8-extended-v2-trials5.json")["results"]
        if trial["task_id"] == "no_tool_weeks"
    ][0]
    assert weeks["failure_mode"] == "max_steps"
    assert weeks["steps"] == 1


def test_the_document_states_the_real_run_was_skipped() -> None:
    lowered = DOC.lower()
    assert "no anthropic credentials" in lowered or "api_key set: false" in lowered
    assert "not yet verified" in lowered or "what this run does *not* establish" in lowered
    assert "agent_error" in DOC
    assert "build_sampling_kwargs" in DOC
