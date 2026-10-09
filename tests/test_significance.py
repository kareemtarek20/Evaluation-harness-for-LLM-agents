"""Stage 6 tests: repeated trials, flakiness and noise-aware CI gating.

These drive the whole pipeline (runner -> reporting -> CLI) with a stub agent
that passes an exact number of trials, so the pass counts under test are the
counts the brief names: 3/5 -> 2/5 must stay quiet, 5/5 -> 0/5 must gate.
"""

from pathlib import Path
from typing import Any

import pytest

from agenteval.agents import MockAgent
from agenteval.cli import main
from agenteval.models import AgentResult, Task
from agenteval.pricing import PriceTable
from agenteval.reporting import compare, render_comparison, render_summary, summarize
from agenteval.runner import run_suite, save_run
from agenteval.taskset import load_task_file
from agenteval.tools import ToolBox

ROOT = Path(__file__).resolve().parent.parent
BASIC = ROOT / "tasks" / "basic.json"
PRICES = PriceTable(input_per_mtok=3.0, output_per_mtok=15.0, source="test")


class CountedAgent:
    """An agent that passes the first ``passes`` trials of every task."""

    name = "stub:counted"
    model = ""

    def __init__(self, passes: int) -> None:
        self.passes = passes
        self._seen: dict[str, int] = {}

    def run(self, task: Task, toolbox: ToolBox) -> Any:
        index = self._seen.get(task.id, 0)
        self._seen[task.id] = index + 1
        answer = task.expected if index < self.passes else "not the expected answer"
        return AgentResult(answer=answer, steps=1, input_tokens=10, output_tokens=3)


def make_suite(count: int = 1) -> list[Task]:
    """Return ``count`` trivial exact-match tasks."""
    return [
        Task.from_dict(
            {
                "id": f"flip_{index}",
                "prompt": "say the answer",
                "expected": "the answer",
                "match": "exact",
                "category": "stability",
            }
        )
        for index in range(count)
    ]


def pair(base_passes: int, new_passes: int | None = None, trials: int = 5) -> tuple[dict, dict]:
    """Run the same one-task suite twice with the given pass counts."""
    tasks = make_suite()
    target = base_passes if new_passes is None else new_passes
    base = run_suite(tasks, CountedAgent(base_passes), trials=trials, prices=PRICES)
    new = run_suite(tasks, CountedAgent(target), trials=trials, prices=PRICES)
    return base, new


def test_three_of_five_to_two_of_five_is_not_gated() -> None:
    base, new = pair(3, 2)
    delta = compare(base, new).regressed[0]
    assert (delta.base, delta.new) == ("3/5", "2/5")
    assert delta.method == "fisher-exact"
    assert delta.p_value > 0.05
    assert not delta.significant
    assert not delta.gating


def test_five_of_five_to_zero_of_five_is_gated() -> None:
    base, new = pair(5, 0)
    comparison = compare(base, new)
    delta = comparison.regressed[0]
    assert delta.significant
    assert delta.gating
    assert [d.task_id for d in comparison.gating_regressions] == ["flip_0"]
    assert comparison.noise_regressions == []


def test_improvements_are_not_gated_even_when_significant() -> None:
    base, new = pair(0, 5)
    comparison = compare(base, new)
    assert [d.task_id for d in comparison.fixed] == ["flip_0"]
    assert comparison.gating_regressions == []


def test_flaky_task_is_detected_from_its_trials() -> None:
    base, _new = pair(3)
    summary = summarize(base)
    assert summary.flaky_tasks == ["flip_0"]
    assert summary.per_task["flip_0"].flaky
    assert summary.passed == 3
    assert summary.trials == 5
    text = render_summary(summary)
    assert "flaky tasks" in text
    assert "flip_0" in text.split("per task")[0]


def test_steady_run_has_no_flakiness() -> None:
    base, _new = pair(5)
    assert summarize(base).flaky_tasks == []
    assert "flaky tasks" not in render_summary(summarize(base))


def test_alpha_controls_the_noise_floor() -> None:
    """4/5 -> 1/5 is p=0.206: noise at 0.05, a real regression at 0.25."""
    base, new = pair(4, 1)
    strict = compare(base, new, alpha=0.25)
    assert strict.regressed[0].significant
    assert len(strict.gating_regressions) == 1
    default = compare(base, new)
    assert not default.regressed[0].significant
    assert len(default.noise_regressions) == 1


def test_render_comparison_labels_both_outcomes() -> None:
    noise_base, noise_new = pair(3, 2)
    text = render_comparison(compare(noise_base, noise_new))
    assert "within noise" in text
    assert "fisher-exact" in text
    assert "gating     0 of 1" in text

    real_base, real_new = pair(5, 0)
    gated = render_comparison(compare(real_base, real_new))
    assert "significant" in gated
    assert "gating     1 of 1" in gated


def test_cli_exits_zero_for_a_noisy_regression(tmp_path: Path, capsys) -> None:
    base, new = pair(3, 2)
    base_path = save_run(base, tmp_path / "base.json")
    new_path = save_run(new, tmp_path / "new.json")
    capsys.readouterr()
    code = main(["compare", str(base_path), str(new_path), "--fail-on-regression"])
    assert code == 0
    printed = capsys.readouterr()
    assert "within noise: flip_0" in printed.err
    assert "FAIL" not in printed.err


def test_cli_exits_one_for_a_real_regression(tmp_path: Path, capsys) -> None:
    base, new = pair(5, 0)
    base_path = save_run(base, tmp_path / "base.json")
    new_path = save_run(new, tmp_path / "new.json")
    capsys.readouterr()
    code = main(["compare", str(base_path), str(new_path), "--fail-on-regression"])
    assert code == 1
    assert "FAIL: 1 regression(s) beyond noise (alpha=0.05): flip_0" in capsys.readouterr().err


def test_cli_alpha_flag_widens_the_gate(tmp_path: Path, capsys) -> None:
    base, new = pair(4, 1)
    base_path = save_run(base, tmp_path / "base.json")
    new_path = save_run(new, tmp_path / "new.json")
    capsys.readouterr()
    assert main(["compare", str(base_path), str(new_path), "--fail-on-regression"]) == 0
    capsys.readouterr()
    code = main(
        ["compare", str(base_path), str(new_path), "--fail-on-regression", "--alpha", "0.25"]
    )
    assert code == 1
    capsys.readouterr()


def test_five_trial_mock_comparison_is_stable_and_finds_the_real_regression() -> None:
    """v1 -> v2 over 5 trials: the scripted bug shows up as significant, not noise."""
    tasks = load_task_file(BASIC)
    base = run_suite(tasks, MockAgent("v1"), trials=5, prices=PRICES)
    new = run_suite(tasks, MockAgent("v2"), trials=5, prices=PRICES)
    base_summary, new_summary = summarize(base), summarize(new)
    assert (base_summary.passed, base_summary.trials) == (5, 50)
    assert (new_summary.passed, new_summary.trials) == (45, 50)
    assert base_summary.success_rate == pytest.approx(0.1)
    assert new_summary.success_rate == pytest.approx(0.9)
    # scripted agents are deterministic, so five trials must not invent flakiness
    assert base_summary.flaky_tasks == []
    assert new_summary.flaky_tasks == []

    comparison = compare(base, new)
    assert [delta.task_id for delta in comparison.gating_regressions] == ["calc_power"]
    regression = comparison.regressed[0]
    assert (regression.base, regression.new) == ("5/5", "0/5")
    assert regression.p_value < 0.05
    assert len(comparison.fixed) == 9
    assert comparison.metric_deltas["success_rate"] == pytest.approx(0.8)
