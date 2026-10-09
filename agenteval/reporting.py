"""Aggregate a run artifact into a summary and diff two runs.

The run file stores trials only; every number here is recomputed from those
trials, so a report can never disagree with its own data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agenteval.models import FailureMode, TrialResult
from agenteval.runner import load_run, trials_from_artifact

__all__ = ["Comparison", "Summary", "TaskDelta", "compare", "summarize"]


@dataclass
class TaskAggregate:
    """Per-task roll-up across trials."""

    task_id: str
    category: str = ""
    trials: int = 0
    passes: int = 0
    steps_total: int = 0
    latency_total_ms: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    failure_modes: Counter[str] = field(default_factory=Counter)

    @property
    def pass_rate(self) -> float:
        """Share of trials that passed."""
        return self.passes / self.trials if self.trials else 0.0

    @property
    def avg_steps(self) -> float:
        """Mean tool+answer steps per trial."""
        return self.steps_total / self.trials if self.trials else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Mean wall-clock latency per trial."""
        return self.latency_total_ms / self.trials if self.trials else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "task_id": self.task_id,
            "category": self.category,
            "trials": self.trials,
            "passes": self.passes,
            "pass_rate": round(self.pass_rate, 4),
            "avg_steps": round(self.avg_steps, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
            "failure_modes": dict(self.failure_modes),
        }


@dataclass
class Summary:
    """Headline metrics for one run."""

    run_id: str
    agent: str
    label: str
    model: str
    task_source: str
    trials_per_task: int
    tasks: int
    trials: int
    passed: int
    answer_ok: int
    tools_ok: int
    steps_total: int
    latency_total_ms: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    per_category: dict[str, TaskAggregate]
    per_task: dict[str, TaskAggregate]
    failure_modes: Counter[str]
    prices: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Share of trials that passed."""
        return self.passed / self.trials if self.trials else 0.0

    @property
    def answer_accuracy(self) -> float:
        """Share of trials whose answer matched."""
        return self.answer_ok / self.trials if self.trials else 0.0

    @property
    def tool_accuracy(self) -> float:
        """Share of trials that satisfied required/forbidden tool constraints."""
        return self.tools_ok / self.trials if self.trials else 0.0

    @property
    def avg_steps(self) -> float:
        """Mean steps per trial."""
        return self.steps_total / self.trials if self.trials else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Mean latency per trial."""
        return self.latency_total_ms / self.trials if self.trials else 0.0

    @property
    def total_cost_usd(self) -> float:
        """Run cost in USD under the configured prices."""
        return self.cost_usd

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the whole summary."""
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "label": self.label,
            "model": self.model,
            "task_source": self.task_source,
            "trials_per_task": self.trials_per_task,
            "tasks": self.tasks,
            "trials": self.trials,
            "passed": self.passed,
            "success_rate": round(self.success_rate, 4),
            "answer_accuracy": round(self.answer_accuracy, 4),
            "tool_accuracy": round(self.tool_accuracy, 4),
            "avg_steps": round(self.avg_steps, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "prices": self.prices,
            "failure_modes": dict(self.failure_modes),
            "by_category": {key: agg.to_dict() for key, agg in sorted(self.per_category.items())},
            "by_task": {key: agg.to_dict() for key, agg in sorted(self.per_task.items())},
        }


def summarize(
    artifact: dict[str, Any], task_categories: dict[str, str] | None = None
) -> Summary:
    """Roll a run artifact up into a Summary.

    Args:
        artifact: a loaded run artifact.
        task_categories: optional task id -> category map, normally taken from
            the suite that was run; without it categories come from the
            artifact's own labels where available.
    """
    trials = trials_from_artifact(artifact)
    categories = dict(artifact.get("categories", {}) or {})
    categories.update(task_categories or {})
    per_task: dict[str, TaskAggregate] = {}
    failure_modes: Counter[str] = Counter()
    for trial in trials:
        agg = per_task.setdefault(
            trial.task_id,
            TaskAggregate(task_id=trial.task_id, category=categories.get(trial.task_id, "")),
        )
        agg.trials += 1
        agg.passes += int(trial.passed)
        agg.steps_total += trial.steps
        agg.latency_total_ms += trial.latency_ms
        agg.cost_usd += trial.cost_usd
        agg.input_tokens += trial.input_tokens
        agg.output_tokens += trial.output_tokens
        if not trial.passed:
            agg.failure_modes[trial.failure_mode.value] += 1
            failure_modes[trial.failure_mode.value] += 1
    per_category: dict[str, TaskAggregate] = {}
    for agg in per_task.values():
        bucket = per_category.setdefault(agg.category or "uncategorized", TaskAggregate(task_id=""))
        bucket.trials += agg.trials
        bucket.passes += agg.passes
        bucket.steps_total += agg.steps_total
        bucket.latency_total_ms += agg.latency_total_ms
        bucket.cost_usd += agg.cost_usd
    return Summary(
        run_id=str(artifact.get("run_id", "")),
        agent=str(artifact.get("agent", "")),
        label=str(artifact.get("label", "")),
        model=str(artifact.get("model", "")),
        task_source=str(artifact.get("task_source", "")),
        trials_per_task=int(artifact.get("trials_per_task", 1)),
        tasks=len(per_task),
        trials=len(trials),
        passed=sum(int(trial.passed) for trial in trials),
        answer_ok=sum(int(trial.answer_ok) for trial in trials),
        tools_ok=sum(int(trial.tools_ok) for trial in trials),
        steps_total=sum(trial.steps for trial in trials),
        latency_total_ms=sum(trial.latency_ms for trial in trials),
        cost_usd=sum(trial.cost_usd for trial in trials),
        input_tokens=sum(trial.input_tokens for trial in trials),
        output_tokens=sum(trial.output_tokens for trial in trials),
        per_category=per_category,
        per_task=per_task,
        failure_modes=failure_modes,
        prices=dict(artifact.get("prices", {}) or {}),
    )


@dataclass
class TaskDelta:
    """How one task moved between two runs."""

    task_id: str
    category: str
    base_passes: int
    base_trials: int
    new_passes: int
    new_trials: int
    status: str

    @property
    def base_rate(self) -> float:
        """Base pass rate."""
        return self.base_passes / self.base_trials if self.base_trials else 0.0

    @property
    def new_rate(self) -> float:
        """New pass rate."""
        return self.new_passes / self.new_trials if self.new_trials else 0.0

    @property
    def base(self) -> str:
        """Base passes over base trials, e.g. ``3/5``."""
        return f"{self.base_passes}/{self.base_trials}"

    @property
    def new(self) -> str:
        """New passes over new trials, e.g. ``2/5``."""
        return f"{self.new_passes}/{self.new_trials}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "task_id": self.task_id,
            "category": self.category,
            "base": f"{self.base_passes}/{self.base_trials}",
            "new": f"{self.new_passes}/{self.new_trials}",
            "base_rate": round(self.base_rate, 4),
            "new_rate": round(self.new_rate, 4),
            "status": self.status,
        }


@dataclass
class Comparison:
    """The result of diffing a baseline run against a new run."""

    base_run_id: str
    new_run_id: str
    base_agent: str
    new_agent: str
    deltas: list[TaskDelta]
    metric_deltas: dict[str, float]
    only_in_base: list[str]
    only_in_new: list[str]

    @property
    def regressed(self) -> list[TaskDelta]:
        """Tasks that got worse."""
        return [delta for delta in self.deltas if delta.status == "regressed"]

    @property
    def fixed(self) -> list[TaskDelta]:
        """Tasks that got better."""
        return [delta for delta in self.deltas if delta.status == "fixed"]

    @property
    def unchanged(self) -> list[TaskDelta]:
        """Tasks that held their pass rate."""
        return [delta for delta in self.deltas if delta.status == "unchanged"]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "base_run_id": self.base_run_id,
            "new_run_id": self.new_run_id,
            "base_agent": self.base_agent,
            "new_agent": self.new_agent,
            "regressed": [delta.to_dict() for delta in self.regressed],
            "fixed": [delta.to_dict() for delta in self.fixed],
            "unchanged_count": len(self.unchanged),
            "metric_deltas": {key: round(value, 6) for key, value in self.metric_deltas.items()},
            "only_in_base": self.only_in_base,
            "only_in_new": self.only_in_new,
        }


def _classify_delta(base_passes: int, base_trials: int, new_passes: int, new_trials: int) -> str:
    """Return fixed | regressed | unchanged from two pass counts.

    A full flip of the same size in both directions (e.g. 1/2 -> 2/2 vs 2/2 ->
    1/2) is treated as movement, so partial flips are visible with repeated
    trials; statistical significance is layered on top in stage 6.
    """
    base_rate = base_passes / base_trials if base_trials else 0.0
    new_rate = new_passes / new_trials if new_trials else 0.0
    if new_rate > base_rate:
        return "fixed"
    if new_rate < base_rate:
        return "regressed"
    return "unchanged"


def compare(base: dict[str, Any], new: dict[str, Any]) -> Comparison:
    """Diff two run artifacts task by task plus the headline metrics."""
    base_summary = summarize(base)
    new_summary = summarize(new)
    task_ids = list(dict.fromkeys([*base_summary.per_task, *new_summary.per_task]))
    deltas: list[TaskDelta] = []
    for task_id in task_ids:
        base_agg = base_summary.per_task.get(task_id)
        new_agg = new_summary.per_task.get(task_id)
        if base_agg is None or new_agg is None:
            continue
        deltas.append(
            TaskDelta(
                task_id=task_id,
                category=new_agg.category or base_agg.category,
                base_passes=base_agg.passes,
                base_trials=base_agg.trials,
                new_passes=new_agg.passes,
                new_trials=new_agg.trials,
                status=_classify_delta(
                    base_agg.passes, base_agg.trials, new_agg.passes, new_agg.trials
                ),
            )
        )
    metric_deltas = {
        "success_rate": new_summary.success_rate - base_summary.success_rate,
        "answer_accuracy": new_summary.answer_accuracy - base_summary.answer_accuracy,
        "tool_accuracy": new_summary.tool_accuracy - base_summary.tool_accuracy,
        "avg_steps": new_summary.avg_steps - base_summary.avg_steps,
        "avg_latency_ms": new_summary.avg_latency_ms - base_summary.avg_latency_ms,
        "cost_usd": new_summary.total_cost_usd - base_summary.total_cost_usd,
    }
    return Comparison(
        base_run_id=base_summary.run_id,
        new_run_id=new_summary.run_id,
        base_agent=base_summary.agent,
        new_agent=new_summary.agent,
        deltas=deltas,
        metric_deltas=metric_deltas,
        only_in_base=sorted(set(base_summary.per_task) - set(new_summary.per_task)),
        only_in_new=sorted(set(new_summary.per_task) - set(base_summary.per_task)),
    )


def render_summary(summary: Summary) -> str:
    """Return the human-readable report for one run."""
    lines = [
        f"run            {summary.run_id}",
        f"agent          {summary.agent}  (model: {summary.model or 'n/a'})",
        f"label          {summary.label or '-'}",
        f"tasks          {summary.tasks} x {summary.trials_per_task} trial(s) = {summary.trials}",
        f"success rate   {summary.success_rate:.1%}  ({summary.passed}/{summary.trials})",
        f"answer acc     {summary.answer_accuracy:.1%}",
        f"tool accuracy  {summary.tool_accuracy:.1%}",
        f"avg steps      {summary.avg_steps:.2f}",
        f"avg latency    {summary.avg_latency_ms:.1f} ms",
        f"cost           ${summary.total_cost_usd:.4f}"
        f"  ({summary.input_tokens} in / {summary.output_tokens} out tokens,"
        f" @{summary.prices.get('input_per_mtok', 0)}/M in,"
        f" @{summary.prices.get('output_per_mtok', 0)}/M out,"
        f" {summary.prices.get('source', 'n/a')})",
        "",
        "by category",
    ]
    for category, agg in sorted(summary.per_category.items()):
        lines.append(
            f"  {category:<14} {agg.passes}/{agg.trials} passed ({agg.pass_rate:.0%})"
            f"  avg steps {agg.avg_steps:.2f}"
        )
    lines.append("")
    lines.append("failure modes")
    if summary.failure_modes:
        for mode, count in summary.failure_modes.most_common():
            lines.append(f"  {mode:<14} {count}")
    else:
        lines.append("  none")
    lines.append("")
    lines.append("per task")
    lines.append(f"  {'task':<26} {'pass':>6} {'steps':>6} {'ms':>8}  failures")
    for task_id, agg in sorted(summary.per_task.items()):
        modes = ", ".join(f"{mode}x{count}" for mode, count in agg.failure_modes.most_common())
        lines.append(
            f"  {task_id:<26} {agg.passes}/{agg.trials:<4} {agg.avg_steps:>6.2f}"
            f" {agg.avg_latency_ms:>8.2f}  {modes}"
        )
    return "\n".join(lines)


def render_comparison(comparison: Comparison) -> str:
    """Return the human-readable diff of two runs."""
    lines = [
        f"base  {comparison.base_run_id}  ({comparison.base_agent})",
        f"new   {comparison.new_run_id}  ({comparison.new_agent})",
        "",
        "metric deltas (new - base)",
    ]
    for key, value in comparison.metric_deltas.items():
        shown = f"{value:+.2f}" if key in ("avg_steps", "avg_latency_ms") else f"{value:+.4f}"
        if key in ("success_rate", "answer_accuracy", "tool_accuracy"):
            shown = f"{value * 100:+.1f} pp"
        if key == "cost_usd":
            shown = f"{value:+.6f} USD"
        lines.append(f"  {key:<16} {shown}")
    lines.append("")
    lines.append(f"fixed      {len(comparison.fixed)}")
    for delta in comparison.fixed:
        lines.append(f"  + {delta.task_id:<24} {delta.base} -> {delta.new}")
    lines.append(f"regressed  {len(comparison.regressed)}")
    for delta in comparison.regressed:
        lines.append(f"  - {delta.task_id:<24} {delta.base} -> {delta.new}")
    lines.append(f"unchanged  {len(comparison.unchanged)}")
    if comparison.only_in_base or comparison.only_in_new:
        lines.append("")
        if comparison.only_in_base:
            lines.append(f"only in base: {', '.join(comparison.only_in_base)}")
        if comparison.only_in_new:
            lines.append(f"only in new:  {', '.join(comparison.only_in_new)}")
    return "\n".join(lines)


def load_summary(path: Any, task_categories: dict[str, str] | None = None) -> Summary:
    """Load a run file from disk and summarize it."""
    return summarize(load_run(path), task_categories)


def failure_mode_counts(trials: list[TrialResult]) -> Counter[str]:
    """Count failure modes across trials (used by tests and the viewer)."""
    return Counter(
        trial.failure_mode.value for trial in trials if trial.failure_mode is not FailureMode.OK
    )
