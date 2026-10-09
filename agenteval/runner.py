"""Run a task suite with an agent over N trials and persist a JSON artifact.

The artifact stores per-trial results and the run configuration; aggregate
numbers live in ``agenteval.reporting`` so a report is always recomputed from
the trials instead of a stale copy that was written at run time.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agenteval.models import AgentResult, Task, TrialResult
from agenteval.pricing import PriceTable
from agenteval.scoring import score_trial
from agenteval.tools import ToolBox

__all__ = ["SCHEMA_VERSION", "load_run", "run_suite", "run_trials", "save_run"]

SCHEMA_VERSION = 1

#: judge(task, answer) -> (verdict, rationale)
JudgeFn = Callable[[Task, str], tuple[str, str]]


def new_run_id() -> str:
    """Return a sortable, collision-resistant run id."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _crashed_result(exc: Exception) -> AgentResult:
    """Turn an agent exception into a recorded error, per the harness contract."""
    return AgentResult(error=f"{type(exc).__name__}: {exc}", finish_reason="error")


def _judge_verdict(judge: JudgeFn | None, task: Task, answer: str) -> tuple[str, str]:
    """Ask the judge only for judge-scored tasks, and never crash on it."""
    if judge is None or task.match.value != "judge":
        return "", ""
    try:
        verdict, rationale = judge(task, answer)
    except Exception as exc:
        return "ERROR", f"judge failed: {type(exc).__name__}: {exc}"
    return verdict, rationale


def run_trials(
    tasks: Sequence[Task],
    agent: Any,
    *,
    trials: int = 1,
    prices: PriceTable | None = None,
    judge: JudgeFn | None = None,
    knowledge_base: Sequence[dict[str, Any]] | None = None,
) -> list[TrialResult]:
    """Run every task ``trials`` times and score each run.

    Args:
        tasks: suite to evaluate.
        agent: an object with ``name`` and ``run(task, toolbox)``.
        trials: repetitions per task, used for flakiness measurement.
        prices: token pricing for the cost column.
        judge: optional callable used for ``match: judge`` tasks.
        knowledge_base: overrides the toolbox KB, for custom fixtures.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    price_table = prices if prices is not None else PriceTable.for_model(
        getattr(agent, "model", "")
    )
    results: list[TrialResult] = []
    for trial_index in range(trials):
        for task in tasks:
            toolbox = ToolBox(knowledge_base=knowledge_base)
            started = time.perf_counter()
            try:
                agent_result = agent.run(task, toolbox)
            except Exception as exc:
                agent_result = _crashed_result(exc)
            latency_ms = (time.perf_counter() - started) * 1000.0
            verdict, rationale = _judge_verdict(judge, task, agent_result.answer)
            answer_ok, tools_ok, passed, mode, forbidden, missing, _detail = score_trial(
                task, agent_result, verdict=verdict or None
            )
            results.append(
                TrialResult(
                    task_id=task.id,
                    trial=trial_index,
                    agent=str(getattr(agent, "name", agent.__class__.__name__)),
                    result=agent_result,
                    passed=passed,
                    answer_ok=answer_ok,
                    tools_ok=tools_ok,
                    forbidden_used=forbidden,
                    missing_required=missing,
                    failure_mode=mode,
                    steps=agent_result.steps,
                    latency_ms=latency_ms,
                    input_tokens=agent_result.input_tokens,
                    output_tokens=agent_result.output_tokens,
                    cost_usd=price_table.cost_usd(
                        agent_result.input_tokens, agent_result.output_tokens
                    ),
                    judge_verdict=verdict,
                    judge_rationale=rationale,
                )
            )
    return results


def run_suite(
    tasks: Sequence[Task],
    agent: Any,
    *,
    trials: int = 1,
    prices: PriceTable | None = None,
    judge: JudgeFn | None = None,
    task_source: str = "",
    label: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a suite and return the JSON-ready artifact (nothing written to disk)."""
    started_at = datetime.now(UTC)
    price_table = prices if prices is not None else PriceTable.for_model(
        getattr(agent, "model", "")
    )
    trials_results = run_trials(tasks, agent, trials=trials, prices=price_table, judge=judge)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": new_run_id(),
        "agent": str(getattr(agent, "name", agent.__class__.__name__)),
        "model": str(getattr(agent, "model", "")),
        "label": label,
        "task_source": task_source,
        "trials_per_task": trials,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": dict(config or {}),
        "prices": price_table.to_dict(),
        "task_ids": [task.id for task in tasks],
        "results": [trial.to_dict() for trial in trials_results],
    }


def save_run(artifact: Iterable[dict[str, Any]] | dict[str, Any], path: Path | str) -> Path:
    """Write a run artifact as pretty JSON and return the path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    is_list = isinstance(artifact, (list, tuple))
    payload: Any = list(artifact) if is_list else artifact
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return destination


def load_run(path: Path | str) -> dict[str, Any]:
    """Read a run artifact written by :func:`save_run`."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    if not isinstance(artifact, dict) or "results" not in artifact:
        raise ValueError(f"{source} is not an agenteval run artifact (missing 'results')")
    version = artifact.get("schema_version", 0)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{source} uses schema version {version}; this build reads <= {SCHEMA_VERSION}"
        )
    return artifact


def trials_from_artifact(artifact: dict[str, Any]) -> list[TrialResult]:
    """Rebuild scored trial objects from a loaded run artifact."""
    return [TrialResult.from_dict(payload) for payload in artifact.get("results", [])]
