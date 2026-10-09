"""Command-line interface: run, report, compare, validate.

`compare --fail-on-regression` is the CI gate: it exits 1 when any task got
worse, so a pull request that degrades the agent cannot merge green.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agenteval.agents import SYSTEM_PROMPT_V1, SYSTEM_PROMPT_V2, AnthropicAgent, MockAgent
from agenteval.judge import AnthropicJudge
from agenteval.pricing import PriceTable
from agenteval.reporting import compare, render_comparison, render_summary, summarize
from agenteval.runner import load_run, run_suite, save_run
from agenteval.taskset import (
    classify_directory,
    discover_task_files,
    load_tasks,
    validate_task_files,
)

__all__ = ["build_parser", "main"]

SYSTEM_PRESETS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT_V2}


def _slug(text: str) -> str:
    """Make a string safe to use in a file name."""
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in text)


def build_agent(spec: str, args: argparse.Namespace) -> Any:
    """Construct an agent from ``mock:v1``, ``mock:v2`` or ``anthropic[:model]``."""
    if spec.startswith("mock:"):
        return MockAgent(spec.split(":", 1)[1])
    if spec == "anthropic" or spec.startswith("anthropic:"):
        model = spec.split(":", 1)[1] if ":" in spec else None
        return AnthropicAgent(
            model=model,
            system=SYSTEM_PRESETS[args.system],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    raise SystemExit(f"unknown agent {spec!r}; use mock:v1, mock:v2 or anthropic[:model]")


def _task_paths(values: list[str]) -> list[Path]:
    """Expand CLI task arguments into a list of JSON paths."""
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(discover_task_files(path))
        else:
            paths.append(path)
    return paths


def build_judge(args: argparse.Namespace) -> AnthropicJudge | None:
    """Construct the judge when ``--judge`` was passed, or explain why it cannot run."""
    if not args.judge:
        return None
    try:
        return AnthropicJudge(model=args.judge_model)
    except RuntimeError as exc:
        raise SystemExit(f"--judge needs a model: {exc}") from exc


def cmd_run(args: argparse.Namespace) -> int:
    """Run a suite and write a JSON artifact."""
    paths = _task_paths(args.tasks)
    tasks = load_tasks(paths)
    agent = build_agent(args.agent, args)
    model = getattr(agent, "model", "")
    prices = PriceTable.for_model(
        model, input_price=args.input_price, output_price=args.output_price
    )
    judge = build_judge(args)
    judge_tasks = sum(1 for task in tasks if task.match.value == "judge")
    if judge_tasks and judge is None:
        print(
            f"note: {judge_tasks} task(s) are judge-scored and will fail without --judge",
            file=sys.stderr,
        )
    artifact = run_suite(
        tasks,
        agent,
        trials=args.trials,
        prices=prices,
        judge=judge,
        task_source=",".join(str(path) for path in paths),
        label=args.label,
        config={
            "system_preset": args.system if model else "",
            "max_tokens": args.max_tokens if model else "",
            "temperature": args.temperature if model else "",
            "judge": judge.name if judge is not None else "",
        },
    )
    default_name = f"{artifact['run_id']}-{_slug(args.agent)}.json"
    destination = Path(args.out) if args.out else Path("runs") / default_name
    save_run(artifact, destination)
    summary = summarize(artifact)
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
        print(f"saved run -> {destination}", file=sys.stderr)
    else:
        print(render_summary(summary))
        print(f"\nsaved run -> {destination}")
    if judge is not None:
        note = (
            f"judge: {judge.calls} calls, {judge.input_tokens} in / {judge.output_tokens} out"
            f" tokens, ${judge.cost_usd:.4f} (not included in the run cost above)"
        )
        print(note, file=sys.stderr if args.json else sys.stdout)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Print the summary of a saved run."""
    artifact = load_run(args.run)
    summary = summarize(artifact)
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(render_summary(summary))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Diff two runs; exit 1 on regression when asked to gate CI."""
    base = load_run(args.base)
    new = load_run(args.new)
    result = compare(base, new, alpha=args.alpha)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(render_comparison(result))
    gating = result.gating_regressions
    if gating and args.fail_on_regression:
        names = ", ".join(delta.task_id for delta in gating)
        print(
            f"\nFAIL: {len(gating)} regression(s) beyond noise (alpha={args.alpha}): {names}",
            file=sys.stderr,
        )
        return 1
    if gating:
        print(f"\nwarning: {len(gating)} regression(s), not gating", file=sys.stderr)
    if result.noise_regressions and not args.json:
        names = ", ".join(delta.task_id for delta in result.noise_regressions)
        print(f"\nwithin noise: {names}", file=sys.stderr)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Check task files without running anything."""
    paths = _task_paths(args.tasks)
    problems = validate_task_files(paths)
    if problems:
        for problem in problems:
            print(f"invalid: {problem}", file=sys.stderr)
        return 1
    tasks = load_tasks(paths)
    categories: dict[str, int] = {}
    for task in tasks:
        categories[task.category] = categories.get(task.category, 0) + 1
    for value in args.tasks:
        directory = Path(value)
        if directory.is_dir():
            ignored = classify_directory(directory)[1]
            if ignored:
                print("not task suites (left alone): " + ", ".join(p.name for p in ignored))
    print(f"ok: {len(tasks)} tasks in {len(paths)} file(s); categories: {categories}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ``agenteval`` command."""
    parser = argparse.ArgumentParser(
        prog="agenteval",
        description="Evaluate and regression-test LLM tool-using agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run an agent against a task suite")
    run.add_argument(
        "--tasks",
        nargs="+",
        default=["tasks/basic.json"],
        help="task JSON file(s) or a directory",
    )
    run.add_argument("--agent", default="mock:v1", help="mock:v1 | mock:v2 | anthropic[:model]")
    run.add_argument("--trials", type=int, default=1, help="repetitions per task (flakiness)")
    run.add_argument("--out", help="where to write the run JSON (default runs/<id>-<agent>.json)")
    run.add_argument("--label", default="", help="free-form label stored in the artifact")
    run.add_argument(
        "--system",
        choices=sorted(SYSTEM_PRESETS),
        default="v1",
        help="system prompt preset (real agent)",
    )
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument(
        "--judge",
        action="store_true",
        help="grade match:judge tasks with an LLM judge (needs AGENTEVAL_MODEL + API key)",
    )
    run.add_argument("--judge-model", help="model for the judge (default: AGENTEVAL_MODEL)")
    run.add_argument("--input-price", type=float, help="USD per million input tokens")
    run.add_argument("--output-price", type=float, help="USD per million output tokens")
    run.add_argument("--json", action="store_true", help="print the summary as JSON")
    run.set_defaults(func=cmd_run)

    report = subparsers.add_parser("report", help="print a saved run's summary")
    report.add_argument("run")
    report.add_argument("--json", action="store_true")
    report.set_defaults(func=cmd_report)

    gate = subparsers.add_parser("compare", help="diff two runs")
    gate.add_argument("base", help="baseline run JSON")
    gate.add_argument("new", help="new run JSON")
    gate.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 if any task got worse than the baseline beyond noise",
    )
    gate.add_argument(
        "--alpha", type=float, default=0.05, help="significance level for regression gating"
    )
    gate.add_argument("--json", action="store_true")
    gate.set_defaults(func=cmd_compare)

    validate = subparsers.add_parser("validate", help="check task files")
    validate.add_argument("tasks", nargs="+", default=["tasks"])
    validate.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
