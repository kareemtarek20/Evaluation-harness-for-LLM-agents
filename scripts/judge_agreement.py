"""Collect LLM-judge verdicts for the label sheet and score them against humans.

Two subcommands, because the two halves run at different times::

    python scripts/judge_agreement.py collect --dry-run   # offline: show the request
    python scripts/judge_agreement.py collect             # needs ANTHROPIC_API_KEY
    python scripts/judge_agreement.py score               # needs human_label filled in

``collect`` writes ``runs/judge_verdicts.json`` and never touches the label
sheet, so the human answers stay the source of truth. ``score`` reports raw
agreement, Cohen's kappa (which subtracts chance agreement), the confusion
matrix and a per-probe breakdown: probes are paired on purpose (same fact,
short vs verbose) so a verbosity or hedging bias shows up as a split pair.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agenteval.agents import MODEL_ENV_VAR  # noqa: E402
from agenteval.judge import (  # noqa: E402
    AnthropicJudge,
    agreement_report,
    build_judge_prompt,
    normalize_label,
)
from agenteval.models import Task  # noqa: E402

SHEET = ROOT / "tasks" / "judge_labels.json"
VERDICTS = ROOT / "runs" / "judge_verdicts.json"

#: Landis & Koch agreement bands, highest floor first.
BANDS = ((0.81, "almost perfect"), (0.61, "substantial"), (0.41, "moderate"), (0.21, "fair"))


def interpret(kappa: float) -> str:
    """Return the Landis & Koch band a kappa value falls in."""
    if kappa < 0.0:
        return "negative (the judge disagrees more than chance)"
    for floor, name in BANDS:
        if kappa >= floor:
            return name
    return "slight"


def load_json(path: Path, hint: str) -> dict[str, Any]:
    """Read a JSON object, or exit with the command that would create it."""
    if not path.exists():
        raise SystemExit(f"missing {path}: {hint}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} is not a judgement object")
    return payload


def task_for(entry: dict[str, Any]) -> Task:
    """Rebuild a grading task from one label-sheet entry."""
    return Task.from_dict(
        {
            "id": entry["task_id"],
            "prompt": entry["prompt"],
            "expected": "any",
            "match": "judge",
            "rubric": entry["rubric"],
            "category": entry.get("category", "general"),
        }
    )


def prompt_for(entry: dict[str, Any]) -> str:
    """Return the exact grading message one entry would send."""
    return build_judge_prompt(task_for(entry), str(entry["answer"]))


def grade_one(judge: AnthropicJudge, entry: dict[str, Any]) -> dict[str, Any]:
    """Grade one entry, recording a judge failure as an error instead of raising."""
    try:
        verdict, rationale = judge.judge(task_for(entry), str(entry["answer"]))
        error = ""
    except Exception as exc:
        verdict, rationale, error = "", "", f"{type(exc).__name__}: {exc}"
    return {
        "id": entry["id"],
        "task_id": entry["task_id"],
        "verdict": verdict,
        "rationale": rationale,
        "error": error,
    }


def cmd_collect(args: argparse.Namespace) -> int:
    """Ask the judge for a verdict on every entry and store them separately."""
    entries = load_json(SHEET, "run `python scripts/make_judge_labels.py` first")["entries"]
    if args.dry_run:
        print(f"entries to grade : {len(entries)}")
        print(f"model          : {args.model or os.environ.get(MODEL_ENV_VAR, 'unset')}")
        print(f"\n--- request for {entries[0]['id']} ---\n{prompt_for(entries[0])}\n--- end ---")
        print("no API call made (--dry-run)")
        return 0
    judge = AnthropicJudge(model=args.model)
    verdicts = []
    for entry in entries:
        row = grade_one(judge, entry)
        verdicts.append(row)
        shown = row["verdict"] or "ERROR"
        print(f"{row['id']:<6} {row['task_id']:<24} {shown}  {row['error'] or row['rationale']}")
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "judge": judge.name,
        "sheet": str(SHEET.relative_to(ROOT)),
        "calls": judge.calls,
        "input_tokens": judge.input_tokens,
        "output_tokens": judge.output_tokens,
        "cost_usd": round(judge.cost_usd, 6),
        "verdicts": verdicts,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"\nwrote {destination}: {judge.calls} calls, ${judge.cost_usd:.4f}")
    return 0


def merge(sheet: dict[str, Any], verdicts: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach the stored judge verdict for each sheet entry."""
    by_id = {str(row["id"]): row for row in verdicts.get("verdicts", [])}
    merged: list[dict[str, Any]] = []
    for entry in sheet["entries"]:
        row = dict(entry)
        row["judge_verdict"] = by_id.get(str(entry["id"]), {}).get("verdict", "")
        merged.append(row)
    return merged


def disagreements(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Entries where a filled human label and a filled verdict differ."""
    out: list[dict[str, Any]] = []
    for row in rows:
        human = str(row.get("human_label", "")).strip()
        judge = str(row.get("judge_verdict", "")).strip()
        if not human or not judge:
            continue
        if normalize_label(human) != normalize_label(judge):
            out.append(row)
    return out


def cmd_score(args: argparse.Namespace) -> int:
    """Compare human labels with the stored judge verdicts."""
    sheet = load_json(SHEET, "run `python scripts/make_judge_labels.py` first")
    verdicts = load_json(
        Path(args.verdicts), "run `python scripts/judge_agreement.py collect` first"
    )
    rows = merge(sheet, verdicts)
    report = agreement_report(rows)
    unlabeled = sum(
        int(not str(entry.get("human_label", "")).strip()) for entry in sheet["entries"]
    )
    if args.json:
        print(json.dumps({**report.to_dict(), "unlabeled": unlabeled}, indent=2))
        return 0 if report.labeled else 1
    print(f"sheet entries    : {len(rows)}")
    print(f"labeled by human : {report.labeled}  (unlabeled: {unlabeled})")
    print(f"graded by judge  : {len(verdicts.get('verdicts', []))}")
    if report.labeled == 0:
        print("\nnothing to score yet: fill in human_label in tasks/judge_labels.json")
        return 1
    print(f"agreement        : {report.agreement:.1%} ({report.agreed}/{report.labeled})")
    print(f"cohen's kappa    : {report.kappa:.3f}  ({interpret(report.kappa)})")
    print("\nconfusion matrix")
    for key, count in sorted(report.confusion.items()):
        print(f"  {key:<28} {count}")
    print("\nby probe (a split pair is evidence of that bias)")
    for probe, stats in report.by_probe.items():
        print(
            f"  {probe:<14} {int(stats['labeled']):>3} labeled,"
            f" agreement {stats['agreement']:.0%}, kappa {stats['kappa']:.3f}"
        )
    rows_apart = disagreements(rows)
    if rows_apart:
        print("\ndisagreements")
        for row in rows_apart:
            print(
                f"  {row['id']:<6} {row['task_id']:<24} human={row['human_label']:<5}"
                f" judge={row['judge_verdict']}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for this script."""
    parser = argparse.ArgumentParser(prog="judge_agreement", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="grade every entry with the judge")
    collect.add_argument("--model", help="override AGENTEVAL_MODEL for this run")
    collect.add_argument("--out", default=str(VERDICTS), help="where to write verdicts")
    collect.add_argument(
        "--dry-run", action="store_true", help="print the request contract, call nothing"
    )
    collect.set_defaults(func=cmd_collect)
    score = subparsers.add_parser("score", help="compare labels with stored verdicts")
    score.add_argument("--verdicts", default=str(VERDICTS), help="verdicts file to compare")
    score.add_argument("--json", action="store_true")
    score.set_defaults(func=cmd_score)
    return parser


def ready_for_live_calls() -> tuple[bool, str]:
    """Return whether the env is configured enough to call the API."""
    missing = []
    if not os.environ.get(MODEL_ENV_VAR, "").strip():
        missing.append(MODEL_ENV_VAR)
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        return False, f"set {', '.join(missing)} first"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.command == "collect" and not args.dry_run:
        ok, reason = ready_for_live_calls()
        if not ok:
            print(
                f"refusing to collect: {reason}.\n"
                "Use --dry-run to inspect the request without an API key.",
                file=sys.stderr,
            )
            return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
