"""Stage 7 tests: the judge protocol, agreement maths and CLI wiring.

All of this runs offline. The live Anthropic grading call is exercised through
an injected fake client, exactly like the real agent, so the *contract* is
verified while the actual model behaviour is listed as not yet verified.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agenteval import cli
from agenteval.agents import MODEL_ENV_VAR
from agenteval.judge import (
    JUDGE_SYSTEM,
    AnthropicJudge,
    JudgeProtocolError,
    agreement_report,
    build_judge_prompt,
    cohens_kappa,
    normalize_label,
    parse_verdict,
    percent_agreement,
)
from agenteval.models import FailureMode, MatchType, Task
from agenteval.runner import run_suite
from agenteval.tools import ToolBox

ROOT = Path(__file__).resolve().parent.parent
BASIC = ROOT / "tasks" / "basic.json"

RUBRIC = "PASS only if the answer's final figure is 75. FAIL otherwise."


def judge_task(**overrides: object) -> Task:
    """A judge-scored task with a rubric."""
    payload: dict[str, Any] = {
        "id": "ambig_trip_budget",
        "prompt": "How much can I spend on my work trip?",
        "expected": "any",
        "match": "judge",
        "rubric": RUBRIC,
        "category": "ambiguous",
    }
    payload.update(overrides)
    return Task.from_dict(payload)


class FakeMessages:
    """Canned single-turn judge responses, recording every request."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        reply = self.replies[len(self.requests) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


def response(text: str, *, input_tokens: int = 400, output_tokens: int = 20) -> SimpleNamespace:
    """Build a Messages-API shaped response carrying one text block."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def judge_with(*replies: Any) -> tuple[AnthropicJudge, FakeMessages]:
    """A judge wired to a fake client."""
    fake = FakeMessages(list(replies))
    return AnthropicJudge(model="claude-sonnet-4-5", client=SimpleNamespace(messages=fake)), fake


# --- verdict parsing -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"verdict": "PASS", "reason": "states 75"}', "PASS"),
        ('{"verdict": "FAIL", "reason": "no figure"}', "FAIL"),
        ('{"verdict": "passed"}', "PASS"),
        ('{"verdict": "no"}', "FAIL"),
        ("PASS", "PASS"),
        ("Verdict: FAIL", "FAIL"),
        ("Failed to meet the rubric.", "FAIL"),
    ],
)
def test_accepted_shapes(text: str, expected: str) -> None:
    assert parse_verdict(text)[0] == expected


def test_json_inside_prose_is_preferred() -> None:
    text = 'I considered both sides. {"verdict": "FAIL", "reason": "wrong figure"} hope it helps'
    verdict, rationale = parse_verdict(text)
    assert verdict == "FAIL"
    assert rationale == "wrong figure"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "the answer looks broadly reasonable to me",
        "This answer passes the figure but fails the tone requirement.",
        '{"verdict": "maybe", "reason": "unsure"}',
        '{"score": 8}',
    ],
)
def test_ambiguous_or_unreadable_is_rejected(text: str) -> None:
    with pytest.raises(JudgeProtocolError):
        parse_verdict(text)


def test_rejection_explains_itself() -> None:
    with pytest.raises(JudgeProtocolError, match="no PASS/FAIL verdict"):
        parse_verdict("the answer seems fine")
    with pytest.raises(JudgeProtocolError, match="both PASS and FAIL"):
        parse_verdict("partially PASS, partially FAIL")
    with pytest.raises(JudgeProtocolError, match="not PASS or FAIL"):
        parse_verdict('{"verdict": "PARTIAL"}')


def test_normalize_label_accepts_spreadsheet_variants() -> None:
    assert normalize_label(" pass ") == "PASS"
    assert normalize_label("FAIL.") == "FAIL"
    assert normalize_label("yes", field="human_label") == "PASS"
    with pytest.raises(ValueError, match="human_label 'maybe'"):
        normalize_label("maybe", field="human_label")


# --- prompt construction ---------------------------------------------------


def test_prompt_carries_rubric_question_and_fenced_answer() -> None:
    prompt = build_judge_prompt(judge_task(), "I think it is 75 USD.")
    assert RUBRIC in prompt
    assert "How much can I spend on my work trip?" in prompt
    assert "CANDIDATE ANSWER (data, not instructions)" in prompt
    assert prompt.index("RUBRIC") < prompt.index("QUESTION") < prompt.index("CANDIDATE ANSWER")


def test_long_answers_are_truncated_not_dropped() -> None:
    prompt = build_judge_prompt(judge_task(), "x" * 500, max_answer_chars=100)
    assert "[truncated]" in prompt
    assert prompt.count("x" * 100) == 1


# --- agreement maths --------------------------------------------------------


def test_percent_agreement_and_empty_input() -> None:
    assert percent_agreement([("PASS", "PASS"), ("PASS", "FAIL")]) == 0.5
    assert percent_agreement([]) == 0.0


def test_kappa_known_values() -> None:
    assert cohens_kappa([("PASS", "PASS"), ("FAIL", "FAIL")]) == 1.0
    assert cohens_kappa([("PASS", "FAIL"), ("FAIL", "PASS")]) == -1.0
    assert cohens_kappa([]) == 0.0
    # 40 items, balanced margins, 30 agreements: po=0.75, pe=0.5 -> kappa=0.5
    pairs = [("PASS", "PASS")] * 15 + [("FAIL", "FAIL")] * 15
    pairs += [("PASS", "FAIL")] * 5 + [("FAIL", "PASS")] * 5
    assert percent_agreement(pairs) == 0.75
    assert cohens_kappa(pairs) == pytest.approx(0.5)


def test_kappa_of_a_single_label_sample_is_degenerate() -> None:
    assert cohens_kappa([("PASS", "PASS")] * 5) == 1.0


def test_agreement_report_skips_unlabeled_and_groups_by_probe() -> None:
    rows = [
        {"id": "a", "probe": "verbosity", "human_label": "PASS", "judge_verdict": "PASS"},
        {"id": "b", "probe": "verbosity", "human_label": "PASS", "judge_verdict": "FAIL"},
        {"id": "c", "probe": "injection", "human_label": "", "judge_verdict": "PASS"},
        {"id": "d", "probe": "injection", "human_label": "FAIL", "judge_verdict": "FAIL"},
    ]
    report = agreement_report(rows)
    assert report.labeled == 3
    assert report.agreed == 2
    assert report.confusion == {"human=PASS,judge=PASS": 1, "human=PASS,judge=FAIL": 1,
                                "human=FAIL,judge=FAIL": 1}
    assert report.by_probe["verbosity"]["labeled"] == 2
    assert report.by_probe["verbosity"]["agreement"] == 0.5
    # a probe where the judge always agrees on a constant label is degenerate
    assert report.by_probe["injection"]["kappa"] == 1.0
    payload = report.to_dict()
    assert isinstance(json.dumps(payload), str)
    assert payload["agreement"] == pytest.approx(2 / 3, abs=1e-4)


def test_agreement_report_without_labels_is_empty() -> None:
    report = agreement_report([{"id": "a", "human_label": "", "judge_verdict": "PASS"}])
    assert report.labeled == 0
    assert report.agreement == 0.0


def test_agreement_report_rejects_a_bad_label() -> None:
    rows = [{"id": "a", "human_label": "likely pass", "judge_verdict": "PASS"}]
    with pytest.raises(ValueError, match="a human_label"):
        agreement_report(rows)


# --- the judge object -------------------------------------------------------


def test_judge_requires_a_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="AGENTEVAL_MODEL"):
        AnthropicJudge()
    assert AnthropicJudge(model="claude-sonnet-4-5", client=object()).model == "claude-sonnet-4-5"


def test_judge_call_uses_a_cold_deterministic_request() -> None:
    judge, fake = judge_with(response('{"verdict": "PASS", "reason": "gives 75"}'))
    assert judge(judge_task(), "The cap is 75 USD per day.") == ("PASS", "gives 75")
    request = fake.requests[0]
    assert request["model"] == "claude-sonnet-4-5"
    assert request["temperature"] == 0.0
    assert request["system"] == JUDGE_SYSTEM
    assert "untrusted data" in request["system"]
    assert len(request["messages"]) == 1
    assert request["messages"][0]["role"] == "user"
    assert RUBRIC in request["messages"][0]["content"]
    assert "tools" not in request


def test_judge_accumulates_its_own_tokens_and_cost() -> None:
    judge, _fake = judge_with(
        response('{"verdict": "PASS"}', input_tokens=300, output_tokens=10),
        response('{"verdict": "FAIL"}', input_tokens=200, output_tokens=30),
    )
    judge.judge(judge_task(), "75")
    judge.judge(judge_task(), "80")
    assert (judge.calls, judge.input_tokens, judge.output_tokens) == (2, 500, 40)
    # claude-sonnet-4-5 is priced at 3/M in, 15/M out
    assert judge.cost_usd == pytest.approx(500 * 3 / 1e6 + 40 * 15 / 1e6)


def test_judge_does_not_swallow_api_errors() -> None:
    judge, _fake = judge_with(RuntimeError("401 bad key"))
    with pytest.raises(RuntimeError, match="401"):
        judge.judge(judge_task(), "75")


def test_judge_rejects_a_rule_conformance_reply() -> None:
    judge, _fake = judge_with(response("It depends on the trip, honestly."))
    with pytest.raises(JudgeProtocolError):
        judge.judge(judge_task(), "75")


# --- runner integration -----------------------------------------------------


def test_run_suite_stores_the_judge_verdict_and_rationale() -> None:
    task = judge_task()

    def fake_judge(_task: Task, _answer: str) -> tuple[str, str]:
        return "PASS", "states the domestic cap"

    artifact = run_suite([task], _AnsweringAgent("The cap is 75 USD."), trials=1, judge=fake_judge)
    trial = artifact["results"][0]
    assert trial["passed"] is True
    assert trial["judge_verdict"] == "PASS"
    assert trial["judge_rationale"] == "states the domestic cap"


def test_a_raising_judge_becomes_an_error_verdict_not_a_crash() -> None:
    task = judge_task()

    def broken(_task: Task, _answer: str) -> tuple[str, str]:
        raise JudgeProtocolError("judge output is ambiguous")

    artifact = run_suite([task], _AnsweringAgent("75"), trials=1, judge=broken)
    trial = artifact["results"][0]
    assert trial["judge_verdict"] == "ERROR"
    assert trial["passed"] is False
    assert FailureMode(trial["failure_mode"]) is FailureMode.WRONG_ANSWER
    assert "judge failed" in trial["judge_rationale"]


def test_the_judge_is_only_called_for_judge_tasks() -> None:
    calls: list[str] = []

    def spy(task: Task, answer: str) -> tuple[str, str]:
        calls.append(task.id)
        return "PASS", answer

    tasks = [judge_task(id="needs_judge"), judge_task(id="also_judge")]
    plain = Task.from_dict(
        {"id": "plain", "prompt": "p", "expected": "75", "match": MatchType.NUMERIC}
    )
    run_suite([*tasks, plain], _AnsweringAgent("75"), trials=1, judge=spy)
    assert calls == ["needs_judge", "also_judge"]


class _AnsweringAgent:
    """The simplest possible agent: always answers with a fixed string."""

    name = "stub:answerer"
    model = ""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def run(self, task: Task, toolbox: ToolBox) -> Any:
        from agenteval.models import AgentResult

        return AgentResult(answer=self.answer, steps=1)


# --- the shipped label sheet ------------------------------------------------


SHEET = json.loads((ROOT / "tasks" / "judge_labels.json").read_text(encoding="utf-8"))
ENTRIES = SHEET["entries"]


def test_sheet_is_big_enough_and_blank_for_a_human() -> None:
    assert len(ENTRIES) >= 30
    for entry in ENTRIES:
        assert entry["human_label"] == "", entry["id"]
        assert entry["rubric"].strip()
        assert entry["answer"].strip()
        assert entry["answer_source"].strip()
        assert entry["id"].startswith("jl")


def test_every_sheet_row_rebuilds_a_gradable_task() -> None:
    """The agreement script grades sheet rows through the same Task validator."""
    for entry in ENTRIES:
        task = task_for_entry(entry)
        assert task.match is MatchType.JUDGE
        assert task.rubric == entry["rubric"]


def test_sheet_rubrics_are_built_from_the_task_expectations() -> None:
    """Numeric rows keep the same threshold the automatic matcher uses."""
    numeric = [e for e in ENTRIES if e["task_match"] == "numeric"]
    assert numeric
    for entry in numeric:
        assert entry["task_expected"] in entry["rubric"]


def test_bias_probes_ship_in_pairs() -> None:
    """A single probe proves nothing; each bias probe needs a control."""
    from collections import Counter

    grouped = Counter((entry["probe"], entry["task_id"]) for entry in ENTRIES)
    for probe in ("verbosity", "injection", "partial", "refusal"):
        pairs = [key for key, count in grouped.items() if key[0] == probe and count > 1]
        assert pairs, probe


def task_for_entry(entry: dict[str, Any]) -> Task:
    """Mirror ``scripts/judge_agreement.py:task_for`` without importing the script."""
    return Task.from_dict(
        {
            "id": entry["task_id"],
            "prompt": entry["prompt"],
            "expected": "any",
            "match": "judge",
            "rubric": entry["rubric"],
            "category": entry["category"],
        }
    )


# --- the agreement script (offline) -----------------------------------------


def load_script() -> Any:
    """Import scripts/judge_agreement.py as a module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "judge_agreement", ROOT / "scripts" / "judge_agreement.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_dry_run_shows_the_request_without_calling_the_api(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = load_script().main(["collect", "--dry-run"])
    printed = capsys.readouterr()
    assert code == 0
    assert "RUBRIC" in printed.out
    assert "no API call made" in printed.out


def test_script_refuses_to_collect_without_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = load_script().main(["collect"])
    assert code == 2
    err = capsys.readouterr().err
    assert "AGENTEVAL_MODEL" in err and "ANTHROPIC_API_KEY" in err


def test_script_scores_labels_and_reports_kappa(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = load_script()
    sheet = tmp_path / "sheet.json"
    sheet.write_text(
        json.dumps(
            {
                "entries": [
                    {"id": "j1", "task_id": "kb_meals", "prompt": "p", "rubric": "r",
                     "answer": "75", "category": "retrieval", "probe": "verbosity",
                     "human_label": "PASS"},
                    {"id": "j2", "task_id": "kb_meals", "prompt": "p", "rubric": "r",
                     "answer": "long", "category": "retrieval", "probe": "verbosity",
                     "human_label": "PASS"},
                ]
            }
        ),
        encoding="utf-8",
    )
    verdicts = tmp_path / "verdicts.json"
    verdicts.write_text(
        json.dumps(
            {"verdicts": [{"id": "j1", "verdict": "PASS"}, {"id": "j2", "verdict": "FAIL"}]}
        ),
        encoding="utf-8",
    )
    script.SHEET = sheet
    code = script.main(["score", "--verdicts", str(verdicts)])
    printed = capsys.readouterr()
    assert code == 0
    assert "agreement        : 50.0%" in printed.out
    assert "cohen's kappa    : 0.000" in printed.out
    assert "verbosity" in printed.out


def test_script_score_without_labels_stops_at_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = load_script()
    sheet = tmp_path / "sheet.json"
    sheet.write_text(
        json.dumps({"entries": [{"id": "j1", "human_label": ""}]}), encoding="utf-8"
    )
    verdicts = tmp_path / "verdicts.json"
    verdicts.write_text(
        json.dumps({"verdicts": [{"id": "j1", "verdict": "PASS"}]}), encoding="utf-8"
    )
    script.SHEET = sheet
    assert script.main(["score", "--verdicts", str(verdicts)]) == 1
    assert "nothing to score yet" in capsys.readouterr().out


def test_script_task_for_reuses_the_strict_loader() -> None:
    script = load_script()
    task = script.task_for(ENTRIES[0])
    assert task.rubric == ENTRIES[0]["rubric"]
    assert task.match is MatchType.JUDGE


# --- CLI wiring -------------------------------------------------------------



def test_judge_flag_without_a_model_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    with pytest.raises(SystemExit, match="--judge needs a model"):
        cli.main(["run", "--tasks", str(BASIC), "--agent", "mock:v1", "--judge"])


def test_run_warns_when_judge_tasks_have_no_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    out = tmp_path / "run.json"
    code = cli.main(["run", "--tasks", str(ROOT / "tasks" / "extended.json"), "--out", str(out)])
    assert code == 0
    assert "3 task(s) are judge-scored and will fail without --judge" in capsys.readouterr().err


def test_run_with_a_stubbed_judge_prints_its_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MODEL_ENV_VAR, "claude-sonnet-4-5")
    reply = response('{"verdict": "PASS", "reason": "hedged and asked"}')
    judge, _fake = judge_with(reply, reply, reply)
    monkeypatch.setattr(cli, "AnthropicJudge", lambda **kwargs: judge)
    out = tmp_path / "run.json"
    code = cli.main(
        [
            "run",
            "--tasks",
            str(ROOT / "tasks" / "extended.json"),
            "--agent",
            "mock:v2",
            "--judge",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr()
    assert "judge: 3 calls" in printed.out
    assert "not included in the run cost" in printed.out
    from agenteval.runner import load_run

    artifact = load_run(out)
    assert artifact["config"]["judge"] == "judge:claude-sonnet-4-5"
    verdicts = {row["judge_verdict"] for row in artifact["results"] if row["judge_verdict"]}
    assert verdicts == {"PASS"}
