"""Stage 5 tests: the CLI end to end, including the CI exit code."""

import json
from pathlib import Path

import pytest

from agenteval.cli import main
from agenteval.runner import load_run

ROOT = Path(__file__).resolve().parent.parent
BASIC = str(ROOT / "tasks" / "basic.json")


def test_run_then_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run.json"
    code = main(
        ["run", "--tasks", BASIC, "--agent", "mock:v1", "--out", str(out), "--label", "cli test"]
    )
    assert code == 0
    assert out.exists()
    printed = capsys.readouterr().out
    assert "success rate" in printed
    assert "cli test" in printed

    artifact = load_run(out)
    assert artifact["agent"] == "mock:v1"
    assert artifact["label"] == "cli test"
    assert len(artifact["results"]) == 10

    code = main(["report", str(out)])
    assert code == 0
    assert "missing_tool" in capsys.readouterr().out

    code = main(["report", str(out), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["tasks"] == 10
    assert payload["passed"] == 1


def test_compare_gates_on_regression(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = tmp_path / "base.json"
    new = tmp_path / "new.json"
    main(["run", "--tasks", BASIC, "--agent", "mock:v1", "--out", str(base)])
    main(["run", "--tasks", BASIC, "--agent", "mock:v2", "--out", str(new)])
    capsys.readouterr()

    assert main(["compare", str(base), str(new)]) == 0
    assert "regressed  1" in capsys.readouterr().out

    code = main(["compare", str(base), str(new), "--fail-on-regression"])
    captured = capsys.readouterr()
    assert code == 1
    assert "FAIL: 1 regression(s): calc_power" in captured.err

    assert main(["compare", str(new), str(new), "--fail-on-regression"]) == 0


def test_compare_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = tmp_path / "base.json"
    new = tmp_path / "new.json"
    main(["run", "--tasks", BASIC, "--agent", "mock:v1", "--out", str(base)])
    main(["run", "--tasks", BASIC, "--agent", "mock:v2", "--out", str(new)])
    capsys.readouterr()
    assert main(["compare", str(base), str(new), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["regressed"][0]["task_id"] == "calc_power"
    assert len(payload["fixed"]) == 9


def test_run_json_and_prices(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run.json"
    main(
        [
            "run", "--tasks", BASIC, "--agent", "mock:v2", "--out", str(out),
            "--json", "--input-price", "1", "--output-price", "2",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["cost_usd"] > 0
    assert payload["prices"]["input_per_mtok"] == 1.0
    assert payload["by_category"]["retrieval"]["passes"] == 4


def test_validate_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(ROOT / "tasks")]) == 0
    assert "56 tasks" in capsys.readouterr().out


def test_validate_rejects_a_bad_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "a"}]), encoding="utf-8")
    assert main(["validate", str(bad)]) == 1
    assert "missing prompt" in capsys.readouterr().err


def test_unknown_agent_is_a_clean_error() -> None:
    with pytest.raises(SystemExit, match="unknown agent"):
        main(["run", "--tasks", BASIC, "--agent", "gpt"])


def test_trials_flag_is_passed_through(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    main(["run", "--tasks", BASIC, "--agent", "mock:v2", "--trials", "3", "--out", str(out)])
    artifact = load_run(out)
    assert artifact["trials_per_task"] == 3
    assert len(artifact["results"]) == 30
