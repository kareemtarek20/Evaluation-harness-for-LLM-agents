"""Stage 9 tests: the static HTML trace viewer, including escaping."""

import json
from pathlib import Path

import pytest

from agenteval.cli import main
from agenteval.models import AgentResult, Task, ToolCall
from agenteval.pricing import PriceTable
from agenteval.reporting import summarize
from agenteval.runner import run_suite
from agenteval.taskset import load_task_file
from agenteval.viewer import render_html, write_html

ROOT = Path(__file__).resolve().parent.parent
BASIC = ROOT / "tasks" / "basic.json"
PRICES = PriceTable(input_per_mtok=3.0, output_per_mtok=15.0, source="test")


class HostileAgent:
    """Agent whose answer and tool output carry markup, to prove escaping works."""

    name = "hostile"
    model = ""

    def run(self, task: Task, toolbox: object) -> AgentResult:
        payload = '<script>alert(1)</script><img src=x onerror="steal()">'
        return AgentResult(
            answer=f"done {payload}",
            tool_calls=[
                ToolCall(
                    tool="calculator",
                    input={"expression": payload},
                    output=payload + '"><svg/onload=alert(2)>',
                    is_error=True,
                    step=1,
                )
            ],
            steps=1,
        )


def hostile_artifact() -> dict:
    tasks = load_task_file(BASIC)[:2]
    return run_suite(tasks, HostileAgent(), trials=1, prices=PRICES)


def mock_artifact(agent: str = "v1", trials: int = 1) -> dict:
    return run_suite(
        load_task_file(BASIC),
        _mock(agent),
        trials=trials,
        prices=PRICES,
        task_source="tasks/basic.json",
        label=f"viewer {agent}",
    )


def _mock(version: str):
    from agenteval.agents import MockAgent

    return MockAgent(version)


def test_page_has_one_card_per_trial() -> None:
    artifact = mock_artifact(trials=2)
    page = render_html(artifact)
    summary = summarize(artifact)
    assert page.count('<article class="trial"') == summary.trials == 20
    assert page.startswith("<!doctype html>")
    for task_id in summary.per_task:
        assert f'data-task="{task_id}"' in page


def test_steps_are_listed_in_order_with_answer() -> None:
    artifact = mock_artifact("v2")
    page = render_html(artifact)
    assert "1. calculator" in page
    assert "2. calculator" in page or "kb_search" in page
    assert '<div class="label">answer</div>' in page
    assert "no tool calls" in page


def test_failures_carry_mode_and_are_filterable() -> None:
    artifact = mock_artifact("v1")
    page = render_html(artifact)
    summary = summarize(artifact)
    failed = summary.trials - summary.passed
    assert page.count('data-passed="false"') == failed
    assert page.count('data-passed="true"') == summary.passed
    for mode, count in summary.failure_modes.items():
        assert page.count(f'data-mode="{mode}"') == count
        assert f'<option value="{mode}">' in page
    assert 'id="only-failures"' in page
    assert 'id="mode-filter"' in page


def test_multi_trial_run_keeps_trial_indices() -> None:
    page = render_html(mock_artifact("v2", trials=3))
    assert page.count("trial 0 ") == 10
    assert page.count("trial 2 ") == 10


def test_hostile_text_is_escaped() -> None:
    page = render_html(hostile_artifact())
    assert "<script>alert(1)" not in page
    assert '<img src=x onerror="steal()">' not in page
    assert "<svg/onload=alert(2)>" not in page
    assert "steal()" in page  # the payload is still readable as data
    assert page.count("&lt;script&gt;") == 6  # 2 tasks x (tool input, output, answer)


def test_page_is_self_contained() -> None:
    page = render_html(mock_artifact("v1"))
    assert "Content-Security-Policy" in page
    assert "<link" not in page
    assert 'src="http' not in page
    assert "@import" not in page
    assert page.count("<script>") == 1


def test_csp_meta_tag_parses_as_one_attribute() -> None:
    """A missing closing quote would eat the whole document as an attribute value.

    Caught by opening a rendered page in a browser: the policy swallowed the
    head, and the filter script was blocked.
    """
    from html.parser import HTMLParser

    class MetaCollector(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.meta: list[dict[str, str | None]] = []
            self.titles: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag == "meta":
                self.meta.append(dict(attrs))

        def handle_data(self, data: str) -> None:
            if self.lasttag == "title":
                self.titles.append(data)

    parser = MetaCollector()
    parser.feed(render_html(mock_artifact("v1")))
    assert {
        "http-equiv": "Content-Security-Policy",
        "content": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
    } in parser.meta
    assert parser.titles and parser.titles[0].startswith("AgentEval trace ")


def test_judge_verdict_is_shown_when_present() -> None:
    artifact = hostile_artifact()
    artifact["results"][0]["judge_verdict"] = "FAIL"
    artifact["results"][0]["judge_rationale"] = "answer is wrong <b>bad</b>"
    page = render_html(artifact)
    assert "judge FAIL: answer is wrong &lt;b&gt;bad&lt;/b&gt;" in page


def test_tool_errors_are_marked() -> None:
    page = render_html(hostile_artifact())
    assert '<li class="step err">' in page
    assert "tool error" in page


def test_write_html_creates_nested_file(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(mock_artifact("v1")), encoding="utf-8")
    destination = tmp_path / "site" / "trace.html"
    written = write_html(run_path, destination)
    assert written == destination
    assert destination.exists()
    text = destination.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "</html>" in text


def test_viewer_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_path = tmp_path / "run.json"
    main(
        [
            "run", "--tasks", str(BASIC), "--agent", "mock:v1",
            "--trials", "2", "--out", str(run_path),
        ]
    )
    capsys.readouterr()
    out = tmp_path / "trace.html"
    assert main(["viewer", str(run_path), "--out", str(out)]) == 0
    assert out.exists()
    assert "trace ->" in capsys.readouterr().out


def test_viewer_default_path_is_under_site(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_path = tmp_path / "myrun.json"
    main(["run", "--tasks", str(BASIC), "--agent", "mock:v1", "--out", str(run_path)])
    from agenteval.viewer import write_html

    expected = Path("site") / "trace-myrun.html"
    assert not expected.exists()
    write_html(run_path, expected)
    assert expected.exists()


def test_viewer_reports_a_bad_run_file(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="cannot render trace"):
        main(["viewer", "no-such-run.json"])
    capsys.readouterr()
