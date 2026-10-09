"""Render a run artifact as one self-contained HTML trace page.

Static HTML rather than Streamlit on purpose: the page has to open from a fresh
clone with no server, no extra dependency and no network, so `agenteval viewer`
writes a file you can double-click. Every model- or tool-produced string is
HTML-escaped, so a hostile tool output cannot inject markup into a report.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from agenteval.models import TrialResult
from agenteval.reporting import summarize
from agenteval.runner import load_run, trials_from_artifact

__all__ = ["render_html", "write_html"]

#: Inline CSS/JS only; no external fetches. The page is a report, not a browser.
_CSP = "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'"

_CSS = """
:root { --ok:#0a7d33; --bad:#b42318; --warn:#b54708; --line:#d9dde3; --ink:#15181d; }
* { box-sizing: border-box; }
body { font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 0; color: var(--ink); background: #f6f7f9; }
header { background: #fff; border-bottom: 1px solid var(--line); padding: 16px 24px; }
h1 { font-size: 18px; margin: 0 0 6px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; color: #454f5c; font-size: 13px; }
main { padding: 16px 24px 40px; max-width: 1100px; }
.toolbar { display: flex; gap: 16px; align-items: center; margin: 4px 0 20px; }
.task { background: #fff; border: 1px solid var(--line); border-radius: 8px; margin: 0 0 14px; }
.task > h2 { font-size: 15px; margin: 0; padding: 10px 14px; border-bottom: 1px solid var(--line);
             display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.trial { padding: 10px 14px; border-bottom: 1px solid #eef0f3; }
.trial:last-child { border-bottom: 0; }
.badge { font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 10px;
         text-transform: uppercase; letter-spacing: .02em; }
.badge.pass { background: #e3f4e8; color: var(--ok); }
.badge.fail { background: #fdeceb; color: var(--bad); }
.badge.mode { background: #fdf3e3; color: var(--warn); }
.kv { color: #55606d; font-size: 12px; margin: 2px 0 6px; }
pre { background: #f2f4f7; border-radius: 6px; padding: 8px 10px; margin: 4px 0;
      white-space: pre-wrap; word-break: break-word; font-size: 12.5px;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
ol.steps { margin: 6px 0 0; padding-left: 0; list-style: none; }
ol.steps > li { border-left: 3px solid var(--line); padding: 0 0 6px 10px; margin: 0 0 8px; }
ol.steps > li.err { border-left-color: var(--bad); }
.label { font-weight: 700; font-size: 12px; color: #3a424d; }
.hide { display: none; }
"""

_JS = """
var only = document.getElementById('only-failures');
var modes = document.getElementById('mode-filter');
function apply() {
  var cards = document.querySelectorAll('article.trial');
  for (var i = 0; i < cards.length; i++) {
    var card = cards[i];
    var failed = card.getAttribute('data-passed') === 'false';
    var mode = card.getAttribute('data-mode');
    var modeOk = modes.value === 'all' || modes.value === mode;
    card.classList.toggle('hide', !modeOk || (only.checked && !failed));
  }
  var tasks = document.querySelectorAll('section.task');
  for (var j = 0; j < tasks.length; j++) {
    var left = tasks[j].querySelectorAll('article.trial:not(.hide)').length;
    tasks[j].classList.toggle('hide', left === 0);
  }
}
only.addEventListener('change', apply);
modes.addEventListener('change', apply);
apply();
"""


def _esc(value: Any) -> str:
    """HTML-escape any value (and attribute quote) for the page."""
    return html.escape("" if value is None else str(value), quote=True)


def _trial_card(trial: TrialResult) -> str:
    """Render one scored trial with its tool calls in step order."""
    label = "pass" if trial.passed else trial.failure_mode.value
    badge = "pass" if trial.passed else "fail"
    passed_attr = "true" if trial.passed else "false"
    lines = [
        f'<article class="trial" data-passed="{passed_attr}" data-mode="{_esc(label)}">',
        f'  <span class="badge {badge}">{_esc(label)}</span>',
        f'  <div class="kv">trial {trial.trial} &middot; {trial.steps} steps'
        f' &middot; {trial.latency_ms:.2f} ms'
        f' &middot; {trial.input_tokens} in / {trial.output_tokens} out tokens'
        f' &middot; ${trial.cost_usd:.6f}</div>',
    ]
    if trial.result.error:
        lines.append(f'  <div class="kv">agent error: {_esc(trial.result.error)}</div>')
    if trial.result.tool_calls:
        lines.append('  <div class="label">tool calls</div>  <ol class="steps">')
        for call in trial.result.tool_calls:
            err = " err" if call.is_error else ""
            note = ' <span class="badge fail">tool error</span>' if call.is_error else ""
            lines.append(
                f'    <li class="step{err}">'
                f'<span class="label">{call.step}. {_esc(call.tool)}</span>{note}'
                f'<pre>{_esc(json.dumps(call.input, indent=2, sort_keys=True))}</pre>'
                f"<pre>{_esc(call.output)}</pre></li>"
            )
        lines.append("  </ol>")
    else:
        lines.append('  <div class="kv">no tool calls</div>')
    lines.append(f'  <div class="label">answer</div><pre>{_esc(trial.result.answer)}</pre>')
    problems = []
    if trial.forbidden_used:
        problems.append("forbidden: " + ", ".join(trial.forbidden_used))
    if trial.missing_required:
        problems.append("missing: " + ", ".join(trial.missing_required))
    if problems:
        lines.append(f'  <div class="kv">{_esc("; ".join(problems))}</div>')
    if trial.judge_verdict:
        lines.append(
            f'  <div class="kv">judge {_esc(trial.judge_verdict)}: '
            f'{_esc(trial.judge_rationale)}</div>'
        )
    lines.append("</article>")
    return "\n".join(lines)


def render_html(artifact: dict[str, Any], task_categories: dict[str, str] | None = None) -> str:
    """Return a complete HTML document for one run artifact."""
    summary = summarize(artifact, task_categories)
    trials = trials_from_artifact(artifact)
    by_task: dict[str, list[TrialResult]] = {}
    for trial in trials:
        by_task.setdefault(trial.task_id, []).append(trial)
    modes = sorted({trial.failure_mode.value for trial in trials if not trial.passed})
    options = "".join(f'<option value="{_esc(mode)}">{_esc(mode)}</option>' for mode in modes)
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta http-equiv="Content-Security-Policy" content="' + _CSP + '">',
        f"<title>AgentEval trace {_esc(summary.run_id)}</title>",
        f"<style>{_CSS}</style></head><body>",
        "<header>",
        f"  <h1>AgentEval trace &mdash; {_esc(summary.run_id)}</h1>",
        '  <div class="meta">',
        f"    <span>agent <b>{_esc(summary.agent)}</b></span>",
        f"    <span>model {_esc(summary.model or 'n/a')}</span>",
        f"    <span>label {_esc(summary.label or '-')}</span>",
        f"    <span>suite {_esc(summary.task_source or 'n/a')}</span>",
        f"    <span>{summary.tasks} tasks x {summary.trials_per_task} trials"
        f" = {summary.trials} trials</span>",
        f"    <span>success {summary.success_rate:.1%} ({summary.passed}/{summary.trials})</span>",
        f"    <span>tool accuracy {summary.tool_accuracy:.1%}</span>",
        f"    <span>avg steps {summary.avg_steps:.2f}</span>",
        f"    <span>cost ${summary.total_cost_usd:.4f}</span>",
        "  </div>",
        "</header>",
        "<main>",
        '  <div class="toolbar">',
        '    <label><input type="checkbox" id="only-failures"> failures only</label>',
        f'    <label>mode <select id="mode-filter"><option value="all">all</option>{options}'
        "</select></label>",
        "  </div>",
    ]
    for task_id, agg in sorted(summary.per_task.items()):
        parts.append(f'<section class="task" data-task="{_esc(task_id)}">')
        strength = "pass" if agg.passes == agg.trials else "fail"
        parts.append(
            f'  <h2>{_esc(task_id)}'
            f' <span class="badge {strength}">{agg.passes}/{agg.trials} passed</span>'
            f' <span class="kv">{_esc(agg.category)} &middot; avg {agg.avg_steps:.2f} steps'
            f" &middot; ${agg.cost_usd:.6f}</span></h2>"
        )
        for trial in sorted(by_task.get(task_id, []), key=lambda item: item.trial):
            parts.append(_trial_card(trial))
        parts.append("</section>")
    parts.append(f"<script>{_JS}</script></main></body></html>")
    return "\n".join(parts)


def write_html(run_path: Path | str, out_path: Path | str) -> Path:
    """Load a run artifact and write its trace page; return the destination."""
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_html(load_run(run_path)), encoding="utf-8", newline="\n")
    return destination
