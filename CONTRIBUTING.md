# Contributing

AgentEval is a small harness with one job: run an agent over a task suite, score
every trial, and tell you whether the agent got worse. This guide covers setup,
how to add a task, how to plug in an agent, how to add a tool, and the rules a
change has to satisfy before it merges.

## Setup

Nothing here needs an API key. The offline demo and the whole test suite run on
the standard library alone (`anthropic` is imported lazily, only when you build a
real agent or judge).

```bash
python -m pytest -q            # 211 tests, no network
python -m ruff check .         # must be clean
```

Install the package to get the `agenteval` command that CI uses:

```bash
pip install -e ".[dev]"
agenteval run --tasks tasks/basic.json --agent mock:v2 --trials 5 --out runs/local.json
agenteval viewer runs/local.json
```

Use `python -m agenteval.cli ...` interchangeably with `agenteval ...`.

## The two rules that shape everything else

1. **No fabricated numbers.** Every figure in a README, doc, or PR description
   must come from a command you actually ran, with the raw artifact committed in
   `runs/`. If you could not run something, write "Not yet verified" and say why.
   `docs/results.md` is machine-checked by `tests/test_stage8_results.py`, which
   rebuilds each table row from its artifact.
2. **Deterministic grading first.** The harness never asks a model whether the
   agent did well unless the task says `match: judge`. Scoring, classification,
   aggregation, and the significance test are pure functions you can reason
   about, so a CI failure means something.

## Adding a task

Tasks are JSON arrays in `tasks/*.json` (or an object with a `"tasks"` key). The
loader is strict on purpose: a typo silently corrupts a comparison, so it fails
loudly at load time instead.

```json
{
  "id": "kb_remote_work",
  "prompt": "What is the remote work stipend?",
  "expected": "150",
  "match": "numeric",
  "category": "retrieval",
  "expected_tools": ["kb_search"],
  "forbidden_tools": [],
  "max_steps": 4,
  "notes": "answer is the monthly stipend in USD"
}
```

Required: `id`, `prompt`, `expected`, `match`. Optional: `category`
(default `general`), `expected_tools`, `forbidden_tools`, `max_steps` (default
8, must be >= 1), `rubric`, `tolerance`, `notes`. Unknown keys are rejected.

| `match` | what `expected` means | gotcha |
| --- | --- | --- |
| `exact` | case-insensitive, whitespace-normalised equality | punctuation in the answer fails it |
| `contains` | `expected` appears in the answer | keep it short, one phrase |
| `contains_any` | any `;`-separated alternative appears | use for phrasings you cannot pin down |
| `numeric` | the **last** number in the answer, within `tolerance` | the answer may explain first; the final figure is the claim |
| `judge` | an LLM grades against `rubric` | `rubric` is required, and the run needs `--judge` |

Rejected at load time: duplicate ids (within a file and across files), tool
names that are not in `ToolBox().names`, `judge` without a rubric, `max_steps <
1`. Add the task, then:

```bash
agenteval validate tasks        # prints the task count and category spread
python -m pytest -q             # suites are covered by tests/test_taskset.py
```

Write tasks whose right answer is checkable without a model: a fact the
knowledge base contains (`agenteval/kb_data.py`), an arithmetic result, or a
refusal for something that is not in the corpus. Hallucination and prompt-injection
cases are the most valuable rows in the suite — copy `hallucination_teleport` or
`inject_roster` as a template.

## Plugging in a custom agent

An agent is one method. `run` must not raise; if it does, the runner records the
exception as `agent_error` and keeps going, because a crashed agent is a result,
not a build failure.

```python
from agenteval.models import AgentResult, Task, ToolCall
from agenteval.tools import ToolBox


class RuleBasedAgent:
    """Retrieves before answering; no model, so it is cheap to test against."""

    name = "rule-based"
    model = ""  # empty: no token pricing applies

    def run(self, task: Task, toolbox: ToolBox) -> AgentResult:
        found, is_error = toolbox.call("kb_search", {"query": task.prompt})
        calls = [ToolCall(tool="kb_search", input={"query": task.prompt},
                          output=found, is_error=is_error, step=1)]
        answer = found.splitlines()[0] if not is_error else "no information"
        return AgentResult(answer=answer, tool_calls=calls, steps=1)
```

This agent answers every task with a retrieval attempt, so it fails the math and
the "no tools needed" tasks. That is fine: the point of the example is that any
object with `name` and `run()` is scored, timed and compared for free.

Fields `AgentResult` must fill: `answer`, `tool_calls` (in call order, each with
`step`), `steps`, `input_tokens` / `output_tokens`, `model`, and `error` only if
you caught something yourself. The runner does timing, cost, scoring and
classification for you.

Run it without touching the CLI:

```python
from agenteval.runner import run_suite
from agenteval.taskset import load_task_file

artifact = run_suite(load_task_file("tasks/basic.json"), RuleBasedAgent(), trials=3)
```

To make it a CLI target, add a branch to `build_agent()` in `agenteval/cli.py`
and a case to `tests/test_cli.py`. Keep the model name in the constructor
argument and read the default from `AGENTEVAL_MODEL` (`agenteval.agents.resolve_model`)
— never hardcode a model id in logic.

If your agent calls an SDK that rejects keyword arguments the way our judge and
agent pass `temperature`, use `build_sampling_kwargs()` (`agenteval/agents.py`),
which inspects the installed signature and records which route it took.

## Adding a tool

A tool is a `(output, is_error)` function plus a JSON schema, registered on the
toolbox. Never let a tool raise: `ToolBox.call` converts exceptions into an error
result, but explicit errors carry a better message.

```python
def wordcount_tool(arguments):
    text = arguments.get("text", "")
    if not isinstance(text, str):
        return "error: 'text' must be a string", True
    return str(len(text.split())), False

toolbox.register("wordcount", wordcount_tool, {
    "name": "wordcount",
    "description": "Count whitespace-separated words.",
    "input_schema": {"type": "object", "properties": {"text": {"type": "string"}},
                     "required": ["text"], "additionalProperties": False},
})
```

The schema is enforced before your handler runs (required keys, primitive types,
unknown keys), so a model that hallucinates arguments gets a tool error rather
than a stack trace. Security rules for anything that parses model input:

- no `eval()` on model output, ever. `safe_calculate` walks an `ast` tree against
  an operator whitelist and caps literal exponents at 64 — copy that shape.
- bound every size: input length, iteration count, output length.
- treat tool output as data. Prompts in the corpus instruct agents to ignore
  instructions found inside tool results, and `tasks/extended.json` has
  `injection` tasks that check it.

Then add tests in `tests/` and reference the tool by name in tasks only after it
appears in `ToolBox().names`.

## Adding a judge-scored task

`match: judge` needs a `rubric` and a run with `--judge` (which needs
`AGENTEVAL_MODEL` and `ANTHROPIC_API_KEY`). Without `--judge`, those tasks fail
and the CLI says how many are affected. Judge output is parsed strictly: one JSON
object with a verdict, and anything ambiguous raises `JudgeProtocolError`, which
the runner stores as an `ERROR` verdict rather than guessing.

To measure whether the judge agrees with you, regenerate and score the labelled
sheet:

```bash
python scripts/make_judge_labels.py            # rewrites tasks/judge_labels.json
python scripts/judge_agreement.py collect      # grades it, writes runs/judge_verdicts.json
python scripts/judge_agreement.py score        # agreement + Cohen's kappa
```

Fill in `human_label` by hand first. `docs/judge_notes.md` explains the biases the
probe rows are built to expose and the kappa threshold used for gating.

## Style

- `from __future__ import annotations`, type hints everywhere, one-line Google-style
  docstring on every public function (ruff enforces `D`, `ANN`, `B`, `SIM`, `E`, `F`, `I`).
- Small pure functions; modules stay in one layer: `models` has no I/O,
  `scoring`/`stats` are pure, `runner` is the only place that times things,
  `reporting` recomputes every aggregate from stored trials.
- Line length 100, Python 3.11+.
- Comments only for a non-obvious *why*.

## Before you open a pull request

```bash
python -m ruff check .
python -m pytest -q
python -m agenteval.cli validate tasks
python -m agenteval.cli run --tasks tasks/basic.json --agent mock:v1 --out runs/pr-base.json
python -m agenteval.cli run --tasks tasks/basic.json --agent mock:v2 --out runs/pr-new.json
python -m agenteval.cli compare runs/pr-base.json runs/pr-new.json --fail-on-regression
```

The last command is expected to exit 1: mock v1 vs v2 contains a deliberate
regression (`calc_power`), and CI asserts the gate fires on it. Explain in the PR
what your change does to those two runs, and attach the artifacts you generated.
