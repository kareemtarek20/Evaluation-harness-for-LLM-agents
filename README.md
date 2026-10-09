# AgentEval

**Evaluation and regression-testing framework for LLM tool-using agents** — run
an agent over a JSON task suite, score every trial on answer, tools, steps,
latency and cost, and fail the build when the agent gets worse.

![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-211%20passing-brightgreen)

> Replace `OWNER/REPO` in the CI badge with your remote; the badge only resolves
> once this repository is pushed to GitHub. The workflow itself has never run on
> a runner — see [Limitations](#12-limitations-and-not-yet-verified).

---

## 2. Why this exists

Teams build LLM agents in an afternoon and then fly blind. A prompt tweak, a tool
rename, a model upgrade — none of it is measured, so "it seems fine" is the
release gate, and a regression is discovered by a user. Classic test suites do
not fit either: the same input can pass 4 of 5 times, the interesting failures
are about *which tools were called*, and a score of 87% tells you nothing about
whether task 12 broke.

AgentEval treats an agent like a service with a contract: a suite of tasks, a
scorer per task, a failure label per trial, repeated trials to separate signal
from noise, and a diff between two runs that exits non-zero when a task got
worse **beyond the noise floor**. No API key or network is needed to try it — the
harness ships with scripted mock agents and passes 211 tests offline.

## 3. Key features

- **Task suites as JSON** — 56 shipped tasks (`tasks/basic.json` 10,
  `tasks/extended.json` 46) across 8 categories: math, retrieval, multi_step,
  efficiency, hallucination, ambiguous, tool_error, injection.
- **Five match types** — `exact`, `contains`, `contains_any`, `numeric` (last
  number in the answer, with tolerance), `judge` (LLM against a rubric).
- **Tool-use scoring** — required tools, forbidden tools, and call order,
  checked independently of the answer.
- **Exactly one failure label per trial** — `wrong_answer | tool_error |
  missing_tool | forbidden_tool | max_steps | agent_error`, chosen by severity.
- **Regression gate** — `compare --fail-on-regression` prints fixed / regressed
  / unchanged with a two-sided Fisher exact p-value per task and exits 1.
- **Trials and flakiness** — `--trials 5` gives per-task pass rates and flags
  tasks that both pass and fail (0 < passes < trials).
- **Cost and latency per trial** — configurable per-million-token prices, judge
  cost reported separately so it is never hidden inside the agent's number.
- **LLM-as-judge with an agreement instrument** — 37 label-able rows
  (`tasks/judge_labels.json`), 10 built-in bias probes, and
  `scripts/judge_agreement.py` computing percent agreement and Cohen's kappa.
- **Sandboxed tools** — the calculator parses with `ast` against an operator
  whitelist; there is no `eval()` anywhere in the codebase.
- **Trace viewer** — one self-contained HTML page per run, tool calls step by
  step, filterable to failures or a single failure mode.
- **Runs are data** — artifacts store trials only; every aggregate in every
  report is recomputed, so a summary cannot drift from its own data.

## 4. Tech stack

| Technology | Purpose | Why chosen |
| --- | --- | --- |
| Python 3.11+ stdlib | models, scoring, stats, runner, reporting, CLI, viewer | An eval harness should be auditable and installable anywhere; `argparse`, `dataclasses`, `ast`, `html`, `statistics` and `math.comb` cover everything the core needs |
| `anthropic>=0.40` | `AnthropicAgent` tool loop, `AnthropicJudge` | The only runtime dependency, imported lazily so the offline demo and the whole test suite need no network and no key |
| `ast` whitelist evaluator | `calculator` tool | Model-supplied arithmetic must never reach `eval`; a syntax-tree walk with an operator whitelist and an exponent cap is both safe and testable |
| pytest (dev) | 211 tests | Plain-`assert` style, parametrised match-type and rejection tables |
| ruff (dev) | lint gate in CI | One tool for `E/F/I/UP/B/SIM/ANN/D`; the docstring and annotation rules enforce the house style mechanically |
| GitHub Actions | CI | Runs the suite and the regression gate with zero credentials |
| Static HTML + inline JS | trace viewer | Chosen over Streamlit: no server, no dependency, no install — the report opens by double-click |
| Fisher exact test | regression significance | 5 trials per task is a small-sample 2x2 table; the exact hypergeometric tail needs no normal approximation and cannot be hand-waved |

## 5. Architecture

```
            tasks/*.json  (suite)                 env: AGENTEVAL_MODEL
                    |                                     |
                    v                                     v
  CLI  agenteval run  -->  runner.run_suite  -->  Agent.run(task, toolbox)  -> AgentResult
   |                          |  x trials            |          ^
   |                          |                      v          | tool calls
   |                          |                 ToolBox (calculator | kb_search)
   |                          v                      (output, is_error)
   |                     scoring.score_trial ----> FailureMode (one label)
   |                          |        ^
   |                          |        | optional AnthropicJudge (match: judge)
   |                          v        |
   |                     runs/<id>.json  (trials only, JSON artifact)
   |                          |
   +--> report / viewer ----->+--> reporting.summarize  (aggregates recomputed)
   +--> compare --------------> reporting.compare + stats.fisher_exact -> exit 0/1
```

Layers point one way and only one module touches the clock. `models.py` is pure
data (no I/O, no imports beyond stdlib typing); `tools.py` and `scoring.py` and
`stats.py` are pure and unit-testable; `agents.py` is the only place that talks to
a model; `runner.py` is the only place that times a trial or catches an agent
exception; `reporting.py` rebuilds every number from the stored trials; and
`cli.py` is thin glue. That is why the same artifact can drive `report`,
`compare` and the HTML viewer without disagreement between them.

## 6. What I built, stage by stage

Each stage was gated: its tests and a manual verification command had to pass
before the next one started, and each ended in a commit.

| Stage | What was done | What was tested | Outcome (commit) |
| --- | --- | --- | --- |
| 0 | Repo layout, `pyproject.toml`, ruff + pytest config, `.gitignore` (keys excluded) | `import agenteval`, empty pytest run, `ruff check .` | Skeleton green before any logic (`2a5d63e`) |
| 1 | Dataclasses `Task`, `ToolCall`, `AgentResult`, `TrialResult`; `ToolBox` with `calculator` (AST sandbox) and `kb_search` | Calculator arithmetic; rejects `__import__('os')...`, `2 ^ 10`, exponents > 64; kb hit and no-match paths | 39 tests, tools return `(output, is_error)` (`390ca12`) |
| 2 | Five match types, tool checks, severity-ordered `classify()` | Each match type incl. numeric edge cases ("126.", "1,000", echoed question must not pass); every failure mode reachable; empty answer fails all types | 68 tests; one label per trial, `tool_error` only when the answer is also wrong (`5c4efa3`) |
| 3 | `Agent` protocol, `MockAgent` v1/v2, `AnthropicAgent` tool loop, runner with timing, cost, artifacts | Mock runs deterministic; agent crash becomes `agent_error`; `max_steps` enforced with a looping stub | 90 tests; `runs/stage3-mock-v1-sample.json` (`d18c1d8`) |
| 4 | `taskset` loader + `tasks/basic.json` (10) and `tasks/extended.json` (46) | Strict loading: duplicate ids, unknown tools, judge without rubric, `max_steps < 1`, non-array files all raise | 56 tasks validated, all 5 match types and 8 categories in use (`ce46845`) |
| 5 | `summarize`, `compare`, CLI `run/report/compare/validate` | v1 vs v2 reports `calc_power` regressed and 9 fixed; exit code 1 under `--fail-on-regression` | Gate demonstrated from the terminal; `runs/baseline-mock-v1.json`, `runs/new-mock-v2.json` (`b9979ff`) |
| 6 | `--trials`, flakiness rule, Fisher exact noise model, gating rules | 3/5 -> 2/5 stays quiet, 5/5 -> 0/5 gates; improvements never gate; alpha control; z-test fallback at >200 trials | 11 significance tests; committed 5-trial pair `runs/trials5-*.json` (`5dc5e78`) |
| 7 | `AnthropicJudge`, strict verdict parsing, kappa maths, `tasks/judge_labels.json` (37 rows, 10 bias probes), `scripts/judge_agreement.py`, `docs/judge_notes.md` | 44 tests: accept/reject reply shapes, prompt construction and truncation, kappa against hand-computed tables, stubbed client contract (`temperature=0.0`, no `tools`), judge crash -> `ERROR` not a guess | Instrument ships with blank `human_label`; live judging **not** run (no key) (`8002047`) |
| 8 | Two configurations x two suites at `--trials 5`, `docs/results.md`, three cited failure cases | 5 tests rebuild every table row from its artifact; doc states the skipped real run verbatim | Real-agent run **skipped for lack of credentials**; measuring pipeline verified; found and fixed a genuine SDK bug (`03dee55`) |
| 9 | `.github/workflows/ci.yml`, HTML trace viewer + `agenteval viewer`, `CONTRIBUTING.md`, this README | 13 viewer tests incl. HTML-escaping of hostile tool output and CSP well-formedness; every CI step simulated locally | 211 tests, ruff clean, browser-verified trace page (`site/trace-stage8-basic-v2-trials5.html`) |

The stage-8 SDK bug is worth calling out because a real run found it: with
`anthropic` 1.12.1, `messages.create()` no longer declares `temperature`, so the
first credentialed call shape raised `TypeError` before touching the network.
`agents.build_sampling_kwargs()` now inspects the installed signature and sends
the value through whichever route exists (`parameter`, `extra_body`, or
`unsupported`), and `tests/test_agents.py` pins that contract against the SDK
that is actually installed.

## 7. Quick start (offline, no key, no install)

From the project root, with Python 3.11+ — every command below runs against the
shipped mock agents and writes real artifacts:

```bash
git clone <your-fork-or-this-repo> agenteval && cd agenteval

# 1. check the suites are valid (strict loader, no execution)
python -m agenteval.cli validate tasks

# 2. two configurations of the same suite, 5 trials each
python -m agenteval.cli run --tasks tasks/basic.json --agent mock:v1 --trials 5 \
  --label demo-v1 --out runs/demo-v1.json
python -m agenteval.cli run --tasks tasks/basic.json --agent mock:v2 --trials 5 \
  --label demo-v2 --out runs/demo-v2.json

# 3. read one run
python -m agenteval.cli report runs/demo-v1.json

# 4. diff them and let the regression fail the command
python -m agenteval.cli compare runs/demo-v1.json runs/demo-v2.json --fail-on-regression
echo "exit code: $?"

# 5. open the trace
python -m agenteval.cli viewer runs/demo-v2.json --out site/trace-demo.html
```

Expected (this is measured output, not illustrative):

```text
validate     -> ok: 56 tasks in 2 file(s); categories: {'math': 8, 'retrieval': 20, ...}
               not task suites (left alone): judge_labels.json
report v1    -> success rate 10.0% (5/50)   tool accuracy 20.0%   avg steps 1.20
compare      -> fixed 9 / regressed 1 / unchanged 0
               gating 1 of 1 regression(s) exceed the noise floor
               FAIL: 1 regression(s) beyond noise (alpha=0.05): calc_power
exit code    -> 1  (intentional: mock v2 keeps one regression on purpose)
viewer       -> site/trace-demo.html  (50 trial cards, 5 of them failures)
```

That block was verified by cloning this repository into a fresh temp directory and
running it there exactly as written - no `pip install`, no API key, no network -
together with `python -m pytest -q` (211 passed) and `python -m ruff check .`
(clean) in the same clone.

To get the `agenteval` command name used in CI, and the dev tools:

```bash
pip install -e ".[dev]"
agenteval --help
python -m pytest -q && python -m ruff check .
```

That editable install is what the CI job uses; it was not run in this offline
environment, so if you skip it use `python -m agenteval.cli` everywhere the docs
say `agenteval`.

## 8. Evaluating a real agent

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # never committed; .env is gitignored
export AGENTEVAL_MODEL=claude-sonnet-4-5 # the harness reads this, no model id in logic

# two system-prompt configurations, judged, 5 trials each
python -m agenteval.cli run --tasks tasks/extended.json --agent anthropic --system v1 \
  --judge --trials 5 --out runs/real-v1.json
python -m agenteval.cli run --tasks tasks/extended.json --agent anthropic --system v2 \
  --judge --trials 5 --out runs/real-v2.json

python -m agenteval.cli compare runs/real-v1.json runs/real-v2.json --fail-on-regression
python -m agenteval.cli viewer runs/real-v2.json
```

Useful knobs: `--agent anthropic:claude-haiku-4-5` pins a model per run;
`--max-tokens`, `--temperature`; `--input-price/--output-price` (USD per million
tokens) override the built-in price table; `--judge-model` grades with a
different model than the agent answers with; `--json` emits machine-readable
summaries for CI logs. Judge tokens are reported separately and are deliberately
**not** folded into the run cost.

Judge agreement, once you have graded the sheet by hand
(`tasks/judge_labels.json`, `human_label` is blank for all 37 rows):

```bash
python scripts/judge_agreement.py collect   # needs key + AGENTEVAL_MODEL
python scripts/judge_agreement.py score     # agreement, Cohen's kappa, disagreements
```

## 9. Results

**All figures below are mock results.** They measure the harness end to end —
scoring, classification, significance, cost arithmetic, the gate, the viewer —
against scripted stub agents. They say nothing about any real model, because no
credentialed run was possible here. `docs/results.md` holds the full write-up and
`tests/test_stage8_results.py` rebuilds each row from its artifact on every test
run, so this table cannot silently drift.

Two configurations, `--trials 5`, pricing overridden at `3 / 15` USD per million
tokens (the published claude-sonnet-4-5 rate) applied to synthetic token counts:

| suite | config | success | answer acc | tool acc | avg steps | cost USD | flaky | top failure modes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic (10 tasks) | A `mock:v1` | 10.0% (5/50) | 20.0% | 20.0% | 1.20 | 0.0572 | none | missing_tool 35, forbidden_tool 5, wrong_answer 5 |
| basic (10 tasks) | B `mock:v2` | 90.0% (45/50) | 90.0% | 100.0% | 2.10 | 0.0822 | none | tool_error 5 |
| extended (46 tasks) | A `mock:v1` | 0.0% (0/230) | 0.0% | 15.2% | 1.02 | 0.3076 | none | missing_tool 195, max_steps 15, wrong_answer 15, tool_error 5 |
| extended (46 tasks) | B `mock:v2` | 4.3% (10/230) | 4.3% | 17.4% | 1.07 | 0.3029 | none | missing_tool 190, max_steps 15, wrong_answer 15 |

Latency is omitted on purpose: the mocks call an in-process AST calculator, so
the measured 0.0-0.2 ms per trial measures Python, not an agent.

A -> B, with the noise model on:

| suite | fixed | regressed | p (Fisher exact) | gate |
| --- | --- | --- | --- | --- |
| basic | 9 tasks, each 0/5 -> 5/5 | `calc_power` 5/5 -> 0/5 | 0.0079 both ways | exit 1 |
| extended | 2 tasks (`tool_error_caret`, `inject_roster`), each 0/5 -> 5/5 | none | 0.0079 | exit 0 |

Success-rate deltas: basic **+80.0 pp**, extended **+4.3 pp**. Cost moved
**+$0.0250** on basic (B does the work) and **-$0.0047** on extended (B answers
unscripted tasks in a shorter sentence).

Three findings that survive being mock data, because each is a property of the
*harness* rather than of the stubs:

1. **A single task can and should fail a build that is otherwise 90% better.**
   `calc_power` went 5/5 -> 0/5 while nine tasks went 0/5 -> 5/5. An aggregate
   score would have hidden it; a per-task test with an exit code did not. The
   cause is visible in one line of the trace: v2 writes `2 ^ 10`, the sandbox
   refuses `BitXor`, the agent answers "Cannot compute, the tool failed." —
   labelled `tool_error`, not `agent_error`, because the tool failed as data.
2. **Failure-mode histograms localise the bug.** Extended A's failures are 195
   `missing_tool` out of 230 trials — the stub never retrieves — versus basic B's
   single `tool_error`. "The agent is weak at retrieval" and "the agent broke"
   are different tickets, and the classifier separates them without any model.
3. **Repetition changes the verdict, not just the decimal places.** With one
   trial per task, any flip gates; with five, the same flip must clear a Fisher
   exact test (p=0.0079 for 5/5 -> 0/5, p~0.206 for 4/5 -> 1/5). That is the
   difference between a red build you trust and one you learn to ignore.

## 10. Engineering decisions

**No `eval()` — an `ast` walk with a whitelist and an exponent cap.** The
calculator receives strings written by a model, so it is the injection surface.
`safe_calculate` parses with `ast.parse(..., mode="eval")`, then accepts only
literals, `+ - * / // % **`, unary `+ -`, and parentheses. Everything else —
names, calls, attribute access, subscripts, comparisons — is rejected by node
type. `2 ^ 10` is rejected rather than reinterpreted, because in Python `^` is
`BitXor` and silently "fixing" it would evaluate something the caller did not
ask for. Literal exponents are capped at 64 so `9 ** 9 ** 9` cannot pin a CPU,
and results are formatted with `%.10g` so float noise does not turn 0.1+0.2 into
a wrong-answer verdict. This is the whole argument for a whitelist over a
blacklist: the accepted set is small enough to enumerate in a test.

**`numeric` match reads the *last* number in the answer.** Agents explain before
they conclude ("meals are $75/day, so for 7 days the cap is $525"), so scoring
the first number rewards the preamble and scoring any number rewards an echoed
question. The last figure is the claim. The matcher strips thousands separators,
accepts trailing punctuation ("126."), compares within a per-task `tolerance`,
and fails an empty answer. `tests/test_scoring.py` pins each of those cases,
including the trap where the agent repeats the question's own numbers.

**Trials exist because a single sample is not a measurement.** One run per task
cannot distinguish a 60%-flaky agent from a stable one, and a regression gate
built on single samples produces red builds nobody reads. `--trials N` stores N
trials per task, reports per-task pass rates, and flags flaky tasks with
`0 < passes < trials`. The cost is linear and stated up front: N=5 multiplies
token spend by 5.

**Significance is a two-sided Fisher exact test, computed with `math.comb`.** The
comparison per task is a 2x2 table (base passes/fails vs new passes/fails). At
five trials the smallest possible p-value is 1/126 = 0.0079, which 5/5 -> 0/5
hits, while 3/5 -> 2/5 comes out at p=1.000 and 4/5 -> 1/5 at p=0.206 - exactly
the "don't gate that" cases. The exact tail is both the right tool for sparse
tables and cheap enough to compute without a dependency: enumerate every table
with the same margins and sum the probabilities at or below the observed one.
Above 200 combined trials that enumeration stops being cheap, so `stats.py`
switches to a two-proportion z-test; the boundary is tested, not assumed. Two
honesty rules sit on top: improvements never gate, and a run with one trial per
task gates on any regression (`method="single-trial"`) because there is no noise
model to hide behind - the CI demo relies on exactly that.

**Severity-ordered single label.** `classify()` walks
`agent_error > max_steps > forbidden_tool > missing_tool > tool_error >
wrong_answer` and returns one label. A crash that also skipped a tool is a crash;
a tool that errored but whose answer still matched is not a failure at all. One
label per trial keeps the histogram honest and the filters in the viewer usable.

**Errors are data.** `ToolBox.call` never raises: unknown tool, schema violation,
or a buggy handler all come back as `(output, is_error=True)`. An agent exception
is caught by the runner and recorded as `agent_error`. A suite full of broken
tasks still finishes and still reports, which is what makes it usable in CI.

**Artifacts store trials, never aggregates.** `runs/*.json` holds per-trial
results plus configuration; `summarize()` and `compare()` recompute everything.
A committed baseline cannot disagree with its own numbers, and a report format
change is a code change, not a data migration.

**Static HTML for the viewer, and hostile output escaped.** A Streamlit viewer
needs a server, a dependency and a port forward; an evaluation report has to be
something you can attach to a PR and open offline. `render_html` escapes every
model- and tool-produced string with `html.escape(..., quote=True)`, ships a
`default-src 'none'` CSP, and has no external references — `tests/test_viewer.py`
renders a tool output containing `<script>alert(1)</script>` and asserts the
markup appears only in escaped form. The CSP tag itself was broken until the page
was opened in a real browser: a missing quote made the browser swallow the whole
document as an attribute value and block the filter script. There is now a
parser-based test for exactly that class of bug.

## 11. Testing

```bash
python -m pytest -q      # 211 passed
python -m ruff check .   # All checks passed!
```

Verified on Python 3.11.6 (Windows) and simulated step-by-step for the
`ubuntu-latest` CI job; the suite is offline, deterministic and takes about two
seconds.

| Area | Tests | What is covered |
| --- | --- | --- |
| `test_calculator.py` | 21 | arithmetic, `^`/names/calls/attribute rejection, exponent cap, float formatting |
| `test_judge.py` | 44 | verdict parsing accept/reject shapes, prompt + truncation, kappa vs hand-computed tables, stubbed client contract, `ERROR` path, agreement script CLI |
| `test_agents.py` | 18 | mock determinism, crash -> `agent_error`, `max_steps` enforcement, sampling-route probing incl. a check against the installed SDK |
| `test_classify.py` | 16 | every failure mode reachable, severity order, tool-error-only-when-wrong |
| `test_taskset.py` | 16 | strict loading, duplicate ids, unknown tools, judge rubric required, suite/`judge_labels` discrimination |
| `test_viewer.py` | 13 | one card per trial, step order, filters, escaping of hostile output, CSP well-formedness, `viewer` command |
| `test_scoring.py` | 13 | five match types, numeric edge cases, tolerance, empty answer |
| `test_significance.py` | 11 | 3/5->2/5 quiet, 5/5->0/5 gates, improvements never gate, alpha control, labels in rendered output |
| `test_stats.py` | 10 | Fisher exact against enumerated tables, z-test fallback boundary, flaky rule |
| `test_runner.py` | 10 | trials x tasks loop, artifact round-trip, schema version guard |
| `test_kb_search.py` | 9 | retrieval ranking, no-match text, schema validation |
| `test_cli.py` / `test_models.py` / `test_reporting.py` | 8 each | end-to-end run/report/compare with exit codes, dataclass round-trips, summary + diff rendering |
| `test_stage8_results.py` | 5 | every number and quoted failure in `docs/results.md` is rebuilt from its artifact |
| `test_import.py` | 1 | package imports with no side effects |

Coverage as a percentage is **not measured** — no coverage tool is configured,
deliberately, so the number is not quoted. What is covered is the table above;
what is *not* covered is any network path (real completions, real judge
verdicts, whether `extra_body` reaches the API as `temperature`), because those
need credentials.

`CONTRIBUTING.md` explains how to add a task, a tool, or a custom agent - the
whole interface is `name` + `run(task, toolbox) -> AgentResult` - and the two
rules that shape this codebase: no fabricated numbers, and deterministic grading
before model grading.

## 12. Limitations and Not yet verified

**Not yet verified** — things this build could not run, listed instead of
guessed:

- **No real model was ever evaluated.** There is no `ANTHROPIC_API_KEY` in this
  environment. Two attempts were made; the second reached the client and failed
  on credentials in every trial, recorded as `agent_error`. Every number in
  section 9 is mock.
- **CI has never executed on a runner.** The repository has no remote, so
  `.github/workflows/ci.yml` has no run history. Each of its steps was
  reproduced locally (validate, both 5-trial runs, report, the gate returning
  exit 1 with `calc_power`, the same-run comparison returning 0, the viewer
  writing its page), which validates the commands but not the runner.
- **The judge has no live verdicts and no measured kappa.** `AnthropicJudge` is
  unit-tested against a stubbed client; `tasks/judge_labels.json` ships with all
  37 `human_label` fields blank, so agreement and Cohen's kappa are uncomputed.
  The verbosity / position / self-preference biases in `docs/judge_notes.md` are
  documented risks with probes built to expose them, not measured effects.
- **Badges and clone URL are placeholders.** The CI badge resolves only after a
  push to GitHub with the badge URL pointed at the real repo.
- **`extra_body` routing is unconfirmed server-side.** The installed SDK accepts
  the call shape; only a credentialed run can show the API honours it.

**Design limitations**, in place today:

- **Single-turn tasks only.** A task is one prompt in, one answer out. Agents
  that need clarification, or that should refuse then retry, are not expressible.
- **Sequential execution.** Trials run one after another; a 46-task x 5-trial
  suite is 230 sequential agent calls, which is the first thing that will make a
  real evaluation slow.
- **`kb_search` is keyword scoring over a 14-record fixture**, deterministic by
  design but weak on paraphrase. A real corpus will need retrieval that is either
  better or explicitly mocked.
- **Cost is arithmetic on reported token counts.** Retries, streaming overhead,
  prompt caching, and cache-busting are not modeled; judge cost is reported
  outside the run cost rather than attributed per task.
- **Latency is wall-clock around one `run()` call**, so it is only meaningful
  against a real model.
- **The noise model compares pass counts, not answers.** Two runs that both pass
  can differ in tool order, steps or phrasing without `compare` noticing.
- **The mock scripts cover the basic suite**, which is why extended is dominated
  by `missing_tool`. That is a stub artifact, not a finding about agents.
- **`contains`/`contains_any` are substring checks** and can be satisfied by an
  answer that quotes the question; the numeric and judge paths are safer, and the
  suite prefers them.

## 13. Roadmap

- **Parallel execution** — a worker pool over `(task, trial)` pairs with a
  per-run budget cap and deterministic artifact ordering, so 230 calls do not
  serialize.
- **More tool types** — read-only HTTP, SQL over a fixture database, file access
  in a temp sandbox, each with the same `(output, is_error)` contract and an
  argument-schema gate, so tool-use regressions cover more than arithmetic and
  retrieval.
- **Multi-turn tasks** — a conversation script (user, agent, tool, user) with
  per-turn checks and an end-of-conversation rubric, plus refusal-then-ask
  behaviour as a first-class expected outcome.
- **Dataset leaderboard** — several agents x several suites in one table, with
  per-category scores, cost per solved task, and flakiness, stored as artifacts
  so the leaderboard is regenerable from data.
- **Judge calibration loop** — ship the agreement script's kappa as a CI check on
  the judge itself, and auto-generate rubric-paraphrase variants to detect judge
  drift when the grading model changes.
- **OpenAI-compatible adapter** — the `Agent` protocol already accepts anything;
  a second backend would exercise the harness's provider neutrality.
- **Trace diffing** — the viewer renders one run today; side-by-side tool-call
  diffs for a regressed task are the natural next page.

## 14. License

MIT. See `LICENSE`.
