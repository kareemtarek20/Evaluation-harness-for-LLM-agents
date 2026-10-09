# Stage 8 - evaluation results

Date: 2026-10-09. Every number on this page is recomputed from a run artifact in
`runs/`; the reproduction commands are at the bottom, and
`tests/test_stage8_results.py` fails if the table below ever stops matching the
artifacts.

## What was skipped, and why (read this first)

The brief asks for a **real** agent evaluated over two configurations. That did
not happen, because this environment has no Anthropic credentials:

```text
$ python -c "import os; print('ANTHROPIC_API_KEY set:', bool(os.environ.get('ANTHROPIC_API_KEY','').strip()))"
ANTHROPIC_API_KEY set: False
```

Two real attempts were made anyway, and the harness recorded them instead of
hiding them:

1. With `anthropic` 1.12.1 installed, the first attempt failed **before any
   network call** with
   `TypeError: Messages.create() got an unexpected keyword argument 'temperature'`.
   That was a genuine bug in this harness, not in the API: newer SDK clients
   declare `extra_body` instead of a typed `temperature`. It is fixed by
   `agents.build_sampling_kwargs`, which probes the installed signature and
   records the route it took (`parameter` / `extra_body` / `unsupported`).
   `tests/test_agents.py::test_installed_sdk_accepts_the_sampling_kwargs_we_send`
   pins that contract against whatever SDK is actually installed.
2. After the fix, `--agent anthropic` reaches the client and stops one step
   later, at credentials, in **every** trial:
   `agent_error` - `TypeError: "Could not resolve authentication method. Expected
   one of api_key, auth_token, or credentials to be set. ..."`
   (scratch artifact, since it contains no signal beyond the error).

So: **no model-generated answers, no real cost figures and no judge verdicts are
reported anywhere on this page.** The two configurations below are the harness's
offline stand-ins - `mock:v1` (six scripted bug classes) and `mock:v2` (those
fixed, one regression kept on purpose). They verify the *measurement pipeline*
end to end; they say nothing about any real model.

## Two configurations, `--trials 5`

Pricing is a CLI override (`--input-price 3 --output-price 15`, the published
claude-sonnet-4-5 rate) applied to the mock agents' synthetic token counts, so
the arithmetic is real but the token counts are not.

| suite | config | success | answer acc | tool acc | avg steps | cost USD | flaky | top failure modes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic (10 tasks) | A `mock:v1` | 10.0% (5/50) | 20.0% | 20.0% | 1.20 | 0.0572 | none | missing_tool 35, forbidden_tool 5, wrong_answer 5 |
| basic (10 tasks) | B `mock:v2` | 90.0% (45/50) | 90.0% | 100.0% | 2.10 | 0.0822 | none | tool_error 5 |
| extended (46 tasks) | A `mock:v1` | 0.0% (0/230) | 0.0% | 15.2% | 1.02 | 0.3076 | none | missing_tool 195, max_steps 15, wrong_answer 15, tool_error 5 |
| extended (46 tasks) | B `mock:v2` | 4.3% (10/230) | 4.3% | 17.4% | 1.07 | 0.3029 | none | missing_tool 190, max_steps 15, wrong_answer 15 |

Latency is omitted from the table on purpose: the mocks run in-process against an
AST calculator and an in-memory keyword index, so the measured 0.0-0.2 ms per
trial measures Python, not an agent.

### A -> B, with the noise model switched on

`compare` over 5 trials per task:

| suite | fixed | regressed | p-value (Fisher exact) | verdict |
| --- | --- | --- | --- | --- |
| basic | 9 tasks, each 0/5 -> 5/5 | `calc_power` 5/5 -> 0/5 | 0.0079 both ways | `--fail-on-regression` exits 1 |
| extended | 2 tasks (`tool_error_caret`, `inject_roster`), each 0/5 -> 5/5 | none | 0.0079 | exits 1 on basic only |

Success-rate deltas: basic +80.0 pp, extended +4.3 pp. Cost moved
**-$0.0047** on extended (B answers unscripted tasks with a shorter sentence than
A does) while rising +$0.0250 on basic, where B actually does the work. That is
the shape a real evaluation should report: capability bought with steps.

## Three specific failure cases

Quoted verbatim from the stored trials (trial 0 of each task).

**1. `calc_power` under B - `tool_error`, and the only regression.**

```text
call 1: calculator({'expression': '2 ^ 10'}) -> is_error=True
        'error: unsafe expression (unsupported syntax: BitXor)'
answer: 'Cannot compute, the tool failed.'   steps=2   mode=tool_error
```

Why: B's script was rewritten to use the system prompt's `**` instruction, but
this one task still carries the `^` habit. In Python's AST `^` is bitwise XOR,
not exponentiation, so the whitelist rejects it rather than evaluating
something the caller did not intend. The classifier reports `tool_error`
(because the answer is also wrong) and the sandbox never saw an `eval`. This is
the case the regression gate exists for: 5/5 -> 0/5, p=0.0079, red build.

**2. `kb_intl_meals` under B - `missing_tool`, not a crash.**

```text
answer: 'Not sure, but I would guess: What is the daily meal reimburse...'
mode=missing_tool   steps=1
```

Why: no script exists for this task id, so the mock answers from its generic
fallback and never calls `kb_search`, which the task requires. The harness files
that as a tool-usage failure rather than an exception, which is what distinguishes
"the agent misbehaved" from "the agent broke". Across extended this dominates the
failure histogram (190 of 230 trials for B) - an honest signal that the mock
scripts only cover the basic suite, not a property of any real agent.

**3. `no_tool_weeks` under B - `max_steps`.**

```text
answer: 'Not sure, but I would guess: If a week has 7 days, how many d...'
mode=max_steps   steps=1 (task budget: 1)
```

Why: efficiency tasks are capped at one step, and the classifier walks failure
modes by severity, so "ran out of budget and still wrong" is labelled
`max_steps` before `wrong_answer`. 15 such trials = 3 efficiency tasks x 5
trials, identical in both configs. It is a deliberately conservative label: a
single-step budget that is fully consumed is worth seeing separately, even when
the answer was never going to be right.

## What this run does *not* establish

- **Flakiness.** The mocks are deterministic, so `flaky: none` above is a property
  of the stubs, not evidence that real runs are stable. The flakiness code path is
  exercised by `tests/test_significance.py` (a stub that passes 3 of 5 trials), and
  the noise floor is tested there too (3/5 -> 2/5 stays quiet, 5/5 -> 0/5 gates).
- **Real cost or latency.** Token counts are synthetic (`24 * rounds + 3 * words`).
- **Judge behaviour.** No `--judge` run happened (no key), so the three
  `match: judge` tasks in extended are recorded as `missing_tool`/`wrong_answer`
  here, and the bias questions in `docs/judge_notes.md` stay unquantified.
- **Whether `extra_body` reaches the API as `temperature`.** The SDK accepts the
  call shape; only a credentialed run can confirm the server honours it.

## Reproduce

```bash
pip install -e ".[dev]"
python -m pytest                                   # 211 tests
python -m ruff check .                              # clean

# the two configurations on both suites, 5 trials each
python -m agenteval.cli run --tasks tasks/basic.json    --agent mock:v1 \
  --trials 5 --input-price 3 --output-price 15 --out runs/stage8-basic-v1-trials5.json
python -m agenteval.cli run --tasks tasks/basic.json    --agent mock:v2 \
  --trials 5 --input-price 3 --output-price 15 --out runs/stage8-basic-v2-trials5.json
python -m agenteval.cli run --tasks tasks/extended.json --agent mock:v1 \
  --trials 5 --input-price 3 --output-price 15 --out runs/stage8-extended-v1-trials5.json
python -m agenteval.cli run --tasks tasks/extended.json --agent mock:v2 \
  --trials 5 --input-price 3 --output-price 15 --out runs/stage8-extended-v2-trials5.json

python -m agenteval.cli compare runs/stage8-basic-v1-trials5.json \
  runs/stage8-basic-v2-trials5.json --fail-on-regression   # exit 1

# once credentials exist, this is the run that replaces this whole page
export AGENTEVAL_MODEL=claude-sonnet-4-5
python -m agenteval.cli run --tasks tasks/extended.json --agent anthropic --system v1 \
  --judge --trials 5 --out runs/real-system-v1.json
python -m agenteval.cli run --tasks tasks/extended.json --agent anthropic --system v2 \
  --judge --trials 5 --out runs/real-system-v2.json
python -m agenteval.cli compare runs/real-system-v1.json runs/real-system-v2.json \
  --fail-on-regression
```
