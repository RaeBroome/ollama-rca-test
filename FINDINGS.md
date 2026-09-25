# Can a local model do root cause analysis?

Tested on [RCAEval](https://github.com/phamquiluan/RCAEval): 735 recorded microservice failures with metrics,
logs and traces, where the ground truth is the service a fault was injected into. Two local models on one
20 GB GPU — `qwen2.5-coder:7b` and `gemma4:26b` — against a deterministic Python baseline, and against
`claude-opus-5` as a ceiling reference.

**The answer, in one paragraph.** Given the same compressed evidence — a median of 43 lines describing what
changed after the fault — claude-opus-5 names the right service in **98%** of the clear-evidence cases, a
plain Python ranking gets **79%**, and the local models get **65–67%**. So the compression is not the
bottleneck: everything a diagnosis needs survives it, and a capable model reads it almost perfectly. The
local models are the bottleneck. Two further results hold at every model size: giving a model more of *our*
structure (rankings, role labels) makes it **worse**, not better; and no model, local or frontier, ever
declines to answer when the evidence does not support one.

---

## The question, and why it was worth asking

The appeal of a local model for incident diagnosis is that telemetry never leaves the building and there is
no per-incident cost. The obvious objection is capability. Between those two there is a design question that
matters more than either: **which parts of root cause analysis should a model do at all, and which should
ordinary code do first?**

A naive attempt fails immediately and instructively. Dump raw telemetry into a prompt and the model answers
with whichever service is *loudest*. In Sock Shop, `front-end` emits ~1,900 log lines a minute against
`carts`' ~180 — and `front-end` is almost always a victim, not a cause. One Sock Shop case is 84,665 log
lines plus a 1,441 × 81 metric table. Nothing useful happens until that is reduced.

So the work was: reduce the evidence with code, hand the model progressively more of the reduction, and
measure where the model stops helping.

## What was built

**Step 0 — compression (Python).** One case becomes a median of 43 lines (12–401 across cases; ~1,800
tokens) describing only what changed after the fault, in unit-free terms:

- **Metrics**: every metric of every service that moved — size as a fold change, shape as step /
  still-moving / jumped-then-recovered, plus *presence* (a metric appearing or going quiet) and
  *error-seconds* (sparse errors a median would hide).
- **Logs**: per-service volume and quiet periods; patterns that are new, vanished, or changed rate; one-off
  and rare lines with their time spans; exception and error class names never truncated.
- Anything dropped for size is announced in the text, never dropped silently.

It is deliberately ground-truth blind: no case name, no fault label, no marker, and service order shuffled,
because alphabetical order is not neutral.

**The pipeline.** Four steps, each with a Python implementation and an LLM implementation, so any step can be
swapped and the difference attributed: step 1 detect symptoms, step 2 trace symptom to origin (label-only —
it annotates, it never reorders), step 3 decide. Plus a **direct arm**: step-0 evidence straight to an
answer, no Python ranking at all.

**Scoring.** Exact match, case-insensitive, trimmed. Never substring — `"carts" in "carts-db"` is true, which
silently inflates accuracy. Cases are split into **clear** and **weak** evidence and always reported
separately. Every arm may answer `"none"`; abstentions are counted apart from wrong answers.

## Results

**Clear-evidence cases (43 of the 50-case stratified sample).** One call per case.

| what the model receives | accuracy | abstention | median time |
|---|---|---|---|
| **claude-opus-5** — plain evidence | **98%** (42/43) | 0% | 7.5 s |
| **Python baseline** — rank by significance, take the top | **79%** (34/43) | — | — |
| gemma4:26b — capped evidence | 72% | 7% | 3.8 s |
| qwen2.5-coder:7b — plain evidence | 67% (29/43) | 0% | 7.9 s |
| gemma4:26b — plain evidence | 65% (28/43) | 5% | 23.3 s |
| qwen2.5-coder:7b — capped evidence | 65% | 0% | 3.4 s |
| gemma / qwen — evidence + our role labels | 51% / 51% | 12% / 2% | 23.2 / 7.8 s |
| either model — a ranked candidate list from Python | 25–40% | — | — |

**Weak-evidence cases (7 in the same sample).** Small enough that one case is 14 points:

| | accuracy |
|---|---|
| gemma4:26b — plain evidence | 71% (5/7) |
| claude-opus-5 — plain evidence | 57% (4/7) |
| Python baseline | 43% (3/7) |
| qwen2.5-coder:7b — plain evidence | 43% (3/7) |

gemma's lead here did not reproduce: on **15 fresh weak cases** pulled from elsewhere in the benchmark it
scored 53% against the baseline's 47% — one case apart.

Claude's clear-case answers were a **strict superset of the baseline's**: 8 cases gained, none lost. Four of
the eight are cases where the baseline crowns a caller instead of the cause (`re3ss_orders_f1_1`,
`re3ss_orders_f3_1` → it said `front-end`; `re3ob_adservice_f3_1`, `re3ob_cartservice_f1_1` → `frontend`).
Two were missed by the baseline *and* both local models. Origin-versus-victim is exactly where ranking by
signal strength fails and reading the evidence wins.

## The three findings that matter

**1. The compression works, and it is sufficient.** This is what the ceiling arm buys. Every earlier result
was consistent with two very different stories: either step 0 throws away what a diagnosis needs, or a
7B–26B model cannot use what it keeps. A frontier model reading the *identical* text — prompt sizes matched
the local arm on all 50 cases — gets 42 of 43. The reduction from 84,665 log lines to 43 preserves the
answer. Effort spent on better compression has little left to win; effort spent on the model has ~30 points
available.

**2. More structure makes local models worse.** Each layer of our own interpretation cost accuracy, in a
consistent order:

| what the model receives | accuracy |
|---|---|
| raw evidence | 67% |
| evidence + the call graph as plain facts | identical answers on all 12 cases tested |
| evidence + dependency role labels (origin / victim), no ranking | 51% |
| a ranked candidate list from Python | 25–40% |

Facts are ignored; conclusions are harmful. Handing the model our ranking was the single worst thing we did
to it — it anchors on our answer and stops reading. This inverts the intuition that a small model needs more
help: it needs *less*, better arranged. (Measured directly: re-ranking step 1's output by step 2's reasoning
was better in 16 of 120 runs, unchanged in 53, worse in 51. Step 2 is label-only for that reason.)

**3. No model abstains, at any size.** Every arm can answer `"none"`, and the instruction to do so is in the
prompt. Across **42 weak-evidence decisions** by the local models: **zero abstentions**. qwen abstained once
in 129 clear-case calls. claude-opus-5: zero in 50 calls, including all 7 weak cases. When abstentions did
appear in the staged pipeline they were not selective — 14 of them landed on cases where the baseline would
have been wrong only 6 times, against a 48% base rate of being wrong. Confidence carries a little
information where abstention carries none (Claude: 28 of 29 `high` answers correct, 18 of 21 `medium`).

## What this means practically

**The approach is sound, and it improves for free.** The pipeline is model-agnostic: step 0 is deterministic
Python, and the model is a swap. The ceiling result says the scaffolding is already good enough for
near-perfect diagnosis on clear cases, so local capability is the only variable — and that is the variable
improving fastest. A stronger local model should land between 67% and 98% with no change to this repo.

**Today, air-gapping costs about 30 points** on clear cases (98% → 67%), and roughly 12 points against just
using deterministic Python (79% → 67%). If a local model is a hard constraint, the honest configuration is
**Python ranking as the primary answer** with the model used for explanation rather than selection — the
baseline is better, free, instant, and reproducible. If data can leave the building, a frontier model on
compressed evidence is both more accurate and cheap at this volume: 50 cases was 6.3 minutes and about $5 of
list-price equivalent.

**The abstention gap is the sharper operational risk.** A diagnostic tool that is wrong 20–35% of the time
but *never says so* is worse than its accuracy suggests, because every answer arrives in the same confident
tone. Nothing we tried fixed it by prompting. It has to be enforced outside the model: gate on the Python
evidence check (the baseline already knows when no candidate has clear evidence), or refuse to show a single
answer at all and show a ranked shortlist with the evidence attached. Treat any single-service answer from
any model as a hypothesis, never a verdict.

**One practical caution for anyone reproducing this.** Most of the wall-clock time is not inference. Ollama
rebuilds its runner whenever the context size changes — ~18 s for a 26B model against 0.2 s for the same call
at an unchanged size — and since context is sized per case, most calls pay it. The 50-case two-model run took
2h13m, mostly reloading. Coarse context buckets were tried and reverted: they saved ~26% and changed an
answer.

## Limitations, and what was not tested

- **Sample sizes are small.** 43 clear and 7 weak cases in the main run, out of RCAEval's 735. One case is 2
  points at 43 and 14 at 7. Only the large gaps — the ceiling result, the structure gradient — are beyond
  noise. The 12-case preset used for smoke tests is not a basis for any conclusion: both of its headline
  results reversed at 50 cases.
- **RCAEval's own methods were never run.** This evaluates local models on RCAEval's *data*. It is not a
  comparison against the benchmark's published baselines, and none of its authors' techniques were
  implemented here. Nothing here is a claim about RCAEval itself.
- **The Claude arm is not temperature-pinned.** The Claude Code CLI exposes no temperature, top-p or seed, so
  that arm alone could not be fixed at temperature 0 the way every local arm is; `--effort` was fixed at
  `medium` and recorded instead, and the call carries a small CLI scaffold rather than being a bare chat
  request. Read 98% as a ceiling measured once, not a controlled comparison. An exact apples-to-apples run
  needs the API at temperature 0.
- **Temperature 0 is not determinism anyway.** Changing only the context size — same prompt, same model —
  flipped one answer of 24 (qwen on a weak case, `orders` → `orders-db`, both wrong). Runtime configuration
  is part of the input, so replication needs the same context sizing, not just temperature 0.
- **Two models, one machine.** RTX A4500 20 GB; qwen fits in ~5.4 GB, gemma needs ~19 GB, so they cannot be
  resident together. No 70B-class local model was tested, and gemma's thinking mode was unusable here (52k
  characters of reasoning, no answer), so "gemma" means gemma-no-think throughout.
- **Thresholds are provisional** and deliberately untuned on the cases inspected. The detection limit for a
  slow drift over a 360 s window is roughly +15% on a quiet metric, worse on a bursty one.
- **One shuffle seed** for service order. Ordering was tested and mattered little, but was not averaged over
  seeds.
- **A likely injection artifact is still in the evidence.** `{svc}_diskio` appearing after injection
  identifies the root-cause service in 17 of 18 cases, including code-level faults, which suggests it
  reflects the redeploy mechanism rather than the fault. It is flagged in the evidence, and scores have
  **not** yet been recomputed with it removed — that comparison is owed.
- **Sock Shop has no traces**, so its dependency graph is inferred from services naming each other in logs.
- **Excluded cases**: two with broken injection timestamps, one whose label evidence duplicates another
  case's, and cases where no metric, log or trace shows any shift (listed in `explore.ipynb` section 7).

---

# The detail

Everything below is supporting material for the sections above.

## The traps in the obvious setup

- **Substring scoring inflates accuracy.** `"carts" in "carts-db"` is true. Scoring is exact match,
  case-insensitive, trimmed.
- **Keyword filtering destroys the evidence.** Across the 8 RCAEval cases that ship a labelled root-cause log
  line, an `Exception|Error` filter kept **0 of 8**. The real markers are a `WARN … PageNotFound` line and
  HTTP 500 access-log lines. What the filter *does* keep is 81–94% background noise per case: a queue-master
  socket exception occurring at the same rate before the fault.

## Two compression lessons, learned the hard way

Metric shape must be measured *after* injection — an earlier version computed drift before it, which cannot
see a fault that ramps up afterwards. And compression tuned for the model destroyed signals our own later
rules needed: templating masked `"msg":"connection accepted"`, so a rule keyed on connection churn could
never fire. The governing principle: filtering must not collapse subtle symptoms — slow drifts, rate changes,
services going quiet, WARN lines, sparse errors, one-off lines, exception class names.

## The ceiling-reference arm in full

The same evidence, instructions and scoring path, answered by claude-opus-5 through the Claude Code CLI on a
subscription login (no API key). Blinding was verified rather than assumed: tools removed
(`--disallowed-tools "*"` drops the tool definitions, not just their permission — the prompt falls from
~24.5k to ~4.5k tokens), the CLI run from an empty directory outside the repo so `CLAUDE.md` is never
discovered, MCP servers off, Claude Code's own system prompt replaced, and one fresh process per case so no
case can inform another. Asked in that exact configuration what context it had, the model reported no project
instructions, no memory files, no repository listing and nothing about RCAEval; the residue is an environment
block naming the sandbox directory. Prompt sizes matched the local plain arm on all 50 cases.

Its one clear-case miss is a good failure: on `re1tt_ts-auth-service_loss_1` it answered
`ts-contacts-service`, arguing that service showed still-climbing CPU saturation while its callers showed
only downstream latency step-ups. Correct reasoning pattern, wrong service.

## Capped evidence costs no accuracy

Cutting the evidence to ~10 log-pattern rows (~2k tokens) *raised* gemma's accuracy by 7 points and cost qwen
2. It is also the only configuration that fits TrainTicket cases, whose full evidence reaches ~25k tokens.

The capped arm also *looked* 3–6× faster, and that part was an artefact of the runner rebuild described
above: similar prompt sizes round to the same context value about twice as often (54% vs 26% reuse) and skip
the reload. gemma calls where the context changed took a median of 25.7 s, against 3.0 s where it did not.
Compression buys accuracy parity and fit, not speed.

## Where tracing breaks down

"Deepest failing node" has nothing to bite on when most services are symptomatic, and it will crown a
database that is merely logging connection churn caused by its caller restarting. Log-derived edges need care
too: `"Creating item for user: …"` is prose, not a call, and produced 1,456 false pieces of evidence before
edges were restricted to call-shaped contexts.

## Reproducing any of this

```bash
python run_sample.py --quickstart --models <model>   # check setup, time one case, print estimates
python run_sample.py --no-llm --label baseline       # the Python arms only
python run_sample.py --preset50 --direct plain capped roles --direct-only --label direct50
python run_sample.py --preset50 --no-llm --claude opus --label ceiling
python analyse_run.py results/<run>                  # accuracy by arm, never pooled
python compare_claude_arm.py results/<run>           # the ceiling comparison, clear and weak separate
```

Every run writes its own committed folder under `results/` with per-case records, the step versions,
thresholds, order seed, timings and the git commit it ran at. See [README.md](README.md) for the layout, and
note that any change to step 0 invalidates earlier step-1/2/3 numbers: keeping a few more log fields moved
qwen's symptom-detection score by 10 points.
