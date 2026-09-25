# Findings

Can a local model (qwen2.5-coder:7b, gemma4:26b) find the root cause of a microservice failure? Tested on
[RCAEval](https://github.com/phamquiluan/RCAEval): 735 failure cases across three systems, each with metrics,
and depending on the suite logs and traces. Ground truth is the service the fault was injected into.

**Short answer: no.** A deterministic Python ranking beat every *local* model configuration tried, and the
models got worse the more structure we gave them. A frontier model (claude-opus-5) on the same evidence got
98% of the clear cases, which locates the problem: the compressed evidence is sufficient, the small local
model is not. Sample sizes are small — read the numbers with the counts attached.

## The one-shot failure

Dumping raw telemetry into a prompt fails for a reason worth stating: the model answers with whichever
service is *loudest*. In Sock Shop, front-end emits ~1,900 log lines a minute against carts' ~180, so
front-end dominates any raw dump. It is almost always a victim.

Two further traps in the obvious setup:

- **Substring scoring inflates accuracy.** `"carts" in "carts-db"` is true. Scoring is exact match,
  case-insensitive, trimmed.
- **Keyword filtering destroys the evidence.** Across the 8 RCAEval cases that ship a labelled root-cause log
  line, an `Exception|Error` filter kept **0 of 8**. The real markers are a `WARN … PageNotFound` line and
  HTTP 500 access-log lines. What the filter *does* keep is 81-94% background noise per case: a
  queue-master socket exception that occurs at the same rate before the fault.

## What step 0 does

Step 0 compresses one case into ~2–3k tokens of evidence, ground-truth blind (no case name, fault label or
root-cause marker; service order shuffled, since alphabetical order is not neutral):

- **Metrics**: every metric of every service whose behaviour changed, described in unit-free terms — size as
  a fold change, shape as step / still-moving / jumped-then-recovered, plus *presence* (a metric appearing or
  going quiet) and *error-seconds* (sparse errors a median would hide).
- **Logs**: per-service volume and quiet periods; patterns that are new, vanished or changed rate; one-off
  and rare lines summarised with time spans; exception and error class names always kept.
- Anything dropped for size is announced, never dropped silently.

Two compression lessons, both learned the hard way. Metric shape must be measured *after* injection —
an earlier version computed drift before it, which cannot see a fault that ramps up afterwards. And
compression tuned for the model destroyed signals our own later rules needed: templating masked
`"msg":"connection accepted"`, so a rule that keyed on connection churn could never fire.

## Results: 50 cases

Accuracy on the **43 clear-evidence cases** (one call per case unless noted):

| configuration | accuracy | abstention | median time |
|---|---|---|---|
| **Python baseline** (rank by significance, take the top) | **79%** (34/43) | — | — |
| gemma capped evidence | 72% | 7% | 3.8 s |
| qwen plain evidence | 67% | 0% | 7.9 s |
| gemma plain evidence | 65% | 5% | 23.3 s |
| qwen capped evidence | 65% | 0% | 3.4 s |
| gemma / qwen + role labels | 51% / 51% | 12% / 2% | 23.2 / 7.8 s |
| staged pipeline (symptom list → tracing → decision) | 25–40% | — | — |

On the **7 weak-evidence cases** in that sample gemma led (71% vs the baseline's 43%), but on **15 fresh weak
cases** drawn from elsewhere in the benchmark it was 53% vs 47% — one case apart. The lead did not reproduce.

An earlier 12-case run had qwen at 80% against a 70% baseline. That was 8 correct versus 7. At 43 cases it is
67% versus 79%.

## A frontier model as a ceiling reference: the evidence was never the problem

The same step-0 evidence, the same instructions, the same exact-match scoring, answered by **claude-opus-5**
through the Claude Code CLI on a subscription login (no API key). 50 cases, one fresh blinded call each:

| configuration | clear (43) | weak (7) | abstention | median time |
|---|---|---|---|---|
| **claude-opus-5, plain evidence** | **98%** (42/43) | 57% (4/7) | 0% | 7.5 s |
| Python baseline (control) | 79% (34/43) | 43% (3/7) | — | — |
| qwen2.5-coder:7b, plain evidence | 67% (29/43) | 43% | 0% | 7.9 s |
| gemma4:26b, plain evidence | 65% (28/43) | 71% (5/7) | 5% | 23.3 s |

**This answers the project's central question.** Every earlier result was consistent with two very different
explanations: either step 0 throws away what a diagnosis needs, or a 7B-26B local model cannot use it. On
the clear cases a frontier model reads the same compressed evidence and gets 42 of 43. The evidence is
sufficient; the limit was the local model.

Its answers were a strict superset of the control's: **8 clear cases it got that the control missed, and
none the control got that it missed.** Two of those eight were missed by the control *and* both local
models — `re1ss_carts_loss_1` (control and qwen said `orders`) and `re2ob_checkoutservice_delay_1` (control
said `emailservice`, both local models said `recommendationservice`). Four of the eight are cases where the
control crowns a caller or a database: `re3ss_orders_f1_1` and `re3ss_orders_f3_1` (control: `front-end`),
`re3ob_adservice_f3_1` and `re3ob_cartservice_f1_1` (control: `frontend`). Origin-versus-victim is exactly
where a ranking by signal strength fails and reading the evidence wins.

The single clear-case miss is instructive rather than sloppy: on `re1tt_ts-auth-service_loss_1` it answered
`ts-contacts-service`, arguing that service showed still-climbing CPU saturation while its callers showed
only downstream latency step-ups. That is the correct reasoning pattern applied to a service that was not
the injection point.

**Abstention fails here too.** Zero abstentions in 50 calls, including all 7 weak-evidence cases. Confidence
carries a little information where abstention carries none: 28 of 29 `high` answers were right against 18 of
21 `medium`. Every model we have tried, local or frontier, answers rather than declines.

### This arm is not like-for-like, and the reason is worth stating

The Claude Code CLI exposes no temperature, top-p or seed. **This arm cannot be pinned to temperature 0, as
every Ollama arm is.** `--effort` is fixed at `medium` and recorded, but the arm is sampled at whatever
defaults the CLI uses, and the call also carries a small Claude Code scaffold rather than being a bare chat
request. Combined with the earlier finding that temperature 0 is not itself determinism, treat 98% as a
ceiling reference measured once, not a controlled comparison against the local numbers. An exact
apples-to-apples run needs the API with temperature 0.

What it is blinded against was checked rather than assumed: tools removed (`--disallowed-tools "*"`, which
drops the tool definitions rather than only denying them), the CLI run from an empty directory outside the
repo so `CLAUDE.md` is never discovered, no MCP servers, Claude Code's own system prompt replaced, and one
fresh process per case so no case can inform another. Asked in that exact configuration what context it
had, the model reported no project instructions, no memory files, no repository listing and nothing about
RCAEval. The residue is an environment block naming the sandbox directory. Prompt sizes matched the local
plain arm on all 50 cases, so the evidence really was identical.

Cost, for scale: 50 calls, 6.3 minutes of model time, about $5 of list-price equivalent consumed from a Max
subscription. The local arms are free and run offline, which is the whole point of asking whether they are
good enough.

## The models get worse as we add structure

Each layer of our own interpretation cost accuracy, in a consistent order:

| what the model receives | accuracy |
|---|---|
| raw evidence | 67% (43 cases) |
| evidence + the call graph as plain facts | **identical answers on all 12 cases tested** |
| evidence + dependency role labels (origin / victim), no ranking | 51% |
| a ranked candidate list from Python | 25–40% |

So **facts are ignored and conclusions are harmful**. Handing the model our ranking was the worst thing we
did to it; handing it who-calls-whom changed nothing at all, on any of the 12 cases.

## Abstention never fires when it should

Every arm could answer "none". Across **42 weak-evidence decisions** in the 50-case run — both models, three
evidence variants — there were **zero abstentions**. qwen abstained once in 129 clear-case calls. When
abstentions did happen, they were not selective: in the staged run, 14 abstentions landed on cases where the
baseline would have been wrong only 6 times, against a 48% base rate of it being wrong. The instruction is
present in the prompt and these models do not act on it.

## Capped evidence costs no accuracy

Cutting the evidence to ~10 log-pattern rows (~2k tokens) *raised* gemma's accuracy by 7 points and cost qwen
2. It is also the only configuration that fits TrainTicket cases, whose full evidence reaches ~12k tokens.

The capped arm also *looked* 3–6× faster, and that part was an artefact. Ollama rebuilds the runner whenever
`num_ctx` changes — a full model reload, ~18 s for gemma4:26b against 0.2 s for the same call at an unchanged
`num_ctx`. Capped prompts are similar in size, so they round to the same context value about twice as often
(54% vs 26% reuse) and skip the reload. Measured over the 50-case run: gemma calls where `num_ctx` changed
took a median of 25.7 s, and 3.0 s where it did not. **Compression buys accuracy parity and fit, not speed.**

## Two things to know before reproducing these numbers

**Temperature 0 is not determinism.** Changing only `num_ctx` — same prompt, same model, same seed-free
settings — flipped one answer of 24: qwen on `re3ss_carts_f4_1` moved from `orders` to `orders-db` (both
wrong; the truth is `carts`). 23 of 24 were identical. It happened on a weak-evidence case where the model
had no strong preference, which is where such flips should be expected. Runtime configuration is part of the
input, so an exact replication needs the same context sizing, not just temperature 0.

**Most of the wall time is model reloads, not inference.** Because context is sized per case, most calls
change `num_ctx` and pay the rebuild. The 50-case run's 2h13m is therefore mostly reloading: for gemma,
93 of 150 calls changed `num_ctx` at a median 25.7 s each, against 3.0 s for the 57 that did not. Coarse
context buckets were tried and reverted — they saved ~26% but changed that one answer, and a runtime setting
that alters output is not worth the time.

## Known limitations

- **Sample sizes.** 43 clear and 7 weak cases in the main run; 12 cases for the facts test; 15 for the fresh
  weak cases. One case is 2 points at 43, 8 points at 12. Only the large gaps are beyond noise.
- **Thresholds are provisional** and deliberately untuned on the inspected cases. The detection limit for a
  slow drift over a 360 s window is roughly +15% on a quiet metric, worse on a bursty one.
- **Sock Shop has no traces**, so its dependency graph comes from services naming each other in logs. That
  needs care: "Creating item **for user**: …" is prose, not a call, and produced 1,456 false pieces of
  evidence before edges were restricted to call-shaped contexts.
- **Tracing can't discriminate when most services are symptomatic** — "deepest failing node" has nothing to
  bite on, and it crowns a database that is merely logging connection churn caused by its caller restarting.
- **`{svc}_diskio` appearing after injection** identifies the root-cause service in 17 of 18 cases, including
  code-level faults, which suggests it reflects the injection mechanism (a redeploy) rather than the fault.
  It is flagged so scores can be run with and without it.
- **Excluded cases**: two with broken `inject_time`, one whose label evidence is a copy of another case's,
  and cases where no metric, log or trace shows any shift — listed in `explore.ipynb` section 7.
- **gemma's thinking mode is unusable here** (52k characters of reasoning, no answer); it runs with thinking
  off, so "gemma" throughout means gemma-no-think.
- **One shuffle seed** was used for service order. Ordering was tested and mattered little, but it was not
  averaged over seeds.
- Any change to step 0 invalidates earlier step-1/2/3 numbers: keeping a few more log fields moved qwen's
  symptom-detection score by 10 points.
