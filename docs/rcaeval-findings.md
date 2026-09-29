# How this repo uses RCAEval

An inspection of `ollama-rca-test` as it stands on `main` at commit `90347db`, answering seven questions about
the data, the pipeline and the weak points. Nothing here changes code; every figure was recomputed from the
repo and the read-only dataset while writing it.

Results and conclusions live in [FINDINGS.md](../FINDINGS.md); this is about the machinery underneath them.

---

## 1. Datasets and systems

All three RCAEval suites and all three systems are used. The index (`RCAEval-data/cases.parquet`) holds **735
cases** in nine dataset/system combinations:

| dataset | cases | faults | logs | traces | `root_cause.txt` |
|---|---|---|---|---|---|
| RE1-OB | 125 | cpu, delay, disk, loss, mem | 0 | 0 | 0 |
| RE1-SS | 125 | cpu, delay, disk, loss, mem | 0 | 0 | 0 |
| RE1-TT | 125 | cpu, delay, disk, loss, mem | 0 | 0 | 0 |
| RE2-OB | 90 | + socket | 90 | 90 | 0 |
| RE2-SS | 90 | + socket | 90 | **0** | 0 |
| RE2-TT | 90 | + socket | **89** | 90 | 0 |
| RE3-OB | 30 | f1–f5 | 30 | 30 | 0 |
| RE3-SS | 30 | f1–f4 | 30 | **0** | **8** |
| RE3-TT | 30 | f1–f4 | 30 | 30 | 0 |

Systems: **Online Boutique** (`ob`), **Sock Shop** (`ss`), **Train Ticket** (`tt`). Three facts that shape
everything downstream:

- **RE1 is metrics only.** No logs, no traces, for all 375 cases. A case whose metric evidence is weak has
  nothing else to fall back on.
- **Sock Shop has no traces in any suite.** Its dependency graph is inferred from services naming each other
  in log lines, plus a hand-entered static architecture (`STATIC_EDGES`).
- **RE3-OB has an `f5` fault class that SS and TT do not**, and only **8 of the 30 RE3-SS cases** ship a
  labelled root-cause log line.

The 50-case stratified sample (`rca_lib.SAMPLE50`) covers all nine combinations — 6 cases each from the RE1
and RE2 datasets, 4–6 from RE3. The 12-case smoke sample (`SAMPLE12`) is deliberately unbalanced (6 of its 12
are RE3-SS). Across all 13 committed runs, **65 distinct cases** have ever been scored, out of 735.

## 2. The data files, per case

Every case is a directory `RCAEval-data/<case>/`. Everything is **Parquet**, except the two text files. The
upstream RCAEval distribution ships CSV; this repo's copy was converted before use, and `RCAEval-data/` is
read-only and git-ignored.

| what | file | format | read by |
|---|---|---|---|
| metrics | `metrics.parquet` | Parquet, wide | `rca_lib.load_metrics()` → `pd.read_parquet` |
| logs | `logs.parquet` | Parquet, long | `rca_lib.load_logs()` → `pd.read_parquet`, then adds `rel`, `null_message`, `tmpl` |
| traces | `traces.parquet` | Parquet, span rows | `pd.read_parquet` with an explicit column subset in `trace_edges()` / `case_call_graph()` |
| injection time | `inject_time.txt` | plain text, one Unix second | `rca_lib.load_inject_time()` → `int(f.read().strip())` |
| labelled root-cause line | `root_cause.txt` | one CSV row | `rca_lib.read_root_cause()` → `csv.reader`, first row |
| case index | `cases.parquet` (dataset root) | Parquet, 735 × 22 | `rca_lib.load_index()`, `lru_cache(maxsize=1)` |

**Metrics** are wide: one row per second, one column per `<service>_<metric>`, plus `time`. For
`re3ss_carts_f1_1` that is **1,441 rows × 81 columns** (80 metrics + `time`). Metric kinds are resolved
through `COL_FALLBACKS`, because naming differs between systems (`latency-50` / `latency-90`, `diskio`,
`socket`, `error`, `workload`).

**Traces** carry `traceID`, `spanID`, `parentSpanID`, `serviceName`, `methodName`, `operationName`,
`startTime`, `duration`, `statusCode` — 391,053 spans for `re2ob_checkoutservice_disk_1`. Only the parent/child
service pairs are used, to build a call graph; span timings are not used for scoring.

**One case in RE2-TT has no logs** although the other 89 do. Code paths key off the index's `has_logs` /
`has_traces` flags rather than probing the filesystem, so this is handled, but it is a reminder that "RE2 has
logs" is not quite true.

## 3. The compression step (step 0)

**In:** one case. For the Sock Shop example in §4 that is 84,665 log lines and a 1,441 × 80 metric table.
**Out:** a plain-text evidence block, median **43 lines / ~1,810 tokens** across the 50-case sample (range
12–401 lines, 301–25,443 tokens).
**Why:** a raw dump makes a model answer with whichever service is *loudest*. In Sock Shop `front-end` emits
~1,900 log lines a minute against `carts`' ~180, and `front-end` is almost always a victim. Nothing useful
happens until the input is reduced to *what changed*.

`render_step0()` (version `step0-v0.2`) emits, in order:

1. **Header** — system name, the service list, how many seconds of normal and faulty data, and a statement
   that nothing is ranked.
2. **METRICS that changed** — one row per changed metric: service, metric, change shape in words (step /
   still moving / jumped then partly recovered / gradual), direction, size as a fold change, when it settled,
   presence change, error-seconds, and `clear` vs `weak`. Sub-threshold movements are collapsed into a single
   `Weak changes (near a detection threshold)` line. Services with no metric change are named explicitly, so
   "quiet" is visible rather than absent.
3. **LOG VOLUME per service** — lines/minute before → after, plus quiet periods (<25% of normal rate for 30 s+).
4. **LOG PATTERNS** — templated patterns that are new, vanished, or changed rate ≥3× (clear) or 2–3× (weak),
   with counts before/after and first appearance. All log levels; **no keyword filter**.
5. **One-off and rare lines** — grouped, with time spans, examples sampled across the span, and exception or
   error class names always preserved.
6. **OMITTED line** — anything dropped for size is announced, never dropped silently.

Two configurations are run: **unranked** (`max_pat_rows=None`, nothing cut, context sized per case) and
**capped** (`CAPPED_PAT_ROWS = 10` pattern rows, ~2k tokens, chosen round-robin one per service so a single
noisy service cannot fill the budget).

The text is deliberately **ground-truth blind**: no case name, no fault label, no root-cause marker, and the
service order is shuffled per case with `DEFAULT_ORDER_SEED = 1234`, because alphabetical order is not
neutral. Thresholds live in `THRESHOLDS` (`shape-v2.3`) and `EVIDENCE_THRESHOLDS`, are marked provisional, and
a value within `BORDERLINE_MARGIN` (20%) of a threshold is labelled `borderline` rather than forced either way.

## 4. Log data for one case

`re3ss_carts_f1_1` — **84,665 log lines**, three columns:

| column | dtype |
|---|---|
| `timestamp` | `Int64` (Unix seconds) |
| `container_name` | `string` |
| `message` | `string` |

Five rows exactly as stored (messages truncated here for width):

| timestamp | container_name | message |
|---|---|---|
| 1732242483 | front-end | `Posting Address: {"number":"123","street":"123","city":"123","post…` |
| 1732242483 | user | `ts=2024-11-22T02:28:03.04841895Z caller=middlewares.go:88 method=P…` |
| 1732242483 | front-end | `POST /addresses 200 - ms - -` |
| 1732242483 | front-end | `Posting Card: {"longNum":"123123321","expires":"1232","ccv":"123",…` |
| 1732242483 | user | `ts=2024-11-22T02:28:03.063549655Z caller=middlewares.go:118 method…` |

Notes that matter for processing: the timestamp is **whole seconds**, so ordering within a second is not
recoverable; `container_name` is the only service identifier (there is no pod or node column here, unlike
`root_cause.txt`); and **one line in this case has a null message**, which `load_logs()` marks with
`null_message` and keeps as a row rather than dropping — a filter must never match it silently.

`load_logs()` adds three derived columns: `rel` (seconds relative to injection), `null_message`, and `tmpl`
(the message reduced to a template by `_TEMPLATE_RULES`, which strips timestamps, thread names, hex ids and
UUIDs but **keeps** HTTP status codes, `msg`, `exception` and `statusCode` fields — `ALWAYS_KEEP_KEYS`).

## 5. Where the ground truth lives

| what | where | how it is read |
|---|---|---|
| root cause **service** | `cases.parquet` column `root_cause_service` | `load_index()` / `case_info(case)` |
| injection time | `inject_time.txt` per case, and `inject_time` in the index | `load_inject_time()` reads the file; the index copy is used for bulk summaries |
| fault kind | index columns `fault`, `fault_description` (e.g. `cpu` / "CPU stress") | `case_info(case)` |
| labelled root-cause log line | `root_cause.txt`, only 8 RE3-SS cases | `read_root_cause()` → `{hhmm, ts, container, message, pod}` |
| data shape | index columns `n_metrics`, `n_timesteps`, `normal_timesteps`, `faulty_timesteps`, `has_logs`, `has_traces`, `has_root_cause_file` | `case_info(case)` |

**There is no root-cause *indicator* column.** RCAEval's own evaluation scores (service, indicator) pairs;
this index carries only the service, so everything here is service-level. Indicator-level scoring is not
possible from this data as converted, which also means none of these numbers are comparable with RCAEval's
published per-indicator results.

Ground truth is kept out of the model's input in one place — `render_step0()` — and checked by a leak test
described in §6. Two exclusion lists are static in `rca_lib.py`:

- `EXCLUDE` (broken data, no usable analysis): `re1ob_currencyservice_loss_1` (inject_time `16933142`, missing
  digits, so there is no normal period) and `re1ob_productcatalogservice_cpu_3` (inject_time after the last
  metric row).
- `BROKEN_LABELS` (data fine, label evidence broken): `re3ss_front-end_f2_2`, whose `root_cause.txt` is a
  byte-for-byte copy of `re3ss_front-end_f2_1`'s, timestamped 3 hours before its own injection.

A third category — "not diagnosable from available data" — is computed rather than listed, by
`diagnosability_status()`.

## 6. The current flow

```
cases.parquet ─┐
metrics.parquet ├─► step 0: render_step0()  →  evidence text (ground-truth blind, shuffled order)
logs.parquet   ─┤            step0_metrics(), step0_logs()
traces.parquet ─┘
                          │
        ┌─────────────────┼──────────────────────────────┐
        ▼                 ▼                              ▼
   step 1 symptoms   direct arm (one call)          claude arm (ceiling)
   step1_python()    direct_llm(): evidence → answer  claude_direct() via the
   step1_llm()       variants: plain / capped /       Claude Code CLI, blinded
        │            roles / facts                    (tools removed, cwd outside
        ▼                                             the repo, one process per case)
   step 2 tracing — step2_python(rule=naive|rule) or step2_llm()
   LABEL ONLY: attaches role / path / direction_evidence, never reorders.
   would_demote records what re-ranking would have done.
        │
        ▼
   step 3 decision — step3_python(rule=top1|role) or step3_llm()
        │
        ▼
   run record: results/<timestamp>-<label>/ written incrementally as jsonl,
   converted to parquet by RunWriter.finalize() plus summary.csv and metadata.json
```

Orchestration is `run_sample.py` (`--no-llm`, `--cases`, `--preset50`, `--direct`, `--direct-only`,
`--claude`, `--keep-warm`, `--quickstart`). Analysis is `analyse_run.py` (accuracy per arm, never pooled
across step-1 sources), `compare_claude_arm.py`, `analyse_diskio.py`, `scan_diskio.py` and
`interrogate_run.py`.

**What the manual validation checks.** `python rca_lib.py inspect <case>` (the `inspect-case` skill) prints,
for one case:

1. **Labels** — dataset, fault kind and description, ground-truth service, and which data types exist.
2. **Exclusions and problems** — membership of `EXCLUDE` or `BROKEN_LABELS`; the `diagnosability_status()` of
   the true root cause's own metrics plus its callers' latency (`diagnosable` / `weak evidence only` / `not
   diagnosable from available data`), with a reminder to check logs and traces before excluding; and whether
   the root cause's evidence includes the `diskio_appears` artifact flag.
3. **The truth's evidence, explicitly marked as not shown to the model** — every clear or weak signal on the
   root-cause service and its callers, so a human can see whether the case is answerable at all.
4. **The labelled log line** from `root_cause.txt` where one exists, with its offset from injection.
5. **The exact step-0 text a scoring run would send**, with token counts for both tokenizers, the `num_ctx`
   that implies, and a **leak check** that searches the rendered text for the case name, the fault label,
   `root_cause` and `ground truth`.

Alongside that, `auto_retention_checks()` derives a per-case checklist from the true root cause's own
evidence — preferring the signal kinds earlier analysis showed are easiest to lose (presence changes,
error-rate rises, new log patterns, then the strongest metric shift) — and every run records, per expected
item, whether it survived compression (`retention.parquet`, one row per item with `kept` true/false).

## 7. What looks wrong, fragile, or unfinished

**Wrong, or at least circular**

- **The clear/weak split is defined by our own compression.** `diagnosability_status()` asks whether step 0
  gave the *true root cause* a clear signal, so "weak evidence" means "our pipeline failed to surface the
  answer", not "the telemetry was faint". Every weak-case number is therefore partly a measurement of the
  classifier. Worse, in 2 of the 7 weak cases in the sample the **largest signal in the whole case** belonged
  to the root cause and was demoted to a footnote (`re1ob_adservice_loss_1`, z=3571, ×36.71;
  `re1ob_cartservice_loss_4`, z=1366, ×30.31). Open as [issue #1](https://github.com/RaeBroome/ollama-rca-test/issues/1).
- **Six figures in FINDINGS cannot be recomputed** from committed code, because the code or configuration that
  produced them is gone (the 1,456 false log edges, the context-bucket saving and the answer it flipped,
  gemma's 52k characters of thinking, the hand-timed 18 s reload, the GPU footprints, the +15% drift limit).
  They are now labelled as one-off observations. This was found after a seventh such figure — `diskio`
  identifying the root cause "in 17 of 18 cases" — turned out to be 17 of **25** when finally measured.
- **`run_test.py` still scores with a substring check** that inflates accuracy (`"carts" in "carts-db"`). It
  is kept deliberately, as the evidence for that trap, but it is a loaded gun in the repo root.

**Fragile**

- **The Python phase costs ~44 s per case and nothing is cached.** Step 0 re-reads `metrics.parquet` and
  `logs.parquet` and recomputes every shape for each call, and a run calls it repeatedly (two step-1 orders,
  two step-2 rules). The 50-case ceiling run took 48 minutes, of which the model calls were 6.3. The
  step-0 output is deterministic given its version, so it is trivially cacheable; it just is not.
- **Ollama rebuilds its runner whenever `num_ctx` changes** (~18 s for a 26B model), and context is sized per
  case, so most calls pay it. Coarse buckets were tried and reverted because they changed an answer — so the
  cost is deliberate, and any timing comparison between arms is really a comparison of how often their prompt
  sizes collide.
- **Temperature 0 is not determinism.** Changing only `num_ctx` flipped one answer of 24. Runtime
  configuration is part of the input.
- **The Claude arm cannot be pinned to temperature 0** — the CLI exposes no temperature, top-p or seed — and
  it **records no `signals` array**, so its 42/43 has no auditable chain of reasoning
  ([issue #3](https://github.com/RaeBroome/ollama-rca-test/issues/3)).
- **Sock Shop's dependency graph is inferred from prose.** Log-derived edges were restricted to call-shaped
  contexts after `"Creating item for user: …"` produced over a thousand false edges; the remaining heuristic
  is still a heuristic, and `STATIC_EDGES` is hand-entered from published architecture diagrams.
- **External tools are assumed present but degrade untidily.** `nvidia-smi` (or `rocm-smi`, untested) for GPU
  memory, `/api/ps` documented as unreliable on 0.34.x, and the Claude arm needing the `claude` CLI logged in.
  Each path reports rather than raises, but only `nvidia-smi` has been exercised here.
- **There are no automated tests.** No `pytest`, no fixtures, no CI. Correctness rests on `inspect` output
  being read by a human and on run records being compared by hand.

**Unfinished**

- **`results/` contains `qwen3.6:27b` runs from 2026-09-25** (1-case and 12-case). Those runs are not
  reflected in FINDINGS, and the 12-case preset they used is explicitly not a basis for conclusions.
  `MODELS_DEFAULT` is now `("gemma4:26b",)`; the committed two-model runs name both models with `--models`.
- **The LLM arms were never re-scored without the `diskio` artifact.** The Python counterfactual is exact and
  showed no change, but re-scoring a model means re-asking it with different evidence, which is a step-0
  change; it was left for whenever #1 forces a re-baseline anyway.
- **Step 1 as an LLM task is measured but unused.** The staged chain scores 20–40% against the baseline's 79%,
  so nothing downstream relies on it; the code path remains.
- **One shuffle seed** (`DEFAULT_ORDER_SEED = 1234`). Position bias was tested once and mattered little, but
  no result here is averaged over seeds.
- **The two earliest run folders have no git block** in `metadata.json`; they predate `git_state()`.
- **Provenance of the notebook is weak.** `explore.ipynb` is where the exclusions, the thresholds and several
  published observations were derived, and its outputs are cleared in the committed file by policy. That is
  good hygiene for diffs and bad for reproducibility: the analysis that justified a number is not recoverable
  from the repo unless a script also does it.
