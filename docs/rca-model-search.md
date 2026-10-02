# RCA local-model search — run tracker

**Goal:** find one local model that scores **≥75% on the 43 clear-evidence cases** of the 50-case
sample. When one clears it, stop. Judge every run on the 50-case score — never on a single case.

## How to read this sheet

Each row is one planned or completed run. The columns are grouped: identity, then the **variables**
you set before the run, then the **results** you fill in after.

| column | meaning |
|---|---|
| **Run ID** | date + short label, matches the `results/<timestamp>-<label>/` folder |
| **Model** | the Ollama model under test |
| *p1 · thinking* | variable 1 — reasoning on or off |
| *p2 · num_ctx* | variable 2 — context rule (`num_ctx_for` = per-case sizing; or a fixed number) |
| *p3 · schema* | variable 3 — structured-output JSON schema on or off |
| *p4 · order* | variable 4 — service-order shuffle seed (repo default 1234) |
| **Status** | planned / running / done / failed |
| **Clear %** | accuracy on the 43 clear cases — **this is the number that decides ≥75%** |
| **Weak %** | accuracy on the 7 weak cases (context only; not the bar) |
| **Loops** | cases that ran away (`length` / `think_budget`) instead of answering |
| **Med s/case** | median wall-clock per case |
| **Verdict** | what this run decided, and what to change next |

**One rule:** change at most one variable between rows, so a result is attributable. If you change
two, you won't know which one moved the number — that's the mistake that cost a day on the `carts` probe.

## Reference points (already measured, from FINDINGS.md)

| Model | thinking | Clear % | source |
|---|---|---:|---|
| claude-opus-5 (ceiling) | off | 98% | FINDINGS |
| Python baseline | — | 79% | FINDINGS |
| gemma4:26b (capped evidence) | off | 72% | FINDINGS |
| qwen2.5-coder:7b | off | 67% | FINDINGS |
| gemma4:26b (plain) | off | 65% | FINDINGS |

So the bar is real: best measured local model is 72%, and that was with capped evidence. The two
**qwen3** models and **glm-4.7-flash** below have never had a 50-case score.

## Campaign

| Run ID | Model | p1 · thinking | p2 · num_ctx | p3 · schema | p4 · order | Status | Clear % | Weak % | Loops | Med s/case | Verdict |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|
| 20261001-095924-qwen38-nothink | qwen3.8:latest | off | num_ctx_for | on | 1234 | done | 63 | 57 | 0 | | baseline for this model |
| 20261002-091740-qwen38-think | qwen3.8:latest | **on** | num_ctx_for | on | 1234 | done | 93 | 43 | 1 | | does thinking beat the baseline above? |
| _tbd_ qwen36-nothink | qwen3.6:27b | off | num_ctx_for | on | 1234 | planned | | | | | only if qwen3.8 misses |
| _tbd_ qwen36-think | qwen3.6:27b | **on** | num_ctx_for | on | 1234 | planned | | | | | |
| _tbd_ glm-nothink | glm-4.7-flash | off | num_ctx_for | on | 1234 | planned | | | | | glm's fair 50-case test; thinking loops on some prompts (see experiments/2026-09-30-thinking-numctx) |
| _tbd_ glm-think | glm-4.7-flash | **on** | num_ctx_for | on | 1234 | planned | | | | | |

### Notes

- Single-case probes (the `carts` num_ctx/order work) live in
  `experiments/2026-09-30-thinking-numctx/` — that was diagnosis, not scoring. This sheet is scoring only.
- Fill a row the moment its run finishes. If a run loops on many cases, record the Loops count and move
  on — a model that can't finish the sample is disqualified regardless of accuracy on the ones it did.
- "thinking on vs off" is the paired comparison that tests whether reasoning actually helps RCA here, or
  just costs time. Read it off the Clear % of each pair, not off any one case.
