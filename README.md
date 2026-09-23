# ollama-rca-test

Does a local Ollama model help with root cause analysis of microservice failures? This repo evaluates
qwen2.5-coder:7b and gemma4:26b on the [RCAEval](https://github.com/phamquiluan/RCAEval) benchmark against a
deterministic Python baseline. Results and conclusions: [FINDINGS.md](FINDINGS.md).

## Requirements

- **Python 3.11+** with `pandas`, `pyarrow`, `numpy`. Optional: `matplotlib`, `nbformat`, `nbclient`,
  `ipykernel` (for `explore.ipynb`), `tokenizers` + `huggingface_hub` (exact token counts; without them
  counts fall back to characters/3.5).
- **[Ollama](https://ollama.com)** running locally (developed against 0.34.2) with the models pulled:
  ```bash
  ollama pull qwen2.5-coder:7b     # 4.7 GB
  ollama pull gemma4:26b           # 18 GB
  ```
- **A GPU with enough VRAM for one model at a time.** Measured here on an RTX A4500 (20 GB): qwen stays
  fully on GPU up to 16k context (~5.4 GB), gemma needs ~19 GB. The two cannot be resident together, so runs
  unload one model before loading the other.
- **The RCAEval dataset** in `RCAEval-data/` (read-only, git-ignored, ~5 GB): download it from
  [huggingface.co/datasets/phamquiluan/RCAEval](https://huggingface.co/datasets/phamquiluan/RCAEval). The
  layout expected is `RCAEval-data/cases.parquet` plus one folder per case containing `metrics.parquet`,
  `inject_time.txt`, and where available `logs.parquet`, `traces.parquet`, `root_cause.txt`.

## Running

```bash
python run_sample.py --no-llm --label python-only            # Python arms only, no model calls
python run_sample.py --preset50 --direct plain capped roles --direct-only --label direct-variants
python run_sample.py --label full-pipeline                   # staged step 1 -> 2 -> 3, all arms
python run_sample.py --cases re3ss_carts_f1_1 --no-llm --label smoke   # one case, ~30 s
```

Useful flags: `--cases <ids>`, `--preset50` (50-case stratified sample), `--models`, `--direct
{plain,capped,roles,facts}`, `--direct-only` (skip the staged LLM arms). Model calls are batched by model,
so a run swaps models a few times, not once per case.

Inspect a single case, including the exact text a model receives:

```bash
python rca_lib.py inspect re3ss_carts_f1_1
```

## Results

Each run writes `results/<timestamp>-<label>/`, committed to the repo so runs are never overwritten:

| file | contents |
|---|---|
| `summary.csv` | one row per case per arm: answer, correct, abstained, rank of truth, stated reason, retention kept/missed, tokens, time, free GPU |
| `step1/step2/step3/direct.parquet` | full candidate records per arm |
| `symptoms.parquet`, `retention.parquet` | the evidence behind them; retention has one row per expected signal with `kept` true/false |
| `metadata.json` | step versions, thresholds, order seed, models, timings, and the git commit plus whether the tree was dirty |

Records are written as each result is produced (`summary.csv` plus append-only `.jsonl`, converted to parquet
at the end), so a crash keeps what already ran.

Read them:

```bash
python analyse_run.py results/<run>                    # accuracy by arm, never pooled across step-1 sources
python -c "from rca_lib import read_case; r = read_case('results/<run>', 're3ss_carts_f1_1'); print(r['summary'].to_string())"
```

## Using a different model

1. `ollama pull <model>`, then either pass `--models <model>` or edit `MODELS_DEFAULT` in `rca_lib.py`.
2. **Answer budget**: add an entry to `ANSWER_RESERVE` in `rca_lib.py`. `num_ctx_for(prompt_tokens, model,
   thinking)` sizes the context per case as prompt + reserve + 15% rather than a flat value, because a flat
   16k wastes VRAM on small cases and truncates large ones. Ollama truncates over-long prompts **silently**,
   so every call checks `prompt_eval_count` with `check_prompt_eval`.
3. **Thinking mode**: detected per model via `/api/show`; Ollama returns HTTP 400 if you pass `think` to a
   model without the capability. For a thinking model, budget for reasoning tokens that never reach
   `message.content` — `THINKING_EXTRA` adds 4096. Read answers from `message.content` only.

   **Caveat we hit:** gemma4:26b's thinking never converged on these prompts. At `num_ctx` 24576 with
   `num_predict` 20000 it produced 52k characters of reasoning and no answer, after the same at 5k, 6k and
   12k tokens. It is run with thinking off. If a new model returns empty answers, check `done_reason` — if it
   is `length` with a large `thinking` field, that is this failure mode, not a parse error.

## Layout

- `rca_lib.py` — reviewed, stable functions: evidence compression (step 0), symptom detection (step 1),
  tracing (step 2), decision (step 3), the direct arm, scoring, run records.
- `run_sample.py`, `analyse_run.py` — orchestration and analysis.
- `explore.ipynb` — the exploration behind the design; section 7 holds scoring exclusions and known
  limitations.
- `run_test.py`, `run_test_logs.py` — the original naive baselines: raw metrics or log lines dumped
  straight into a prompt. Kept as the evidence behind the one-shot failure described in
  [FINDINGS.md](FINDINGS.md); `run_test.py` still scores with the substring check that inflates
  accuracy, left as-is to show the trap.
- `CLAUDE.md` — working rules for this project.
