# ollama-rca-test

Does a local Ollama model help with root cause analysis of microservice failures? This repo evaluates
qwen2.5-coder:7b and gemma4:26b on the [RCAEval](https://github.com/phamquiluan/RCAEval) benchmark against a
deterministic Python baseline, and against claude-opus-5 as a ceiling reference. Short version: on the same
compressed evidence Claude gets 98% of the clear cases, the Python baseline 79%, the local models 65-67% - so
the compression is sufficient and the local model is the limit. Results and conclusions:
[FINDINGS.md](FINDINGS.md).

## Requirements

- **Python 3.13** with `pandas`, `pyarrow`, `numpy`. 3.13.0 is the only version this has been run on;
  nothing in the code needs it specifically, but the dependency versions used here do (numpy 2.5 requires
  3.12+, pandas 3.0 requires 3.11+), so older Pythons need older pinned dependencies and are untested.
  Optional: `matplotlib`, `nbformat`, `nbclient`, `ipykernel` (for `explore.ipynb`), `tokenizers` +
  `huggingface_hub` (exact token counts; without them counts fall back to characters/3.5).
- **[Ollama](https://ollama.com)** (developed against 0.34.x; older versions work — the harness detects
  missing endpoints and degrades). Set `OLLAMA_HOST` if it is not on `127.0.0.1:11434`. Models pulled:
  ```bash
  ollama pull qwen2.5-coder:7b     # 4.7 GB
  ollama pull gemma4:26b           # 18 GB
  ```
- **A GPU with enough VRAM for one model at a time**, or none — CPU works, just slower. Measured here on an
  RTX A4500 (20 GB): qwen stays fully on GPU up to 16k context (~5.4 GB), gemma needs ~19 GB. The two cannot
  be resident together, so runs unload one before loading the other; if unloading fails it warns and carries
  on. Free VRAM is read from `nvidia-smi`, or `rocm-smi` on AMD (untested), and simply recorded as unknown
  when neither is present.
- **The RCAEval dataset** in `RCAEval-data/` (read-only, git-ignored, ~5 GB): download it from
  [huggingface.co/datasets/phamquiluan/RCAEval](https://huggingface.co/datasets/phamquiluan/RCAEval). The
  layout expected is `RCAEval-data/cases.parquet` plus one folder per case containing `metrics.parquet`,
  `inject_time.txt`, and where available `logs.parquet`, `traces.parquet`, `root_cause.txt`.

## Running

Start here — it checks Ollama and the dataset, detects the model's thinking capability, times one case cold
and warm, prints estimates for the real runs, and stops without running anything else:

```bash
python run_sample.py --quickstart --models <your model>
```

Then:

```bash
python run_sample.py --no-llm --label python-only            # Python arms only, no model calls
python run_sample.py --preset50 --direct plain capped roles --direct-only --label direct-variants
python run_sample.py --label full-pipeline                   # staged step 1 -> 2 -> 3, all arms
python run_sample.py --cases re3ss_carts_f1_1 --no-llm --label smoke   # one case, ~30 s
```

Useful flags: `--cases <ids>`, `--preset50` (50-case stratified sample), `--models`, `--direct
{plain,capped,roles,facts}`, `--direct-only` (skip the staged LLM arms), `--keep-warm` (leave the model
loaded at the end), `--quickstart` (check the setup and stop). Model calls are batched by model, so a run
swaps models a few times, not once per case.

`--claude opus` adds a **ceiling reference**: the same evidence answered by Claude through the Claude Code
CLI, authenticated by your own subscription login rather than an API key. It needs `claude` on PATH and a
completed `claude` login, uses no GPU, and is skipped with a printed reason if either is missing. The call is
blinded — tools removed, run from an empty directory outside the repo, Claude Code's system prompt replaced,
one process per case — but the CLI exposes no temperature setting, so this arm alone is not pinned to
temperature 0. See [FINDINGS.md](FINDINGS.md); compare a run with `python compare_claude_arm.py <run>`.

**The default 12-case preset is for checking that a model works, not for drawing conclusions.** One case is
worth 8 points there, and both of its headline results reversed at 50 cases: qwen scored 80% on 12 (8 correct
of 10 clear cases) and 67% on 50, while gemma looked clearly worse than qwen on 12 and turned out level or
ahead on 50. Use `--preset50` for any number you intend to compare or quote.

Inspect a single case, including the exact text a model receives:

```bash
python rca_lib.py inspect re3ss_carts_f1_1
```

## Results

Each run writes `results/<timestamp>-<label>/`, committed to the repo so runs are never overwritten:

| file | contents |
|---|---|
| `summary.csv` | one row per case per arm: answer, correct, abstained, rank of truth, stated reason, retention kept/missed, tokens, time, free GPU |
| `step1/step2/step3/direct.parquet` | full candidate records per arm (`direct.parquet` holds the direct arm) |
| `symptoms.parquet`, `retention.parquet` | the evidence behind them; retention has one row per expected signal with `kept` true/false |
| `metadata.json` | step versions, thresholds, order seed, models, timings, and the git commit plus whether the tree was dirty (`sample12-step1-step2` and `sample12-full-pipeline` predate the git block) |

Records are written as each result is produced (`summary.csv` plus append-only `.jsonl`, converted to parquet
at the end), so a crash keeps what already ran.

Read them:

```bash
python analyse_run.py results/<run>                    # accuracy by arm, never pooled across step-1 sources
python -c "from rca_lib import read_case; r = read_case('results/<run>', 're3ss_carts_f1_1'); print(r['summary'].to_string())"
```

## Using a different model

1. `ollama pull <model>`, then either pass `--models <model>` or edit `MODELS_DEFAULT` in `rca_lib.py`.
   Run `python run_sample.py --quickstart --models <model>` first: it verifies the model is installed,
   reports its thinking capability and answer reserve, and times a call on your hardware.
2. **Answer budget (optional)**: an unknown model falls back to the default reserve (1024 tokens, plus 4096
   if thinking is on), so nothing needs changing to get started. Add an entry to `ANSWER_RESERVE` in
   `rca_lib.py` only if a model needs a different budget. `num_ctx_for(prompt_tokens, model, thinking)` sizes
   the context per case as prompt + reserve + 15% rather than a flat value, because a flat 16k wastes VRAM on
   small cases and truncates large ones. Ollama truncates over-long prompts **silently**, so every call checks
   `prompt_eval_count` with `check_prompt_eval`.
3. **Thinking mode**: detected per model via `/api/show`; Ollama returns HTTP 400 if you pass `think` to a
   model without the capability. For a thinking model, budget for reasoning tokens that never reach
   `message.content` — `THINKING_EXTRA` adds 4096. Read answers from `message.content` only.

   **Caveat we hit:** gemma4:26b's thinking never converged on these prompts. At `num_ctx` 24576 with
   `num_predict` 20000 it produced 52k characters of reasoning and no answer, after the same at 5k, 6k and
   12k tokens. It is run with thinking off. If a new model returns empty answers, check `done_reason` — if it
   is `length` with a large `thinking` field, that is this failure mode, not a parse error.

4. **Runtime**: the 50-case run here took **2h13m** for two models — qwen 7B and gemma 26B — on a dedicated
   20 GB GPU, but most of that was **model reloads, not inference**. Ollama rebuilds the runner whenever
   `num_ctx` changes (~18 s for a 26B model, against 0.2 s for the same call at an unchanged `num_ctx`), and
   context is sized per case, so most calls pay it. Expect the same on your hardware; coarse context buckets
   were tried and reverted because they changed an answer. A much larger model, or one on unified memory
   (say a 120B on a Framework Desktop at ~256 GB/s), will be slower again per call, and the cost scales with
   call count: the 50-case three-variant run is 300 calls. Use `--quickstart` to measure before committing.

5. **Models are unloaded when a run finishes**, freeing VRAM; pass `--keep-warm` to leave the last one
   resident if another run follows immediately. `rca_lib.KEEP_ALIVE` (default `"5m"`) sets how long Ollama
   holds a model between calls. Before each model's phase, other resident models are evicted to free VRAM —
   announced in the output, because on a shared Ollama that evicts them for other users too.

## Attribution

The benchmark and all failure data are **not mine**. They are RCAEval, by Luan Pham, Hongyu Zhang, Huong Ha,
Flora Salim and Xiuzhen Zhang, published in the Companion Proceedings of the ACM on Web Conference 2025
(WWW 2025 Companion, pages 777–780), [arXiv:2412.17015](https://arxiv.org/abs/2412.17015).

- Code and benchmark: <https://github.com/phamquiluan/RCAEval>
- Dataset: <https://huggingface.co/datasets/phamquiluan/RCAEval> — **MIT licensed**
- Package: <https://pypi.org/project/RCAEval>

```bibtex
@inproceedings{pham2025rcaeval,
  title={RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data},
  author={Pham, Luan and Zhang, Hongyu and Ha, Huong and Salim, Flora and Zhang, Xiuzhen},
  booktitle={Companion Proceedings of the ACM on Web Conference 2025},
  pages={777--780},
  year={2025}
}
```

**Theirs:** the 735 failure cases, the fault injection, the ground-truth labels, and the three microservice
systems (Online Boutique, Sock Shop, Train Ticket) the data was collected from.

**Mine:** the evaluation harness in this repo — evidence compression, the pipeline steps, the scoring and run
records — and the findings in [FINDINGS.md](FINDINGS.md). Those findings are about local models on this
benchmark; they are not claims about RCAEval itself, and none of the RCAEval authors' methods were evaluated
here. This repo does not redistribute the dataset: `RCAEval-data/` is git-ignored and must be downloaded from
the link above.

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
- `results/` — one committed folder per run, described under [Results](#results) above.
- `CLAUDE.md` — working rules for this project. `.claude/` holds local settings, git-ignored.
