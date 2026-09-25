# ollama-rca-test

Evaluating local Ollama models (qwen2.5-coder:7b, gemma4:26b) on root cause analysis using RCAEval. Goal: find which RCA steps a small local model can do reliably vs. where Python must prepare evidence first.

## Rules
- `RCAEval-data/` is read-only. Run records go to `results/<timestamp>-<label>/`, committed to the repo, one folder per run so re-runs never overwrite earlier results: `step1.parquet`, `step2.parquet`, `step3.parquet` (decisions), `direct.parquet` (the direct arm), `symptoms.parquet`, `retention.parquet` (one row per expected item, kept True/False), `summary.csv` (one row per case per arm) and `metadata.json` (step versions, thresholds, seed, models, timings, GPU, git commit). Write them with `rca_lib.start_run`, which returns a `RunWriter`: records are appended as jsonl during the run and converted to parquet by `finalize`, so a crash keeps what already ran. Read one case back with `rca_lib.read_case`, and accuracy per arm with `python analyse_run.py results/<run>`. `RCAEval-data/`, `__pycache__/` and `.claude/settings.local.json` stay git-ignored.
- Work step by step. Stop and report after each part; don't chain steps or automate without approval.
- Compression tuned for the model can destroy signals our own rules need. Step 0's templating masked `"msg":"connection accepted"`, `"exception":"java.lang...."`, `"statusCode":500` and `method=Authorise`, so step 2's connection-churn rule could never fire. When adding a rule that reads templated text, check what the templating already removed (`ALWAYS_KEEP_KEYS`), and prefer reading the raw field.
- Principle: filtering/compression must not collapse subtle symptoms (slow drifts, rate changes, services going quiet, WARN lines, sparse errors, one-off/rare lines, exception class names). Flag any filter or threshold that could hide one; never skip cases or rows silently.
- Thresholds are provisional; don't tune them on the cases being inspected.
- Any step-0 change invalidates existing step-1 and step-2 numbers: re-baseline before comparing. (Keeping `msg`/`exception`/`statusCode` in log templates moved qwen's step-1 recall@1 from 50% to 40% on the same 12 cases, same seed, temperature 0.)
- Arms: Python and LLM versions of step 1 (symptom detection), step 2 (tracing, label-only) and step 3 (decision), plus a direct arm that goes from step-0 evidence straight to an answer (`--direct plain|capped|roles|facts`). Never pool accuracy across step-1 sources; `analyse_run.py` breaks the deciders out by source.
- Step 2 is label-only: it attaches `role`, `path` and `direction_evidence` to step 1's ranking and never reorders. Re-ranking was measured over 120 runs: 16 better, 53 unchanged, 51 worse. `would_demote` records what re-ranking would have done.
- Scoring: exact match, case-insensitive, trimmed (`rca_lib.is_correct`). Never substring match.
- Report "clear" and "weak evidence" cases as separate scores. **Still pending:** no scoring run has yet been done with and without `diskio_appears` evidence (likely injection artifact, see FINDINGS) - do it before quoting a final number.
- Step 0 has two configurations: unranked (`num_ctx` per case, tests the model's own narrowing) and capped at ~2k tokens (`CAPPED_PAT_ROWS` log-pattern rows). Both run on all datasets, including TrainTicket. Step 0 itself never ranks: Python ranking is a separate arm, and handing the model a ranked list was the worst configuration measured.
- Don't call Ollama unless asked. When calling it: temperature 0, `num_ctx` per case from `num_ctx_for(prompt tokens, model, thinking)` rather than a flat value, and check `prompt_eval_count` with `check_prompt_eval` for silent truncation.
- gemma4:26b reports the thinking capability, but its thinking never converges on these prompts (52k characters of reasoning, no answer at several budgets), so it runs with thinking **off** - `run_sample.py` passes `thinking=False`, and every "gemma" number means gemma-no-think. Thinking stays an experiment variable for other models (capability detected via `/api/show`, budget `THINKING_EXTRA`): if a model is run both ways, report the two separately. Read answers from `message.content` only, never `thinking` (with too small an answer budget, `content` comes back empty).
- Sample `gpu_memory_mb()` right before every timed run and record it with the result. Don't trust `/api/ps` for the GPU/CPU split: on 0.34.2 it returned an empty model list, and on 0.34.3 it reports plausible-looking but wrong numbers (`SIZE 1.3 GB`, `24%/76% CPU/GPU` for a model `nvidia-smi` showed occupying 18.7 GB and running at full GPU speed). Use it only to see *which* models are resident; read memory from `nvidia-smi` / `rocm-smi`.
- Experiments run on this machine: NVIDIA RTX A4500, 20 GB. qwen2.5-coder:7b stays fully on GPU up to 16k context (~5.4 GB, ~94 tok/s); gemma4:26b needs ~18.7-19.0 GB (94-96% of the card) and can't be resident at the same time, so unload between models.

## Data
- Per case: `metrics.parquet`, `logs.parquet`, `traces.parquet` (none for Sock Shop), `inject_time.txt`; 8 RE3-SS cases have `root_cause.txt`. Index: `cases.parquet` (`root_cause_service` = ground truth).
- RE3 fault labels F1-F5 are defined only in the paper (incorrect parameter values, missing parameters, missing function call, incorrect return values, missing exception handlers); the same label looks different per service.
- Scoring exclusions, not-diagnosable cases and known limitations: `explore.ipynb` section 7 (static part: `EXCLUDE`, `BROKEN_LABELS` in `rca_lib.py`).

## Code
- Stable functions: `rca_lib.py` (reviewed; the notebook imports them). Exploration: `explore.ipynb`. Keep notebook outputs cleared in the saved file.
- Inspect one case: `python rca_lib.py inspect <case>` (the `inspect-case` skill).
- Reproduce a run: `python run_sample.py --label <name>` (`--no-llm` for the Python arms only, `--cases <ids>` or `--preset50` to choose cases, `--direct`/`--direct-only` for the direct arm, `--quickstart` to check the setup and time one case without running anything). Models are unloaded when a run finishes unless `--keep-warm`. Orchestration lives in the repo; only genuine scratch belongs in temp. Each run's `metadata.json` records the git commit, branch and whether the working tree was dirty - except the first two runs (`sample12-step1-step2`, `sample12-full-pipeline`), which predate it.
- The step-0 text given to a model must stay ground-truth blind: no case name, fault label, or root-cause marker. Service order is shuffled per case (`DEFAULT_ORDER_SEED`, recorded per run), because alphabetical order is not neutral.
- Update CLAUDE.md when renaming anything it references.

## Reports
- Lead with what's wrong or needs a decision. Keep it short. Reduce detail on things going as expected.
