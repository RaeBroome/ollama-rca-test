# ollama-rca-test

Evaluating local Ollama models (qwen2.5-coder:7b, gemma4:26b) on root cause analysis using RCAEval. Goal: find which RCA steps a small local model can do reliably vs. where Python must prepare evidence first.

## Rules
- `RCAEval-data/` is read-only. Step outputs go to `derived/stepN_*/` as parquet, with a metadata file recording rule/threshold versions (`THRESHOLDS_VERSION`, `STEP0_VERSION` in `rca_lib.py`). Don't write step outputs until asked.
- Work step by step. Stop and report after each part; don't chain steps or automate without approval.
- Principle: filtering/compression must not collapse subtle symptoms (slow drifts, rate changes, services going quiet, WARN lines, sparse errors, one-off/rare lines, exception class names). Flag any filter or threshold that could hide one; never skip cases or rows silently.
- Thresholds are provisional; don't tune them on the cases being inspected.
- Scoring: exact match, case-insensitive, trimmed (`rca_lib.is_correct`). Never substring match.
- Report "clear" and "weak evidence" cases as separate scores. Run scores with and without `diskio_appears` evidence (likely injection artifact).
- Step 0 has two configurations: unranked (`num_ctx` per case, tests the model's own narrowing) and capped at ~3k tokens with a swappable ranking step. Both run on all datasets, including TrainTicket.
- Don't call Ollama unless asked. When calling it: temperature 0, `num_ctx` per case from `num_ctx_for(prompt tokens, model, thinking)` rather than a flat value, and check `prompt_eval_count` with `check_prompt_eval` for silent truncation.
- gemma4:26b is a thinking model. Thinking is an experiment variable, **default on**; when comparing models, run gemma both ways and report the two separately. Read answers from `message.content` only, never `thinking` (with too small an answer budget, `content` comes back empty).
- Sample `gpu_memory_mb()` right before every timed run and record it with the result. Ollama 0.34.2's `/api/ps` returns an empty model list, so there is no GPU/CPU split from the API and a spill is only visible afterwards from free memory plus throughput.
- Experiments run on this machine: NVIDIA RTX A4500, 20 GB. qwen2.5-coder:7b stays fully on GPU up to 16k context (~5.4 GB, ~94 tok/s); gemma4:26b needs ~18.7-19.0 GB (94-96% of the card) and can't be resident at the same time, so unload between models.

## Data
- Per case: `metrics.parquet`, `logs.parquet`, `traces.parquet` (none for Sock Shop), `inject_time.txt`; 8 RE3-SS cases have `root_cause.txt`. Index: `cases.parquet` (`root_cause_service` = ground truth).
- RE3 fault labels F1-F5 are defined only in the paper (incorrect parameter values, missing parameters, missing function call, incorrect return values, missing exception handlers); the same label looks different per service.
- Scoring exclusions, not-diagnosable cases and known limitations: `explore.ipynb` section 7 (static part: `EXCLUDE`, `BROKEN_LABELS` in `rca_lib.py`).

## Code
- Stable functions: `rca_lib.py` (reviewed; the notebook imports them). Exploration: `explore.ipynb`. Keep notebook outputs cleared in the saved file.
- Inspect one case: `python rca_lib.py inspect <case>` (the `inspect-case` skill).
- The step-0 text given to a model must stay ground-truth blind: no case name, fault label, or root-cause marker; services in a neutral order.
- Update CLAUDE.md when renaming anything it references.

## Reports
- Lead with what's wrong or needs a decision. Keep it short. Reduce detail on things going as expected.
