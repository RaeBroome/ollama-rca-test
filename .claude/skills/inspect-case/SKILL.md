---
name: inspect-case
description: Inspect one RCAEval case in this repo - labels, exclusion/diagnosability problems, evidence on the true root cause, and the ground-truth-blind step-0 text exactly as a model would receive it (with token counts and num_ctx). Use when the user names a case (e.g. re3ss_carts_f1_1), asks "what does the model see for X", "why is X excluded/weak", or wants to sanity-check step 0 on a case.
---

# inspect-case

Uses `rca_lib.py` in the repo root. Read-only: it never writes files and never calls Ollama.

## Steps

1. Resolve the case id. If the user gave a partial name ("carts f1", "auth loss 2"), find matches in the index and ask only if more than one fits:
   ```bash
   python -c "from rca_lib import load_index; i=load_index(); print(i[i.case.str.contains('PATTERN')].case.tolist())"
   ```
2. Run the inspection from the repo root:
   ```bash
   python rca_lib.py inspect <case>
   ```
   Options: `--no-artifacts` (drop `diskio_appears` evidence from the step-0 text, for the with/without-artifact comparison); `--max-pat-rows N` (capped variant: log pattern rows taken one per service in turn, the rest announced as OMITTED).
3. Report following CLAUDE.md "Reports": lead with the `## Problems / decisions` section (exclusions, weak or no evidence, artifact-only evidence, leak-check failures), then the key evidence. Keep it short.
   - Show the step-0 text itself only if asked, or quote the lines that matter (e.g. whether a WARN / rare-event / exception-class line survived).
   - Say explicitly if a subtle symptom (slow drift, rate change, service going quiet, WARN line, sparse errors, one-off or rare lines) is missing from the step-0 text but visible in the ground-truth evidence section.

## Rules

- The "Evidence on the true root cause" and `root_cause.txt` sections contain ground truth. Never paste them into anything that will be sent to a model.
- The step-0 text must pass the leak check (`leak check: none`). If it doesn't, report that first.
- Token counts: qwen is exact; gemma4 is an approximation (Gemma 3 tokenizer). `num_ctx` shown is `num_ctx_for(max of the two)`.
- First run downloads two tokenizer files to the Hugging Face cache; if `tokenizers` isn't installed, counts fall back to chars/3.5 - say so.
