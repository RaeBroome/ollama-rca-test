# Thinking and num_ctx on glm-4.7-flash (2026-09-30)

One model (`glm-4.7-flash`), one case (`re3ss_carts_f1_1`, ground truth `carts`), temperature 0. A probe, not a
score: one case says nothing about accuracy.

## Question

Thinking is off by default in this repo because gemma4:26b never converged: it reasoned until it hit the token
limit and returned nothing. Does that apply to other models, and was it a budget problem?

## Runs

| test | thinking | num_ctx | num_predict | eval_count | done_reason | answer | wall_s |
|---|---|---:|---:|---:|---|---|---:|
| 1 nothink | off | 4,608 | (default) | 175 | stop | carts | 42.7 |
| 2 think | on | 123,010 | 100,000 | 3,696 | stop | carts | 168.5 |
| 3 bigctx | on | 123,010 | 8,192 | 3,696 | stop | carts | 169.1 |

Tests 2 and 3 produced identical output (13,304 thinking chars, the same 868-char answer), as expected at
temperature 0: the model stopped at 3,696 tokens, so a num_predict of 8,192 or 100,000 made no difference.

Earlier failures on the same model and case, before the big context:

| num_ctx | num_predict | thinking chars | content | done_reason |
|---:|---:|---:|---:|---|
| 20,992 | 17,408 | 67,365 | 0 | length |
| 20,992 | 17,408 | 64,351 | 0 | length |
| 20,992 | 8,192 | 32,305 | 0 | length |

The first two are the two attempts of `results/20260929-122117-glm-think-1` (`run_sample.py --think`,
`THINKING_EXTRA` 16,384), recorded there as `no_output`. The third was not kept as a run; its num_ctx is what
`num_ctx_for` gives with the same budget.

## Finding

num_ctx looks like the variable, not num_predict. Test 3 succeeded on the same 8,192 budget that failed at
num_ctx 20,992, converging in 3,696 tokens. All three runs answered `carts`, which is correct.

**Not yet isolated.** The failures and tests 2-3 differ in three ways, not one:

| | failures (via `direct_llm`) | tests 2-3 (`ollama_chat` called directly) |
|---|---|---|
| num_ctx | 20,992 | 123,010 |
| structured output (`format`) | `DIRECT_SCHEMA` | none |
| service order in the evidence | shuffled, seed 1234 (3,011 tokens) | alphabetical, `render_step0` default (3,010 tokens) |

Any of the three could explain the change. Constrained decoding in particular is a plausible cause of a
thinking model running on. The test that separates them: `direct_llm` itself (schema on, shuffled order) with
only num_ctx raised to ~123k and num_predict 8,192. If it converges, num_ctx is the variable. If it runs away,
the schema or the order is.

In favour of num_ctx: this repo has seen num_ctx alone change an answer at temperature 0 before (coarse
context buckets flipped one of 24, see the comment above `num_ctx_for` in `rca_lib.py`).

## Timing

On this case, thinking took 169 s against 43 s for the same correct answer. That comparison is rough:

- **All three times include a cold model load**, because each script loads the model and unloads it at the end.
- **GPU memory was not recorded.** `ollama_chat` samples it (`gpu_free_before_MB`), but the scripts did not save
  it. glm-4.7-flash is 19.0 GB on a 20 GB card, so a 123k-token KV cache probably did not fit on the GPU, and
  part of the 169 s may be a CPU spill rather than thinking.

## Known issue

`test2_thinking_dump.txt` came out empty despite 13,304 thinking chars, so it was not kept. `r.get("thinking")`
returns nothing because `ollama_chat` never returns the thinking text. It keeps only its length
(`thinking_chars`), by design: answers are read from `message.content` only. Saving the reasoning needs either
a change to `ollama_chat` or a direct `/api/chat` call.

## Still open

- Whether gemma4:26b behaves the same way with a large num_ctx.
- Whether thinking improves accuracy over 50 cases.
- The separating test above.

## Files

`run_test1.py`, `run_test2.py` and `run_test3.py` produced `test1_nothink.json`, `test2_think.json` and
`test3_bigctx.json`. The scripts were moved here unchanged: they import `rca_lib` and write their JSON to the
current directory. To re-run one, run it from the repo root with the root on the path, e.g.
`PYTHONPATH=. python experiments/2026-09-30-thinking-numctx/run_test3.py`. That makes model calls.
