# glm-4.7-flash loops instead of answering when num_ctx is too small

Preliminary, one case, new model. This is **not** part of the main study in
[FINDINGS.md](../FINDINGS.md), which covers qwen2.5-coder:7b, gemma4:26b and claude-opus-5.
`glm-4.7-flash` is a thinking model being evaluated separately. Recorded here because it is the
reproducible version of an effect FINDINGS lists only as one-off observations — gemma's "52k
characters of reasoning, no answer", the context-bucket run that flipped `orders`→`orders-db`,
and "num_ctx flipped one answer of 24" (temperature 0 is not determinism).

## Finding

For `re3ss_carts_f1_1`, glm-4.7-flash answers correctly only at a large context window. Shrinking
`num_ctx` does not merely slow it — it drives the model into a reasoning loop that never emits an
answer.

| num_ctx | done_reason | thinking_chars | eval_count | answer | wall_s |
|--------:|-------------|---------------:|-----------:|--------|-------:|
| 123,010 | stop          | 13,304 | 3,696  | carts (correct) | 154 |
| 123,010 | stop          | 13,304 | 3,696  | carts (correct) | 155 |
| 27,010  | think_budget  | 40,003 *(guard)* | — | none | 211 |
| 8,704   | length        | 340,517 | 87,040 | none | 1,176 |

num_ctx is `prompt_tokens + {120000, 24000, num_ctx_for}`; prompt is 3,010 tokens. The two 123k
runs are identical in token counts, so at that size the case is deterministic, not a lucky sample.
The failures are not variance either: the smaller the window, the worse the loop.

## Why

- `num_ctx_for()` sized the window at 8,704 (prompt + a 4,096-token thinking reserve + margin).
  That reserve is far too small for glm's reasoning on these prompts.
- With too little room the model keeps reasoning instead of committing — a failure to **terminate**,
  not to remember: within a run it sees its own prior tokens (KV cache); it never closes the loop.
- The large window spills a sliver of the model to CPU (~21 tok/s vs ~45, GPU ~40% vs ~84%) but
  still answers. Spill-but-correct beat no-spill-but-looping.

## Carried into the harness

- **Use a large `num_ctx` for glm-4.7-flash**; do not use `num_ctx_for()` for it until the thinking
  reserve is re-measured per model.
- **think-budget guard** at 40,000 thinking chars: the one correct run used 13,304, so 40k is a
  ceiling a looping run cannot slip under, and it caps wasted time near 2 min instead of 20+.
- `done_reason=think_budget` is its own outcome (looped), distinct from a wrong answer or an
  abstention — scored apart, as `_ran_out` already treats `length`.

## Not established / honest scope

- **One case, one model.** Whether 123k is right for other cases or models is unmeasured.
- **Not reproducible from a clean commit yet.** The runs were made from a dirty tree (`68778c3`)
  with a throwaway harness (`test_logged.py`, `runlog.py`, `ollama_stream.py`), not `run_sample.py`,
  and were not written to `results/`. To meet this repo's bar they need a re-run on a committed SHA
  with the records persisted. Until then, read the table as "observed", like the one-offs in FINDINGS.
- The three failed runs each diverged; thinking models are not perfectly deterministic, so a case
  that returns `think_budget`/`length` needs repeated runs to see if it *sometimes* answers. A `stop`
  at 123k looks stable enough that one run may suffice.

## Evidence

Readable run logs kept alongside this note: `logs/2026-10-01_073558_glm-think-on.log` (123k, stop),
`logs/2026-10-01_072251_glm-think-on.log` (27k, think_budget),
`logs/2026-10-01_063703_glm-think-on.log` (8k, length).
