# Thinking and num_ctx on glm-4.7-flash (2026-09-30)

One model (`glm-4.7-flash`), one case (`re3ss_carts_f1_1`, ground truth `carts`), temperature 0. A probe, not a
score: one case says nothing about accuracy.

## Question

Thinking is off by default in this repo because gemma4:26b never converged: it reasoned until it hit the token
limit and returned nothing. Does that apply to other models, and was it a budget problem?

## Runs

| test | path | thinking | num_ctx | num_predict | eval_count | done_reason | answer | wall_s |
|---|---|---|---:|---:|---:|---|---|---:|
| 1 nothink | `direct_llm` | off | 4,608 | (default) | 175 | stop | carts | 42.7 |
| 2 think | `ollama_chat` | on | 123,010 | 100,000 | 3,696 | stop | carts | 168.5 |
| 3 bigctx | `ollama_chat` | on | 123,010 | 8,192 | 3,696 | stop | carts | 169.1 |
| 4 isolated | `direct_llm` | on | 123,010 | 8,192 | 8,192 | length | (none) | 311.5 |
| 5 noschema | `direct_llm`, schema dropped | on | 20,992 | 8,192 | 8,192 | length | (none) | 177.6 |

The two paths differ in more than the call:

| | `direct_llm` (tests 1, 4, 5 and the failures) | `ollama_chat` called directly (tests 2-3) |
|---|---|---|
| structured output (`format`) | `DIRECT_SCHEMA` (not in test 5) | none |
| service order in the evidence | shuffled, seed 1234 (3,011 tokens) | alphabetical, `render_step0` default (3,010 tokens) |

Tests 2 and 3 produced identical output (13,304 thinking chars, the same 868-char answer), as expected at
temperature 0: the model stopped at 3,696 tokens, so a num_predict of 8,192 or 100,000 made no difference.

Earlier failures on the same model and case, all through `direct_llm`:

| num_ctx | num_predict | thinking chars | content | done_reason |
|---:|---:|---:|---:|---|
| 20,992 | 17,408 | 67,365 | 0 | length |
| 20,992 | 17,408 | 64,351 | 0 | length |
| 20,992 | 8,192 | 32,305 | 0 | length |

The first two are the two attempts of `results/20260929-122117-glm-think-1` (`run_sample.py --think`,
`THINKING_EXTRA` 16,384), recorded there as `no_output`. The third was not kept as a run; its num_ctx is what
`num_ctx_for` gives with the same budget.

## Finding

**num_ctx is ruled out.** Test 4 went through `direct_llm` itself, so the schema stayed on and services stayed
shuffled, with only num_ctx raised to tests 2-3's 123,010. It ran away exactly as at 20,992: 32,307 thinking
chars against 32,305, `done_reason=length`, no answer. The model's behaviour does not depend on the context size.

**The schema is ruled out too.** Test 5 went through `direct_llm` with the schema dropped and everything else
as in the third failure (shuffled order, num_ctx 20,992, num_predict 8,192). It still ran away: 31,423 thinking
chars against 32,305 with the schema, `done_reason=length`, no answer. The schema changed the reasoning a little
but not the outcome.

**The runaway is prompt-sensitive: one order converged and one ran away.** The only remaining difference between
test 3 (converged) and test 5 (ran away) is the order of services in the evidence: alphabetical against
shuffled with seed 1234. That does not make shuffling the cause, or alphabetical order a fix (alphabetical is not
neutral either, which is why the repo shuffles). At temperature 0, a small change to the prompt is enough to tip
glm between converging and reasoning to the limit. What matters for using thinking is how often it converges,
which needs more than one order to measure.

One gap remains: num_ctx was ruled out with the schema on (test 4). An effect of num_ctx only when the schema is
off (test 3 had both no schema and 123k) is unlikely but untested.

With thinking off (test 1), the shuffled order and the schema gave a correct answer in 175 tokens.

## Timing: large contexts spill to CPU

The 169 s of tests 2-3 is mostly large-context slowdown, not the cost of thinking. The evidence is the speed:
generation ran at ~26 tok/s at num_ctx 123,010 (test 4: 8,192 tokens in 311.5 s) against ~66 tok/s at 20,992
(`glm-think-1`: 17,408 tokens in 262 s). Tests 2-3 generated 3,696 tokens in ~169 s, about 22 tok/s, which
matches the slow rate. glm-4.7-flash is 19.0 GB on a 20 GB card, so a 123k-token context spilling part of the
work to the CPU is the likely reason.

Peak GPU memory does not show the spill. The card is equally full either way: 19,478 of 20,470 MB at 123,010
(test 4) and 19,490 MB at 20,992 (test 5). Ollama fills what fits on the GPU in both cases; the peak cannot say
how much ended up on the CPU.

All the times include a cold model load, because each script loads the model and unloads it at the end. So
169 s against test 1's 43 s is not a measure of what thinking costs. Only tests 4 and 5 recorded GPU memory.

## Notes on the records

- **`truncation_ok: false` in tests 4 and 5 is a false alarm.** The repo counts glm prompts with the gemma tokenizer
  (3,011 tokens), while glm itself counted 2,655 (`prompt_eval_count`). A 3k-token prompt cannot be truncated in
  a 123k context.
- **Test 4 reached num_ctx by replacing `rca_lib.num_ctx_for` inside the script.** `direct_llm` sizes num_ctx
  itself and has no parameter for it. Test 5 dropped the schema by wrapping `ollama_chat`, and set
  `THINKING_EXTRA` to the failures' 16,384 so `num_ctx_for` gave their 20,992. Both scripts block `direct_llm`'s
  retry so each test is a single call.
- **With the retry blocked there is no `direct_llm` result, so `no_output` is judged on the call itself.** The
  first version of script 4 read it from the missing result and wrote `false`. It was corrected to `true` from
  the recorded `done_reason` and `content_chars`, without re-running.
- **`test2_thinking_dump.txt` came out empty despite 13,304 thinking chars, so it was not kept.**
  `r.get("thinking")` returns nothing because `ollama_chat` never returns the thinking text. It keeps only its
  length (`thinking_chars`), by design: answers are read from `message.content` only. Saving the reasoning needs
  either a change to `ollama_chat` or a direct `/api/chat` call.

## Still open

- How often thinking converges across service orders (several shuffle seeds on this case).
- Whether gemma4:26b behaves the same way.
- Whether thinking improves accuracy over 50 cases.

## Files

Each `run_testN.py` produced the matching JSON: `test1_nothink.json`, `test2_think.json`, `test3_bigctx.json`,
`test4_isolated.json`, `test5_noschema.json`. Scripts 1-3 were moved here unchanged. They import `rca_lib` and
write their JSON to the current directory. Scripts 4 and 5 write next to themselves. To re-run one, run it from the repo root with the root on the
path, e.g. `PYTHONPATH=. python experiments/2026-09-30-thinking-numctx/run_test4.py`. Every one of them makes model
calls.
