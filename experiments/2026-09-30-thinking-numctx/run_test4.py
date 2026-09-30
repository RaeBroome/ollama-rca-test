"""Separating test for this folder's README: direct_llm itself - the code path of the failures, so the JSON
schema stays on and services stay shuffled with DEFAULT_ORDER_SEED - with only num_ctx raised to tests 2-3's
123,010 and num_predict 8,192 (the budget of the third failure). Converges: num_ctx is the variable. Runs to
the limit: the schema or the service order is.

Run from the repo root (it imports rca_lib):
    PYTHONPATH=. python experiments/2026-09-30-thinking-numctx/run_test4.py
Writes test4_isolated.json next to this script. One model call: direct_llm's retry is blocked.
"""
import json
import threading
from pathlib import Path

import rca_lib

m, case = "glm-4.7-flash", "re3ss_carts_f1_1"
NUM_CTX, NUM_PREDICT = 123010, 8192

# direct_llm sizes num_ctx itself and passes no answer_reserve, so without changing repo code the only way in
# is to replace num_ctx_for for this process. Nothing else in direct_llm is touched.
rca_lib.num_ctx_for = lambda n_tok, model=None, thinking=False, **kw: NUM_CTX


class SecondAttempt(Exception):
    pass


calls, peak = [], {"used": None}
real_chat = rca_lib.ollama_chat


def one_call(*a, **kw):
    """ollama_chat, once. direct_llm retries an unparseable answer; this test is a single call."""
    if calls:
        raise SecondAttempt
    stop = threading.Event()

    def sample():  # peak GPU memory during the call: a 123k context may not fit next to a 19 GB model
        while not stop.wait(2):
            g = rca_lib.gpu_memory_mb()
            if g:
                peak["used"] = max(peak["used"] or 0, g[1])
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    try:
        r = real_chat(*a, **kw)
    finally:
        stop.set()
        t.join()
    calls.append((a, kw, r))
    return r


rca_lib.ollama_chat = one_call

rca_lib.ensure_only(m)
try:
    res = rca_lib.direct_llm(case, model=m, thinking=True, num_predict=NUM_PREDICT)
    answer, blocked = res["meta"]["answer"], False
except SecondAttempt:  # the first answer did not parse; direct_llm would have asked again
    res, answer, blocked = None, None, True

a, kw, r = calls[0]
meta = r["meta"]
out = {
    "test": "thinking ON, direct_llm path (schema on, shuffled order), num_ctx 123010, num_predict 8192",
    "answer": answer,
    "content": r["content"][:2000],
    "expected": "carts",
    "thinking_chars": r["thinking_chars"],
    "content_chars": len(r["content"]),
    "done_reason": meta.get("done_reason"),
    "eval_count": meta.get("eval_count"),
    "num_ctx": meta.get("num_ctx"),
    "wall_s": meta.get("wall_s"),
    # what makes this the isolated test
    "num_predict": kw.get("num_predict"),
    "schema_sent": kw.get("schema") is rca_lib.DIRECT_SCHEMA and not meta.get("fallback"),
    "server_fallback": meta.get("fallback"),
    "order_seed": rca_lib.DEFAULT_ORDER_SEED,
    "prompt_tokens": res["meta"]["prompt_tokens"] if res else None,
    "prompt_eval_count": meta.get("prompt_eval_count"),
    "truncation_ok": meta.get("truncation_ok"),
    # judged on the call itself: when the retry is blocked there is no direct_llm result to read it from
    "no_output": meta.get("done_reason") == "length" and not r["content"],
    "retry_blocked": blocked,
    "gpu_free_before_MB": meta.get("gpu_free_before_MB"),
    "gpu_total_MB": meta.get("gpu_total_MB"),
    "gpu_peak_used_MB": peak["used"],
}
print(json.dumps(out, indent=2))
(Path(__file__).parent / "test4_isolated.json").write_text(json.dumps(out, indent=2))
rca_lib.ollama_unload(m)
