"""Schema or service order? direct_llm itself - so the prompt is exactly the failures' shuffled one
(DEFAULT_ORDER_SEED) and num_ctx is sized by num_ctx_for as usual - with only the JSON schema dropped.
num_predict 8,192 and THINKING_EXTRA 16,384 as in the third failure, so num_ctx comes out at its 20,992 and the
schema is the one difference from it. Converges: the schema is the cause. Runs away: the service order is.

Run from the repo root (it imports rca_lib):
    PYTHONPATH=. python experiments/2026-09-30-thinking-numctx/run_test5.py
Writes test5_noschema.json next to this script. One model call: direct_llm's retry is blocked.
"""
import json
import threading
from pathlib import Path

import rca_lib

m, case = "glm-4.7-flash", "re3ss_carts_f1_1"
NUM_PREDICT = 8192
rca_lib.THINKING_EXTRA = 16384  # the failures' budget: num_ctx_for then sizes this prompt at 20,992


class SecondAttempt(Exception):
    pass


calls, peak = [], {"used": None}
real_chat = rca_lib.ollama_chat


def one_call(*a, **kw):
    """ollama_chat, once, with the schema removed. Everything else is what direct_llm asked for."""
    if calls:
        raise SecondAttempt
    kw = {**kw, "schema": None}
    stop = threading.Event()

    def sample():
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
    "test": "thinking ON, direct_llm path, schema OFF, shuffled order, num_ctx as num_ctx_for sizes it, num_predict 8192",
    "answer": answer,
    "content": r["content"][:2000],
    "expected": "carts",
    "thinking_chars": r["thinking_chars"],
    "content_chars": len(r["content"]),
    "done_reason": meta.get("done_reason"),
    "eval_count": meta.get("eval_count"),
    "num_ctx": meta.get("num_ctx"),
    "wall_s": meta.get("wall_s"),
    # what makes this the separating test
    "num_predict": kw.get("num_predict"),
    "thinking_extra": rca_lib.THINKING_EXTRA,
    "schema_sent": kw.get("schema") is not None,
    "server_fallback": meta.get("fallback"),
    "order_seed": rca_lib.DEFAULT_ORDER_SEED,
    "prompt_tokens": res["meta"]["prompt_tokens"] if res else None,
    "prompt_eval_count": meta.get("prompt_eval_count"),
    "truncation_ok": meta.get("truncation_ok"),  # compares against the gemma tokenizer: false is expected for glm
    # judged on the call itself: when the retry is blocked there is no direct_llm result to read it from
    "no_output": meta.get("done_reason") == "length" and not r["content"],
    "retry_blocked": blocked,
    "gpu_free_before_MB": meta.get("gpu_free_before_MB"),
    "gpu_total_MB": meta.get("gpu_total_MB"),
    "gpu_peak_used_MB": peak["used"],
}
print(json.dumps(out, indent=2))
(Path(__file__).parent / "test5_noschema.json").write_text(json.dumps(out, indent=2))
rca_lib.ollama_unload(m)
