"""How often does thinking converge across service orders? Test 5 once per shuffle seed: direct_llm itself, schema
dropped, num_ctx as num_ctx_for sizes it (THINKING_EXTRA 16,384 -> 20,992 for every seed here), num_predict 8,192,
retry blocked. Seeds 1-5: all give distinct orders, none equal to seed 1234 (test 5) or alphabetical (test 3).

Run from the repo root (it imports rca_lib):
    PYTHONPATH=. python experiments/2026-09-30-thinking-numctx/run_test6.py
Writes test6_seeds.json next to this script after every seed, so an interrupted run keeps what finished.
Five model calls, one per seed.
"""
import json
import threading
from pathlib import Path

import rca_lib

m, case, truth = "glm-4.7-flash", "re3ss_carts_f1_1", "carts"
SEEDS = [1, 2, 3, 4, 5]
NUM_PREDICT = 8192
rca_lib.THINKING_EXTRA = 16384  # the failures' budget: num_ctx_for then sizes these prompts at 20,992
OUT = Path(__file__).parent / "test6_seeds.json"


class SecondAttempt(Exception):
    pass


calls, peak = [], {"used": None}
real_chat = rca_lib.ollama_chat


def one_call(*a, **kw):
    """ollama_chat, once per seed, with the schema removed. Everything else is what direct_llm asked for."""
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
results = []
for seed in SEEDS:
    calls.clear()
    peak["used"] = None
    order = rca_lib.service_order(case, seed)
    try:
        res = rca_lib.direct_llm(case, model=m, thinking=True, num_predict=NUM_PREDICT, order_seed=seed)
        answer, blocked = res["meta"]["answer"], False
    except SecondAttempt:  # the first answer did not parse; direct_llm would have asked again
        res, answer, blocked = None, None, True
    a, kw, r = calls[0]
    meta = r["meta"]
    no_output = meta.get("done_reason") == "length" and not r["content"]
    results.append({
        "test": f"thinking ON, direct_llm path, schema OFF, shuffle seed {seed}, num_predict 8192",
        "seed": seed,
        "outcome": "ran away" if no_output else "converged" if answer else "no usable answer",
        "answer": answer,
        "correct": bool(answer and rca_lib.is_correct(answer, truth)),
        "content": r["content"][:2000],
        "expected": truth,
        "thinking_chars": r["thinking_chars"],
        "content_chars": len(r["content"]),
        "done_reason": meta.get("done_reason"),
        "eval_count": meta.get("eval_count"),
        "num_ctx": meta.get("num_ctx"),
        "wall_s": meta.get("wall_s"),
        "num_predict": kw.get("num_predict"),
        "thinking_extra": rca_lib.THINKING_EXTRA,
        "schema_sent": kw.get("schema") is not None,
        "server_fallback": meta.get("fallback"),
        "order_seed": seed,
        "service_order": order,
        "truth_position": order.index(truth) + 1,  # for analysis only; the prompt stays ground-truth blind
        "prompt_eval_count": meta.get("prompt_eval_count"),
        "no_output": no_output,
        "retry_blocked": blocked,
        "gpu_free_before_MB": meta.get("gpu_free_before_MB"),
        "gpu_total_MB": meta.get("gpu_total_MB"),
        "gpu_peak_used_MB": peak["used"],
    })
    x = results[-1]
    print(f"seed {seed}: {x['outcome']:9s} answer={x['answer']} eval={x['eval_count']} "
          f"thinking_chars={x['thinking_chars']} wall={x['wall_s']}s truth_position={x['truth_position']}", flush=True)
    OUT.write_text(json.dumps({"case": case, "model": m, "seeds": SEEDS, "runs": results}, indent=2))

n_conv = sum(x["outcome"] == "converged" for x in results)
print(f"\nconverged {n_conv} of {len(results)}; ran away {sum(x['outcome'] == 'ran away' for x in results)}")
rca_lib.ollama_unload(m)
