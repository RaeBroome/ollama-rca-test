"""Streaming version. Run:  python test_logged.py
Now the heartbeat shows live token counts, and a runaway thinking loop stops itself."""
from runlog import RunLog
from ollama_stream import ollama_chat_stream
import rca_lib, json

m, case = "glm-4.7-flash", "re3ss_carts_f1_1"

# Stop a runaway: if the model produces this many THINKING chars with no answer yet, bail.
# ~40,000 chars is roughly 10k tokens of pure reasoning - well past a real answer for these cases.
THINK_BUDGET_CHARS = 40000

with RunLog("glm-think-on", config={"model": m, "case": case,
            "num_predict": 100000, "thinking": True, "think_budget_chars": THINK_BUDGET_CHARS}) as log:

    log.event("prep", "building evidence text (render_step0)...")
    ev = rca_lib.render_step0(case)
    log.event("prep", "evidence built", chars=len(ev))

    prompt = rca_lib.DIRECT_INSTRUCTIONS + ev
    log.event("prep", "counting prompt tokens...")
    n = rca_lib.count_tokens(prompt, "gemma")
    num_ctx = n + 120000
    # num_ctx = n + 24000   # generous headroom for thinking; num_ctx_for's reserve is too small for glm <-- no answer token issue
    log.event("prep", "tokens counted", prompt_tokens=n, num_ctx=num_ctx)

    log.event("load", "freeing VRAM, unloading other models (ensure_only)...")
    rca_lib.ensure_only(m)
    log.event("load", "other models unloaded")

    # This callback runs ~once a second while the model streams. It feeds the live
    # numbers into log.progress, which the 5s heartbeat then prints and records.
    def on_progress(info):
        log.progress.update(tokens=info["tokens"], tok_s=info["tok_s"],
                            think_chars=info["thinking_chars"], phase=info["phase"])

    log.event("generate", "streaming request sent (live token counts below)...")
    r = ollama_chat_stream(m, prompt, num_ctx, thinking=True, num_predict=100000,
                           on_progress=on_progress, think_budget_chars=THINK_BUDGET_CHARS)
    log.event("generate", "model returned",
              done_reason=r["meta"].get("done_reason"),
              eval_count=r["meta"].get("eval_count"),
              wall_s=r["meta"].get("wall_s"),
              prompt_eval_count=r["meta"].get("prompt_eval_count"))

    log.event("wrap", "writing result + artifacts...")
    log.result(done_reason=r["meta"].get("done_reason"),
               eval_count=r["meta"].get("eval_count"),
               num_ctx=r["meta"].get("num_ctx"),
               wall_s=r["meta"].get("wall_s"),
               thinking_chars=r["thinking_chars"],
               content_chars=len(r["content"]),
               answer=r["content"][:200].replace(chr(10), " "),
               expected="carts")

    log.event("wrap", "unloading model (ollama_unload)...")
    rca_lib.ollama_unload(m)
    log.event("wrap", "done")

json.dump({"content": r["content"][:2000], "meta": r["meta"]},
          open("test2_think.json", "w"), indent=2)
open("test2_thinking_dump.txt", "w", encoding="utf-8").write(r.get("thinking", ""))
