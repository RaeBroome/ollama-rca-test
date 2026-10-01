"""ollama_chat_stream - same inputs/outputs as rca_lib.ollama_chat, but streams.

Why it exists: ollama_chat blocks until the whole answer is done, so you can't see progress
(and can't tell a slow run from a stuck one). This version reads the reply token-by-token,
so it can report live counts and, if asked, stop a runaway thinking loop early.

Returns the SAME dict shape as rca_lib.ollama_chat, plus r["thinking"] (full reasoning text).
Drop this file next to rca_lib.py. It reuses rca_lib's helpers; it does not change rca_lib.

    r = ollama_chat_stream(m, prompt, num_ctx, thinking=True, num_predict=100000,
                           on_progress=cb, think_budget_chars=40000)

on_progress(info) is called every ~1s with:
    {"tokens": int, "tok_s": float, "thinking_chars": int, "content_chars": int, "phase": "think"|"answer"}
think_budget_chars: if the model produces more than this many THINKING chars before any answer
    content appears, stop and return done_reason="think_budget". None disables the guard.
"""
import json
import time
import urllib.request

import rca_lib  # reuse count_tokens, check_prompt_eval, gpu_memory_mb, model_supports_thinking, ollama_url


def ollama_chat_stream(model, prompt, num_ctx, thinking=True, num_predict=1024, timeout=1800,
                       keep_alive=None, on_progress=None, progress_every=1.0, think_budget_chars=None):
    gpu_before = rca_lib.gpu_memory_mb()
    supported = rca_lib.model_supports_thinking(model)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True,
            "keep_alive": rca_lib.KEEP_ALIVE if keep_alive is None else keep_alive,
            "options": {"num_ctx": num_ctx, "temperature": 0, "num_predict": num_predict}}
    if supported:
        body["think"] = thinking

    url = rca_lib.ollama_url("/api/chat")
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})

    content, thinking_text = [], []
    final = {}                       # the last chunk (done=true) carries the timing/counts
    tokens = 0                       # response chunks seen (a proxy for generated tokens, live)
    t0 = time.time()
    last_emit = 0.0
    budget_hit = False

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:             # one JSON object per line
            if not raw.strip():
                continue
            chunk = json.loads(raw)
            msg = chunk.get("message", {})
            if msg.get("thinking"):
                thinking_text.append(msg["thinking"])
            if msg.get("content"):
                content.append(msg["content"])
            tokens += 1

            now = time.time()
            if on_progress and (now - last_emit) >= progress_every:
                tc = sum(len(x) for x in thinking_text)
                cc = sum(len(x) for x in content)
                on_progress({"tokens": tokens, "tok_s": round(tokens / max(now - t0, 1e-6), 1),
                             "thinking_chars": tc, "content_chars": cc,
                             "phase": "answer" if cc else "think"})
                last_emit = now

            # runaway guard: too much thinking and still no answer -> bail
            if (think_budget_chars and not content
                    and sum(len(x) for x in thinking_text) > think_budget_chars):
                budget_hit = True
                break

            if chunk.get("done"):
                final = chunk
                break

    wall = time.time() - t0
    content_str = "".join(content)
    thinking_str = "".join(thinking_text)
    expected = rca_lib.count_tokens(prompt, "qwen" if "qwen" in model else "gemma")
    ok, note = rca_lib.check_prompt_eval(expected, final.get("prompt_eval_count"))
    done_reason = "think_budget" if budget_hit else final.get("done_reason")

    return {"content": content_str, "thinking": thinking_str, "thinking_chars": len(thinking_str),
            "meta": {"model": model, "thinking": thinking and supported, "fallback": "",
                     "thinking_supported": supported, "num_ctx": num_ctx, "wall_s": round(wall, 2),
                     "prompt_tokens_expected": expected, "prompt_eval_count": final.get("prompt_eval_count"),
                     "eval_count": final.get("eval_count"), "done_reason": done_reason,
                     "truncation_ok": ok, "truncation_note": note,
                     "gpu_free_before_MB": gpu_before[0] if gpu_before else None,
                     "gpu_total_MB": gpu_before[2] if gpu_before else None,
                     "stream_chunks": tokens, "think_budget_chars": think_budget_chars}}
