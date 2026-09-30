import rca_lib, json
m, case = "glm-4.7-flash", "re3ss_carts_f1_1"
rca_lib.ensure_only(m)
ev = rca_lib.render_step0(case)
prompt = rca_lib.DIRECT_INSTRUCTIONS + ev
n = rca_lib.count_tokens(prompt, "gemma")
r = rca_lib.ollama_chat(m, prompt, n + 120000, thinking=True, num_predict=8192)
out = {
    "test": "thinking ON, big num_ctx, num_predict=8192",
    "content": r["content"][:2000],
    "expected": "carts",
    "thinking_chars": r["thinking_chars"],
    "content_chars": len(r["content"]),
    "done_reason": r["meta"].get("done_reason"),
    "eval_count": r["meta"].get("eval_count"),
    "num_ctx": r["meta"].get("num_ctx"),
    "wall_s": r["meta"].get("wall_s"),
}
print(json.dumps(out, indent=2))
open("test3_bigctx.json", "w").write(json.dumps(out, indent=2))
rca_lib.ollama_unload(m)