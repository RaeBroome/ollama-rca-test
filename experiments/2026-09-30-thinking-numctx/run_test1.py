import rca_lib, json
m, case = "glm-4.7-flash", "re3ss_carts_f1_1"
rca_lib.ensure_only(m)
r = rca_lib.direct_llm(case, model=m, thinking=False)
a = r["meta"]["attempts"][-1]
out = {
    "test": "thinking OFF",
    "answer": r["meta"]["answer"],
    "expected": "carts",
    "thinking_chars": a.get("thinking_chars"),
    "content_chars": a.get("content_chars"),
    "done_reason": a.get("done_reason"),
    "eval_count": a.get("eval_count"),
    "num_ctx": a.get("num_ctx"),
    "wall_s": a.get("wall_s"),
}
print(json.dumps(out, indent=2))
open("test1_nothink.json", "w").write(json.dumps(out, indent=2))
rca_lib.ollama_unload(m)