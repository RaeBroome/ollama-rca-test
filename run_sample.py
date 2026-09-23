"""Run the RCA pipeline (step 1 -> 2 -> 3) over a set of cases and write a results/ run folder.

Reproduce the committed 12-case run:

    python run_sample.py --label sample12-full-pipeline

Cheap variants (no Ollama calls, seconds rather than an hour):

    python run_sample.py --cases re3ss_carts_f1_1 --no-llm --label smoke
    python run_sample.py --preset sample12 --no-llm --label python-only

Model calls are batched by model (one resident at a time: the two models do not
fit on a 20 GB card together). Records are written incrementally, so a crash
keeps everything produced up to that point.
"""
import argparse
import time

from rca_lib import (MODELS_DEFAULT, SAMPLE12, ensure_only, start_run, step1_llm, step1_python,
                     step2_llm, step2_python, step3_llm, step3_python)


def run(cases, models, label, notes="", use_llm=True, step2_rules=("naive", "rule"),
        step3_rules=("top1", "role")):
    w = start_run(label, notes=notes or f"{len(cases)} cases, models={models if use_llm else 'none'}")
    print("run dir:", w.dir, flush=True)
    t0 = time.time()
    s1, s2 = {}, {}

    def add_python_step2(case, s1key):
        for rule in step2_rules:
            r = step2_python(case, s1[(case, s1key)], rule=rule)
            s2[(case, f"{s1key}+py2:{rule}")] = r
            w.add(case, "step2", r["meta"]["source"], r)

    # ---- Python step 1 (+ its Python step 2 arms)
    for case in cases:
        for order in ("strength", "onset"):
            r = step1_python(case, order=order)
            s1[(case, f"python:{order}")] = r
            w.add(case, "step1", r["meta"]["source"], r)
        add_python_step2(case, "python:strength")
    print(f"python step 1/2 done {time.time() - t0:.0f}s", flush=True)

    # ---- one model resident at a time: its step 1, step 2 and step 3 for every chain it touches
    if use_llm:
        for model in models:
            ensure_only(model)
            tag = "qwen" if "qwen" in model else "gemma"
            for case in cases:
                r = step1_llm(case, model=model, thinking=False)
                s1[(case, tag)] = r
                w.add(case, "step1", r["meta"]["source"], r)
                add_python_step2(case, tag)
                for s1key in ("python:strength", tag):
                    r2 = step2_llm(case, s1[(case, s1key)], model=model, thinking=False)
                    s2[(case, f"{s1key}+llm2:{tag}")] = r2
                    w.add(case, "step2", r2["meta"]["source"], r2)
                for chain in (f"python:strength+py2:rule", f"python:strength+llm2:{tag}",
                              f"{tag}+py2:rule", f"{tag}+llm2:{tag}"):
                    if (case, chain) in s2:
                        r3 = step3_llm(case, s2[(case, chain)], model=model, thinking=False)
                        w.add(case, "step3", r3["meta"]["source"], r3)
                print(f"{tag} {case:32s} {time.time() - t0:.0f}s", flush=True)

    # ---- Python step 3 on every chain (free)
    for (case, chain), r2 in list(s2.items()):
        for rule in step3_rules:
            r3 = step3_python(case, r2, rule=rule)
            w.add(case, "step3", r3["meta"]["source"], r3)

    summary = w.finalize(extra_meta={"cases": list(cases), "models": models if use_llm else [],
                                     "use_llm": use_llm, "command": "run_sample.py"})
    print(f"\nrecords: {w.n} | summary rows: {len(summary)} | {time.time() - t0:.0f}s")
    print(w.dir)
    return w.dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", help="case ids (default: the --preset sample)")
    ap.add_argument("--preset", default="sample12", choices=["sample12"], help="named case set")
    ap.add_argument("--models", nargs="*", default=list(MODELS_DEFAULT))
    ap.add_argument("--no-llm", action="store_true", help="Python arms only; makes no Ollama calls")
    ap.add_argument("--label", default="run", help="folder name suffix under results/")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    cases = a.cases or [c for c, _ in SAMPLE12]
    run(cases, a.models, a.label, notes=a.notes, use_llm=not a.no_llm)


if __name__ == "__main__":
    main()
