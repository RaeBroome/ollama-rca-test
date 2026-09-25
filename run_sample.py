"""Run the RCA pipeline (step 1 -> 2 -> 3) over a set of cases and write a results/ run folder.

Reproduce the committed 12-case run:

    python run_sample.py --label sample12-full-pipeline

First time here? Check the setup and measure your hardware before committing to
a long run - this checks Ollama and the dataset, times one case, and prints the
commands with estimates, without running anything else:

    python run_sample.py --quickstart --models <your model>

Cheap variants (no Ollama calls, seconds rather than an hour):

    python run_sample.py --cases re3ss_carts_f1_1 --no-llm --label smoke
    python run_sample.py --preset sample12 --no-llm --label python-only

Model calls are batched by model (one resident at a time: the two models do not
fit on a 20 GB card together). Records are written incrementally, so a crash
keeps everything produced up to that point.
"""
import argparse
import time
from pathlib import Path

from rca_lib import (ANSWER_RESERVE, CAPPED_PAT_ROWS, DATA_DIR, MODELS_DEFAULT, SAMPLE12, SAMPLE50,
                     answer_reserve_for, direct_llm, ensure_only, gpu_memory_mb, model_supports_thinking,
                     ollama_models, ollama_unload, ollama_url, ollama_version, resolve_model, start_run,
                     step1_llm, step1_python, step2_llm, step2_python, step3_llm, step3_python)


def run(cases, models, label, notes="", use_llm=True, step2_rules=("naive", "rule"),
        step3_rules=("top1", "role"), staged=True, direct=(), keep_warm=False):
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
    for case in cases:  # always: the control chain (python step 1/2 -> py3:top1) is the baseline every run needs
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
            tag = model.split("/")[-1].split(":")[0]  # any model, not just our two
            for case in cases:
                for variant in direct:  # step-0 evidence straight to a final answer, no Python ranking
                    kw = {}
                    if variant == "capped":
                        kw["max_pat_rows"] = CAPPED_PAT_ROWS
                    elif variant == "facts":
                        kw["graph_facts"] = True
                    elif variant == "roles":
                        kw["roles_from"] = s2.get((case, "python:strength+py2:rule"))
                        if kw["roles_from"] is None:
                            continue
                    rd = direct_llm(case, model=model, thinking=False,
                                    variant="" if variant == "plain" else variant, **kw)
                    w.add(case, "direct", rd["meta"]["source"], rd)
                    print(f"direct:{variant:6s} {tag} {case:32s} answer={rd['meta']['answer']} "
                          f"{rd['meta']['attempts'][-1]['wall_s']}s", flush=True)
                if not staged:
                    continue
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
    for (case, chain), r2 in list(s2.items()):  # includes the control chain whenever Python step 1/2 ran
        for rule in step3_rules:
            r3 = step3_python(case, r2, rule=rule)
            w.add(case, "step3", r3["meta"]["source"], r3)

    if use_llm and not keep_warm:
        # Free the VRAM instead of leaving the last model resident until Ollama times it out. --keep-warm
        # skips this when the next run follows immediately (a reload costs ~18 s for a 26B model).
        for model in models:
            if ollama_unload(model):
                print(f"unloaded {model}", flush=True)
    summary = w.finalize(extra_meta={"cases": list(cases), "models": models if use_llm else [],
                                     "use_llm": use_llm, "command": "run_sample.py", "keep_warm": keep_warm})
    print(f"\nrecords: {w.n} | summary rows: {len(summary)} | {time.time() - t0:.0f}s")
    print(w.dir)
    return w.dir


def _print_installed(installed):
    if not installed:
        print("              (none installed - pull one, e.g. `ollama pull qwen2.5-coder:7b`)")
        return
    print("              installed models:")
    for m in installed:
        print(f"                {m['name']:32s} {m['size_gb']:>6.1f} GB")


def quickstart(models, case="re3ss_carts_f1_1"):
    """Check the setup, time one case, and print the commands for a real run. Runs one case only.
    Everything here degrades: a missing endpoint, tool or model is reported, never raised."""
    print("== checking the setup\n")

    version = ollama_version()
    installed = ollama_models()
    if not installed and version is None:
        print(f"  Ollama      NOT reachable at {ollama_url()}")
        print("              Start it (`ollama serve`), or set OLLAMA_HOST if it runs elsewhere,")
        print("              e.g. OLLAMA_HOST=192.168.1.10:11434")
        return
    print(f"  Ollama      {ollama_url()}" + (f", version {version}" if version else ", version unknown (older build)"))

    if not models:  # --models omitted: show the options rather than assuming ours
        print(f"  models      no --models given.")
        _print_installed(installed)
        chosen = [m for m in MODELS_DEFAULT if resolve_model(m)]
        if not chosen:
            print("\n  Pick one from the list and run:")
            print("    python run_sample.py --quickstart --models <model>")
            return
        models = chosen
        print(f"              defaulting to: {', '.join(models)}")

    resolved, missing = [], []
    for m in models:
        r = resolve_model(m)
        (resolved.append(r) if r else missing.append(m))
        print(f"  model       {m}: {'installed' if r else 'NOT INSTALLED'}" + (f" (as {r})" if r and r != m else ""))
    if missing:
        print(f"              pull it with: ollama pull {missing[0]}")
        _print_installed(installed)
        return
    models = resolved

    data = Path(DATA_DIR)
    if (data / "cases.parquet").exists():
        print(f"  dataset     {data} ({len(list(data.glob('re*')))} case folders)")
    else:
        print(f"  dataset     NOT FOUND at {data}")
        print("              Download RCAEval from https://huggingface.co/datasets/phamquiluan/RCAEval,")
        print("              or point RCA_DATA_DIR at an existing copy.")
        return

    gpu = gpu_memory_mb()
    print(f"  GPU         {gpu[0]} MB free of {gpu[2]} MB" if gpu
          else "  GPU         not detected (no nvidia-smi or rocm-smi). Fine - it is only recorded with each\n"
               "              result to show whether a run spilled to CPU; expect slower calls on CPU.")

    print()
    for m in models:
        thinking = model_supports_thinking(m)
        known = m in ANSWER_RESERVE
        print(f"  {m}: thinking capability = {thinking}")
        if thinking:
            print(f"  {' ' * len(m)}  runs with thinking OFF. Some thinking models never stop reasoning on these")
            print(f"  {' ' * len(m)}  prompts (gemma4:26b produced 52k characters and no answer). If answers come")
            print(f"  {' ' * len(m)}  back empty, check done_reason=length with a large thinking field.")
        print(f"  {' ' * len(m)}  answer reserve = {answer_reserve_for(m)} tokens"
              + ("" if known else "  (unknown model, using the default; add to ANSWER_RESERVE in rca_lib.py to change)"))

    print(f"\n== timing one case ({case}), cold then warm\n")
    timings = {}
    for m in models:
        ensure_only(m)
        t0 = time.time()
        r = direct_llm(case, model=m, thinking=False)   # cold: includes loading the model
        cold = time.time() - t0
        t0 = time.time()
        r = direct_llm(case, model=m, thinking=False)   # warm: what a run actually pays per call
        warm = time.time() - t0
        a = r["meta"]["attempts"][-1]
        timings[m] = warm
        print(f"  {m}: {cold:.1f}s cold (includes load), {warm:.1f}s warm  "
              f"(prompt {r['meta']['prompt_tokens']} tokens, num_ctx {a['num_ctx']}, answer {r['meta']['answer']})")
        if a.get("fallback"):
            print(f"  {' ' * len(m)}  server compatibility: {a['fallback']}")
        if r["meta"].get("parse_error"):
            print(f"  {' ' * len(m)}  could not parse the answer: {r['meta']['parse_error']}")

    per_call = sum(timings.values())
    print("\n== rough estimates, from the WARM time on this one case\n")
    est = lambda calls_per_case, cases: (per_call * calls_per_case * cases) / 60
    print(f"  12 cases, direct plain           ~{est(1, 12):.0f} min")
    print(f"  50 cases, direct plain           ~{est(1, 50):.0f} min")
    print(f"  50 cases, 3 direct variants      ~{est(3, 50):.0f} min")
    print(f"  12 cases, full staged pipeline   ~{est(7, 12):.0f} min   (7 model calls per case per model)")
    print("  TrainTicket cases add ~40 s each of one-off trace parsing, and each model swap costs one load.")
    print("  Prompt sizes vary by case (~700 to ~12k tokens), so a real run will differ from this estimate.")

    ms = " ".join(models)
    print("\n== commands\n")
    print(f"  python run_sample.py --models {ms} --direct plain --direct-only --label direct12")
    print(f"  python run_sample.py --models {ms} --preset50 --direct plain capped roles --direct-only --label direct50")
    print(f"  python run_sample.py --models {ms} --label full-pipeline")
    print(f"  python run_sample.py --no-llm --label python-only            # baseline, no model calls")
    print("\nNothing else has been run. Pick a command above when you are ready.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", help="case ids (default: the --preset sample)")
    ap.add_argument("--preset", default="sample12", choices=["sample12"], help="named case set")
    ap.add_argument("--models", nargs="*", default=None,
                    help="models to use; omit with --quickstart to list what is installed")
    ap.add_argument("--no-llm", action="store_true", help="Python arms only; makes no Ollama calls")
    ap.add_argument("--label", default="run", help="folder name suffix under results/")
    ap.add_argument("--notes", default="")
    ap.add_argument("--direct", nargs="*", default=[], choices=["plain", "capped", "roles", "facts"],
                    help="direct variants: plain, capped (~3k tokens), roles (+ step-2 role labels), facts (+ raw call graph)")
    ap.add_argument("--direct-only", action="store_true", help="skip the staged step-1/2/3 LLM arms")
    ap.add_argument("--preset50", action="store_true", help="use the 50-case stratified sample")
    ap.add_argument("--keep-warm", action="store_true",
                    help="leave the model loaded at the end (default: unload, freeing VRAM)")
    ap.add_argument("--quickstart", action="store_true",
                    help="check the setup, time one case, print run commands and estimates, then stop")
    a = ap.parse_args()
    if a.quickstart:
        quickstart(a.models or [], case=(a.cases or ["re3ss_carts_f1_1"])[0])
        return
    models = a.models or list(MODELS_DEFAULT)
    cases = a.cases or (SAMPLE50 if a.preset50 else [c for c, _ in SAMPLE12])
    direct = a.direct or (["plain"] if a.direct_only else [])
    run(cases, models, a.label, notes=a.notes, use_llm=not a.no_llm,
        staged=not a.direct_only, direct=direct, keep_warm=a.keep_warm)


if __name__ == "__main__":
    main()
