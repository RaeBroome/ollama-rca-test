"""Run the RCA pipeline (step 1 -> 2 -> 3) over a set of cases and write a results/ run folder.

Reproduce the committed 12-case run (it used both models; the default is now gemma4:26b only):

    python run_sample.py --models qwen2.5-coder:7b gemma4:26b --label sample12-full-pipeline

First time here? Check the setup and measure your hardware before committing to
a long run - this checks Ollama and the dataset, times one case, and prints the
commands with estimates, without running anything else:

    python run_sample.py --quickstart --models <your model>

Cheap variants (no Ollama calls, seconds rather than an hour):

    python run_sample.py --cases re3ss_carts_f1_1 --no-llm --label smoke
    python run_sample.py --no-llm --label python-only              # the default 12-case sample

Model calls are batched by model (one resident at a time: gemma4:26b alone takes
~19 GB of a 20 GB card). Records are written incrementally, so a crash
keeps everything produced up to that point.
"""
import argparse
import os
import sys
import time
from pathlib import Path

from rca_lib import (ANSWER_RESERVE, CAPPED_PAT_ROWS, CLAUDE_ARM_VERSION, CLAUDE_EFFORT,
                     CLAUDE_SYSTEM_PROMPT, DATA_DIR, MODELS_DEFAULT, SAMPLE12, SAMPLE50,
                     answer_reserve_for, claude_available, claude_direct, claude_sandbox, direct_llm,
                     ensure_only, gpu_memory_mb, model_supports_thinking, ollama_models, ollama_unload,
                     ollama_url, ollama_version, resolve_model, start_run, step1_llm, step1_python,
                     step2_llm, step2_python, step3_llm, step3_python)
import rca_lib

# ---------------------------------------------------------------- progress display
# Display only: nothing below changes what a run computes or writes to results/. It prints each stage as it
# starts, so a slow case visibly sits on the stage it is stuck in, and keeps a plain-text copy in
# logs/<run folder name>.txt so progress is never only in a terminal or a temp file.
LOGS_DIR = Path(__file__).resolve().parent / "logs"
SLOW_S = 20  # a case or a model call slower than this is shown in yellow
SYSTEMS = {"ob": "Online Boutique", "ss": "Sock Shop", "tt": "Train Ticket — slower"}
_ANSI = {"cyan": "36", "grey": "90", "green": "32", "yellow": "33", "red": "31"}


class Progress:
    """Timestamped lines, coloured on a terminal (NO_COLOR turns it off, FORCE_COLOR on), plain in the log."""

    def __init__(self, log_path):
        self.color = bool(os.environ.get("FORCE_COLOR")) or (sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
        if self.color and os.name == "nt":
            os.system("")  # turns on ANSI escape handling in a Windows console
        try:  # piped output on Windows defaults to cp1252, which garbles "—"; and an unprintable character must
            # never stop a run, so replace rather than raise
            sys.stdout.reconfigure(errors="replace", **({} if sys.stdout.isatty() else {"encoding": "utf-8"}))
        except Exception:
            pass
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = log_path
        self.f = open(log_path, "a", encoding="utf-8")

    def line(self, *parts, stamp=True):
        """parts: plain strings or (text, colour) pairs, so one segment of a line can be coloured."""
        parts = [p if isinstance(p, tuple) else (p, None) for p in parts]
        head = time.strftime("%H:%M:%S") + "  " if stamp else ""
        plain = head + "".join(t for t, _ in parts)
        shown = head + "".join(f"\033[{_ANSI[c]}m{t}\033[0m" if c and self.color else t for t, c in parts)
        print(shown, flush=True)
        self.f.write(plain + "\n")
        self.f.flush()

    def blank(self):
        print(flush=True)
        self.f.write("\n")
        self.f.flush()

    def section(self, title):
        self.blank()
        self.line((title, "cyan"))

    def case(self, i, n, case):
        sysname = SYSTEMS.get(case[3:5], "")  # re1ob_..., re2ss_..., re3tt_...
        self.line(f"[{i}/{n}] {case}  ", (f"({sysname})", "yellow" if case[3:5] == "tt" else None))

    def stage(self, text):
        self.line((f"  {text}", "grey"))

    def ok(self, text, secs=0.0):
        self.line((f"  {text}", "yellow" if secs > SLOW_S else "green"))

    def bad(self, text):
        self.line((f"  {text}", "red"))

    def close(self):
        self.f.close()


def _dur(s):
    s = int(round(s))
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}h {m}m {sec}s" if h else f"{m}m {sec}s" if m else f"{sec}s"


def data_shape(case):
    """What the case holds, from the parquet file footers only (row counts and column names; no data is read).
    rca_lib loads the data inside step 1 and keeps no copy, so the footers are the cheap way to say it here."""
    try:
        import pyarrow.parquet as pq
        info = rca_lib.case_info(case)
        d = Path(DATA_DIR) / case
        m = pq.ParquetFile(d / "metrics.parquet")
        services = len({c.rsplit("_", 1)[0] for c in m.schema_arrow.names if c != "time"})
        has_logs, has_traces = bool(info["has_logs"]), bool(info["has_traces"])
        out = f"Read: {m.metadata.num_rows:,}s of metrics across {services} services"  # one row per second
        if has_logs:
            out += f", {pq.ParquetFile(d / 'logs.parquet').metadata.num_rows:,} log lines"
        if has_traces:
            out += f", {pq.ParquetFile(d / 'traces.parquet').metadata.num_rows:,} trace spans"
        if not has_logs and not has_traces:
            out += "  (metrics only — no logs or traces)"
        elif not has_traces:
            out += "  (no traces)"
        elif not has_logs:
            out += "  (no logs)"
        return out
    except Exception as e:  # the display must never stop a run
        return f"Read: (could not read the file headers: {type(e).__name__})"


def _answered(p, res, secs):
    """One line for a model reply: green answer, red for no output / abstention / unusable reply."""
    m = res["meta"]
    if m.get("no_output"):
        tc = (m.get("attempts") or [{}])[-1].get("thinking_chars", "?")
        p.bad(f"No output — hit the token limit ({tc} thinking chars, {secs:.1f}s)")
    elif m.get("abstained"):
        p.bad(f"Abstained ({secs:.1f}s): {str(m.get('justification', ''))[:90]}")
    elif "answer" in m:
        p.ok(f"Answered: {m['answer']} ({secs:.1f}s)", secs)
    elif len(res["candidates"]):  # step 1 / step 2: a ranking, not a single answer
        p.ok(f"Answered: {len(res['candidates'])} suspects, top {res['candidates'].service.iloc[0]} ({secs:.1f}s)", secs)
    else:
        p.bad(f"No usable answer ({secs:.1f}s): {m.get('parse_error') or 'empty'}")


def run(cases, models, label, notes="", use_llm=True, step2_rules=("naive", "rule"),
        step3_rules=("top1", "role"), staged=True, direct=(), keep_warm=False, claude_models=(), think=False,
        skip_baseline=False):
    w = start_run(label, notes=notes or f"{len(cases)} cases, models={models if use_llm else 'none'}"
                  + (", baseline skipped" if skip_baseline else ""))
    print("run dir:", w.dir, flush=True)  # kept verbatim and first: sweep.py reads the run folder from it
    p = Progress(LOGS_DIR / f"{w.dir.name}.txt")
    p.f.write(f"run dir: {w.dir}\n")
    p.line(f"progress log: {p.path}")
    t0 = time.time()
    n = len(cases)
    s1, s2 = {}, {}
    # What each model actually ran with. Arm names only carry "-think" when thinking really ran, so a
    # --think run on a model without the capability is labelled as the no-think arm it is.
    thinking_by_model = {m: think and model_supports_thinking(m) for m in models} if use_llm else {}
    if think and use_llm:
        p.line(f"--think: thinking budget THINKING_EXTRA = {rca_lib.THINKING_EXTRA} tokens")
        for m, on in thinking_by_model.items():
            if not on:
                p.line((f"--think: {m} has no thinking capability - it runs with thinking OFF", "yellow"))

    def add_python_step2(case, s1key):
        for rule in step2_rules:
            r = step2_python(case, s1[(case, s1key)], rule=rule)
            s2[(case, f"{s1key}+py2:{rule}")] = r
            w.add(case, "step2", r["meta"]["source"], r)

    # ---- Python step 1 (+ its Python step 2 arms)
    # Stage lines are printed as each stage starts. They map onto the calls only roughly: the data is read
    # inside step1_python (rca_lib keeps no copy), so "Reading data" covers only the file-header summary and the
    # actual load falls under "Finding what changed"; and the onset-ordered step-1 pass under "Ranking
    # suspects" recomputes the symptoms before ranking them, so most of its time is not ranking.
    if skip_baseline:
        # No Python step 1/2/3 records. What remains is the scoring facts every written result needs (clear/weak
        # status and the retention checklist, which runs step-1 symptom detection for cases without a hand-written
        # one). Computing them here, cached per case, shows that time as its own stage instead of hiding it inside
        # the first model answer; the values are the ones w.add would compute anyway. The prompt's compression
        # (render_step0) happens inside direct_llm, under "Asking the model".
        p.section(f"Analyzing the data — no GPU, baseline skipped ({n} cases)")
        for i, case in enumerate(cases, 1):
            tc = time.time()
            p.case(i, n, case)
            try:
                p.stage("Reading data...")
                p.line(f"  {data_shape(case)}")
                p.stage("Finding what changed...")
                rca_lib._case_facts(case)
            except BaseException as e:
                p.bad(f"Failed: {type(e).__name__}: {e}")
                raise
            p.ok(f"Done ({time.time() - tc:.1f}s)", time.time() - tc)
            p.blank()
        p.line((f"Data analysis complete — {n} cases in {_dur(time.time() - t0)}", "cyan"))
    else:
        p.section(f"Analyzing the data — no GPU, this is the slow part ({n} cases)")
    for i, case in enumerate([] if skip_baseline else cases, 1):  # the control chain (python step 1/2 -> py3:top1) is the baseline
        tc = time.time()
        p.case(i, n, case)
        try:
            p.stage("Reading data...")
            p.line(f"  {data_shape(case)}")
            for order in ("strength", "onset"):
                p.stage("Finding what changed..." if order == "strength" else "Ranking suspects...")
                r = step1_python(case, order=order)
                s1[(case, f"python:{order}")] = r
                w.add(case, "step1", r["meta"]["source"], r)
            p.stage("Tracing the cause...")
            add_python_step2(case, "python:strength")
        except BaseException as e:
            p.bad(f"Failed: {type(e).__name__}: {e}")
            raise
        p.ok(f"Done ({time.time() - tc:.1f}s)", time.time() - tc)
        p.blank()
    if not skip_baseline:
        p.line((f"Data analysis complete — {n} cases in {_dur(time.time() - t0)}", "cyan"))

    # ---- one model resident at a time: its step 1, step 2 and step 3 for every chain it touches
    if use_llm:
        for model in models:
            ensure_only(model)
            tag = model.split("/")[-1].split(":")[0]  # any model, not just our two
            th = thinking_by_model[model]
            tm = time.time()
            p.section(f"Asking the model — {model} loaded, GPU working ({n} cases)" + (", thinking on" if th else ""))
            for i, case in enumerate(cases, 1):
                tc = time.time()
                p.case(i, n, case)
                try:
                    for variant in direct:  # step-0 evidence straight to a final answer, no Python ranking
                        kw = {}
                        if variant == "capped":
                            kw["max_pat_rows"] = CAPPED_PAT_ROWS
                        elif variant == "facts":
                            kw["graph_facts"] = True
                        elif variant == "roles":
                            kw["roles_from"] = s2.get((case, "python:strength+py2:rule"))
                            if kw["roles_from"] is None:
                                p.line(("  Skipped direct:roles — no Python step-2 result for this case", "yellow"))
                                continue
                        p.stage(f"Asking the model (direct: {variant})...")
                        tq = time.time()
                        rd = direct_llm(case, model=model, thinking=th,
                                        variant="" if variant == "plain" else variant, **kw)
                        w.add(case, "direct", rd["meta"]["source"], rd)
                        _answered(p, rd, time.time() - tq)
                    if staged:
                        p.stage("Asking the model (step 1: find and rank suspects)...")
                        tq = time.time()
                        r = step1_llm(case, model=model, thinking=th)
                        s1[(case, tag)] = r
                        w.add(case, "step1", r["meta"]["source"], r)
                        _answered(p, r, time.time() - tq)
                        p.stage("Tracing the cause (Python rules on the model's ranking)...")
                        add_python_step2(case, tag)
                        for s1key in ("python:strength", tag):
                            p.stage(f"Asking the model (step 2: trace the cause from {s1key})...")
                            tq = time.time()
                            r2 = step2_llm(case, s1[(case, s1key)], model=model, thinking=th)
                            s2[(case, f"{s1key}+llm2:{tag}")] = r2
                            w.add(case, "step2", r2["meta"]["source"], r2)
                            _answered(p, r2, time.time() - tq)
                        for chain in (f"python:strength+py2:rule", f"python:strength+llm2:{tag}",
                                      f"{tag}+py2:rule", f"{tag}+llm2:{tag}"):
                            if (case, chain) in s2:
                                p.stage(f"Asking the model (step 3: decide, {chain})...")
                                tq = time.time()
                                r3 = step3_llm(case, s2[(case, chain)], model=model, thinking=th)
                                w.add(case, "step3", r3["meta"]["source"], r3)
                                _answered(p, r3, time.time() - tq)
                except BaseException as e:
                    p.bad(f"Failed: {type(e).__name__}: {e}")
                    raise
                p.ok(f"Done ({time.time() - tc:.1f}s)", time.time() - tc)
                p.blank()
            p.line((f"Model answers complete — {n} cases in {_dur(time.time() - tm)}", "cyan"))

    # ---- the Claude reference arm (subscription CLI, no Ollama involvement, no GPU)
    claude_meta = {}
    if claude_models:
        ok, detail = claude_available()
        p.section(f"Claude reference arm — {'ready' if ok else 'SKIPPED'}: {detail}")
        claude_meta = {"available": ok, "detail": detail, "cli_version": rca_lib.CLAUDE_CLI_VERSION,
                       "effort": CLAUDE_EFFORT, "system_prompt": CLAUDE_SYSTEM_PROMPT,
                       "arm_version": CLAUDE_ARM_VERSION, "sandbox_cwd": str(claude_sandbox()),
                       "flags": ["-p", "--disallowed-tools *", "--strict-mcp-config", "--system-prompt",
                                 "--effort", "--output-format json"],
                       "temperature": "NOT controllable via the CLI - this arm is not pinned to "
                                      "temperature 0 like the Ollama arms",
                       "auth": "the machine's Claude subscription login (OAuth), no API key",
                       "models_requested": list(claude_models), "model_ids": [], "cost_usd_list": 0.0}
        if ok:
            for cm in claude_models:
                for i, case in enumerate(cases, 1):
                    rc = claude_direct(case, model=cm)
                    w.add(case, "direct", rc["meta"]["source"], rc)
                    a = rc["meta"]["attempts"][-1]
                    if a.get("model_id") and a["model_id"] not in claude_meta["model_ids"]:
                        claude_meta["model_ids"].append(a["model_id"])
                    claude_meta["cost_usd_list"] += (a.get("cost_usd_list") or 0)
                    p.line(f"claude:{cm:6s} [{i:>2}/{len(cases)}] {case:32s} "
                           f"answer={rc['meta']['answer']} {a.get('wall_s')}s",
                           ((f" ERROR {a['error']}", "red") if a.get("error") else ""))

    # ---- Python step 3 on every chain (free)
    if skip_baseline:
        p.blank()
        p.line(("Python step 3 skipped — there is no baseline chain to decide on (--skip-baseline)", "grey"))
    else:
        p.section("Deciding with the Python rules (step 3, every chain)")
        tq = time.time()
        for (case, chain), r2 in list(s2.items()):  # includes the control chain whenever Python step 1/2 ran
            for rule in step3_rules:
                r3 = step3_python(case, r2, rule=rule)
                w.add(case, "step3", r3["meta"]["source"], r3)
        p.ok(f"Done ({time.time() - tq:.1f}s)")

    if use_llm and not keep_warm:
        # Free the VRAM instead of leaving the last model resident until Ollama times it out. --keep-warm
        # skips this when the next run follows immediately (a reload costs ~18 s for a 26B model).
        for model in models:
            if ollama_unload(model):
                p.line(f"unloaded {model}")
    summary = w.finalize(extra_meta={"cases": list(cases), "models": models if use_llm else [],
                                     "use_llm": use_llm, "command": "run_sample.py", "keep_warm": keep_warm,
                                     "think_requested": think, "thinking_by_model": thinking_by_model,
                                     # a skipped baseline means no Python step 1/2/3 records and no control chain:
                                     # this folder is not a full run and cannot be compared against its own control
                                     "baseline_skipped": skip_baseline,
                                     **({"claude_arm": claude_meta} if claude_meta else {})})
    p.blank()
    p.line((f"Run complete — records: {w.n} | summary rows: {len(summary)} | {_dur(time.time() - t0)}", "cyan"))
    p.line(str(w.dir))
    p.close()
    return w.dir


def _print_installed(installed):
    if not installed:
        print("              (none installed - pull one, e.g. `ollama pull gemma4:26b`)")
        return
    print("              installed models:")
    for m in installed:
        print(f"                {m['name']:32s} {m['size_gb']:>6.1f} GB")


def quickstart(models, case="re3ss_carts_f1_1", think=False, think_flags=""):
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
    thinking_by_model = {}
    for m in models:
        capable = model_supports_thinking(m)
        thinking_by_model[m] = think and capable
        known = m in ANSWER_RESERVE
        pad = " " * len(m)
        print(f"  {m}: thinking capability = {capable}")
        if think and capable:
            print(f"  {pad}  runs with thinking ON (--think), budget {rca_lib.THINKING_EXTRA} extra tokens.")
            print(f"  {pad}  Some thinking models never stop reasoning on these prompts (gemma4:26b produced")
            print(f"  {pad}  52k characters and no answer); those calls are recorded as no_output, not wrong.")
        elif think:
            print(f"  {pad}  --think requested, but this model cannot think: it runs with thinking OFF.")
        elif capable:
            print(f"  {pad}  runs with thinking OFF (pass --think to turn it on).")
        print(f"  {pad}  answer reserve = {answer_reserve_for(m, thinking_by_model[m])} tokens"
              + ("" if known else "  (unknown model, using the default; add to ANSWER_RESERVE in rca_lib.py to change)"))

    print(f"\n== timing one case ({case}), cold then warm\n")
    timings = {}
    for m in models:
        ensure_only(m)
        t0 = time.time()
        r = direct_llm(case, model=m, thinking=thinking_by_model[m])   # cold: includes loading the model
        cold = time.time() - t0
        t0 = time.time()
        r = direct_llm(case, model=m, thinking=thinking_by_model[m])   # warm: what a run actually pays per call
        warm = time.time() - t0
        a = r["meta"]["attempts"][-1]
        timings[m] = warm
        print(f"  {m}: {cold:.1f}s cold (includes load), {warm:.1f}s warm  "
              f"(prompt {r['meta']['prompt_tokens']} tokens, num_ctx {a['num_ctx']}, answer {r['meta']['answer']})")
        if a.get("fallback"):
            print(f"  {' ' * len(m)}  server compatibility: {a['fallback']}")
        if r["meta"].get("no_output"):
            print(f"  {' ' * len(m)}  NO OUTPUT: hit the token limit with empty content "
                  f"({a['thinking_chars']} thinking chars) - try a larger --think-budget")
        elif r["meta"].get("parse_error"):
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

    ms = " ".join(models) + think_flags
    print("\n== commands\n")
    print(f"  python run_sample.py --models {ms} --direct plain --direct-only --label direct12")
    print(f"  python run_sample.py --models {ms} --preset50 --direct plain capped roles --direct-only --label direct50")
    print(f"  python run_sample.py --models {ms} --label full-pipeline")
    print(f"  python run_sample.py --no-llm --label python-only            # baseline, no model calls")
    print("\nNothing else has been run. Pick a command above when you are ready.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", help="case ids (default: the 12-case sample)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="models to use; omit with --quickstart to list what is installed")
    ap.add_argument("--no-llm", action="store_true", help="Python arms only; makes no Ollama calls")
    ap.add_argument("--label", default="run", help="folder name suffix under results/")
    ap.add_argument("--notes", default="")
    ap.add_argument("--direct", nargs="*", default=[], choices=["plain", "capped", "roles", "facts"],
                    help="direct variants: plain, capped (~2k tokens), roles (+ step-2 role labels), facts (+ raw call graph)")
    ap.add_argument("--direct-only", action="store_true", help="skip the staged step-1/2/3 LLM arms")
    ap.add_argument("--preset50", action="store_true", help="use the 50-case stratified sample")
    ap.add_argument("--claude", nargs="*", default=None, metavar="ALIAS",
                    help="also run the Claude reference arm via the Claude Code CLI on this machine's "
                         "subscription (e.g. --claude opus sonnet); no API key, no temperature control")
    ap.add_argument("--keep-warm", action="store_true",
                    help="leave the model loaded at the end (default: unload, freeing VRAM)")
    ap.add_argument("--quickstart", action="store_true",
                    help="check the setup, time one case, print run commands and estimates, then stop")
    ap.add_argument("--think", action="store_true",
                    help="turn thinking on for models that have the capability (default off; models without it "
                         "run with thinking off and their arms are not labelled -think)")
    ap.add_argument("--think-budget", type=int, metavar="N",
                    help=f"extra output tokens reserved for thinking (THINKING_EXTRA, default "
                         f"{rca_lib.THINKING_EXTRA}); needs --think")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip the Python step-1/2/3 baseline chain (~36 min for 50 cases) and run only compression "
                         "-> the direct model arm; needs --direct-only, and not with --direct roles")
    a = ap.parse_args()
    if a.skip_baseline:  # refuse the combinations that would need the baseline, rather than produce wrong output
        if not a.direct_only:
            ap.error("--skip-baseline needs --direct-only: the staged LLM arms run on the Python step 1 it skips")
        if "roles" in (a.direct or []):
            ap.error("--skip-baseline cannot run --direct roles: that variant is built from the Python step-2 "
                     "output it skips. Drop roles, or drop --skip-baseline")
        if a.no_llm and a.claude is None:
            ap.error("--skip-baseline with --no-llm leaves nothing to run")
    think_flags = ""
    if a.think_budget is not None:
        if not a.think:
            ap.error("--think-budget needs --think")
        if a.think_budget <= 0:
            ap.error("--think-budget must be a positive number of tokens")
        rca_lib.THINKING_EXTRA = a.think_budget  # read at call time by answer_reserve_for; recorded in metadata
        think_flags = f" --think --think-budget {a.think_budget}"
    elif a.think:
        think_flags = " --think"
    if a.quickstart:
        quickstart(a.models or [], case=(a.cases or ["re3ss_carts_f1_1"])[0], think=a.think,
                   think_flags=think_flags)
        return
    models = a.models or list(MODELS_DEFAULT)
    cases = a.cases or (SAMPLE50 if a.preset50 else [c for c, _ in SAMPLE12])
    direct = a.direct or (["plain"] if a.direct_only else [])
    run(cases, models, a.label, notes=a.notes, use_llm=not a.no_llm,
        staged=not a.direct_only, direct=direct, keep_warm=a.keep_warm,
        claude_models=() if a.claude is None else (a.claude or ["opus"]), think=a.think,
        skip_baseline=a.skip_baseline)


if __name__ == "__main__":
    main()
