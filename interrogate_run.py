"""Recompute the derived figures quoted in FINDINGS, so none of them rests on a conversation.

Every number here was originally worked out by hand against the run records and then published. This script
is the missing half: it recomputes each one from committed records and read-only data, so a claim can be
checked or corrected rather than trusted.

    python interrogate_run.py                 # every section
    python interrogate_run.py agreement size  # named sections only

Sections:
    coverage    per-arm accuracy, the union table, and baseline/qwen agreement
    attractor   which services wrong answers land on
    confidence  the ceiling arm's confidence against correctness
    latency     median wall time per arm
    size        step-0 evidence size (lines, tokens) and the raw input it replaces
    keyword     what an Exception|Error filter keeps and drops on the 8 labelled cases
    rerank      would_demote: how often re-ranking would have helped
    abstain     whether abstentions land on cases the baseline gets wrong
    reload      how much wall time went on Ollama runner rebuilds
    borderline  the demoted-signal cases, with the magnitudes involved

What it does NOT cover, because the code or configuration that produced them no longer exists: the 1,456
false log edges from the old extractor, the ~26% context-bucket saving and the answer it flipped, gemma's 52k
characters of thinking, the reload timings measured by hand, the GPU footprints, and the +15% drift detection
limit. Those are marked in FINDINGS as one-off observations.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLAUDE_RUN = "results/20260924-192739-claude50-opus"
LOCAL_RUN = "results/20260923-102621-sample50-direct-variants"
STAGED_RUN = "results/20260923-083629-sample12-full-pipeline"
CONTROL = "python:strength+py2:rule+py3:top1"
CLAUDE_ARM = "claude-opus"
LOCAL_ARMS = {"qwen_plain": "direct:qwen2.5-coder:7b", "gemma_plain": "direct:gemma4:26b",
              "qwen_capped": "direct-capped:qwen2.5-coder:7b", "gemma_capped": "direct-capped:gemma4:26b",
              "qwen_roles": "direct-roles:qwen2.5-coder:7b", "gemma_roles": "direct-roles:gemma4:26b"}


def _arm(df, arm):
    return df[df.arm == arm].drop_duplicates("case").set_index("case")


def joined():
    """One row per case: status, truth, and each arm's answer and correctness."""
    c, l = pd.read_csv(f"{CLAUDE_RUN}/summary.csv"), pd.read_csv(f"{LOCAL_RUN}/summary.csv")
    arms = {"claude": _arm(c, CLAUDE_ARM), "control": _arm(c, CONTROL)}
    arms.update({k: _arm(l, v) for k, v in LOCAL_ARMS.items()})
    rows = []
    for case in [x for x in arms["claude"].index if x in arms["control"].index]:
        m = re.match(r"(re\d)(ob|ss|tt)_.+_(cpu|mem|disk|delay|loss|f\d)_\d+$", case)
        r = {"case": case, "status": arms["claude"].at[case, "status"],
             "truth": arms["claude"].at[case, "truth"], "suite": m.group(1) if m else "?",
             "system": m.group(2) if m else "?", "fault": m.group(3) if m else "?",
             "tokens": int(arms["claude"].at[case, "prompt_tokens"])}
        for name, d in arms.items():
            r[name] = bool(d.at[case, "correct"]) if case in d.index else None
            r[name + "_ans"] = d.at[case, "answer"] if case in d.index else None
        rows.append(r)
    return pd.DataFrame(rows)


def coverage(t):
    print("\n== coverage: per arm, the union table, and agreement")
    for status in ("clear", "weak"):
        sel = t[t.status == status]
        print(f"\n  {status} cases ({len(sel)})")
        for a in ["claude", "control", "qwen_plain", "gemma_plain", "qwen_capped", "gemma_capped",
                  "qwen_roles", "gemma_roles"]:
            d = sel[sel[a].notna()]
            if len(d):
                print(f"    {a:13s} {int(d[a].sum()):>2}/{len(d)}  {100*d[a].mean():5.1f}%")
    cle = t[t.status == "clear"]
    print("\n  union over the cheap arms (clear cases)")
    for label, v in [("control", cle.control), ("qwen", cle.qwen_plain),
                     ("control or qwen", cle.control | cle.qwen_plain),
                     ("control or gemma", cle.control | cle.gemma_plain),
                     ("control or qwen or gemma", cle.control | cle.qwen_plain | cle.gemma_plain),
                     ("claude", cle.claude)]:
        print(f"    {label:26s} {int(v.sum()):>2}/{len(cle)}  {100*v.mean():5.1f}%")
    ag = cle.control_ans.astype(str).str.lower() == cle.qwen_plain_ans.astype(str).str.lower()
    print(f"\n  baseline and qwen agree on {int(ag.sum())}/{len(cle)} clear cases; "
          f"the shared answer is right {int(cle[ag].control.sum())}/{int(ag.sum())}")
    dis = cle[~ag]
    print(f"  they disagree on {len(dis)}; one of the two is right "
          f"{int((dis.control | dis.qwen_plain).sum())}/{len(dis)} "
          f"({100*(dis.control | dis.qwen_plain).mean():.0f}%)")
    n = cle[["claude", "control", "qwen_plain", "gemma_plain", "qwen_capped", "gemma_capped"]].fillna(False)
    print(f"\n  clear cases solved by no arm: {int((~n.any(axis=1)).sum())}; by all six: "
          f"{int((n.sum(axis=1) == 6).sum())}; by exactly one: {int((n.sum(axis=1) == 1).sum())}")


def attractor(t):
    print("\n== attractor: where wrong answers land (clear cases)")
    cle = t[t.status == "clear"]
    for a in ["control", "qwen_plain", "gemma_plain", "claude"]:
        w = cle[~cle[a].fillna(True)]
        vc = w[a + "_ans"].value_counts()
        print(f"  {a:12s} {len(w):>2} wrong: " + ", ".join(f"{k} x{v}" for k, v in vc.items()))
    print("\n  Online Boutique only:")
    ob = cle[cle.system == "ob"]
    for a in ["control", "qwen_plain", "gemma_plain"]:
        w = ob[~ob[a].fillna(True)]
        vc = w[a + "_ans"].value_counts()
        top = f"{vc.index[0]} {vc.iloc[0]}/{len(w)}" if len(vc) else "-"
        print(f"    {a:12s} {len(w)} wrong of {len(ob)}, most often {top}")


def confidence(t):
    print("\n== confidence: the ceiling arm's stated confidence against correctness")
    c = pd.read_csv(f"{CLAUDE_RUN}/summary.csv")
    a = _arm(c, CLAUDE_ARM)
    x = pd.crosstab(a.confidence, a.correct)
    print("  " + x.to_string().replace("\n", "\n  "))
    for conf in a.confidence.dropna().unique():
        d = a[a.confidence == conf]
        print(f"    {conf:8s} {int(d.correct.sum())}/{len(d)} correct")


def latency(t):
    print("\n== latency: median wall seconds per arm")
    for run, arms in ((CLAUDE_RUN, {"claude": CLAUDE_ARM}), (LOCAL_RUN, LOCAL_ARMS)):
        s = pd.read_csv(f"{run}/summary.csv")
        for name, arm in arms.items():
            d = s[(s.arm == arm) & s.wall_s.notna()]
            if len(d):
                print(f"  {name:13s} median {d.wall_s.median():6.1f}s   total {d.wall_s.sum()/60:5.1f} min "
                      f"({len(d)} calls)")


def size(t):
    print("\n== size: what step 0 produces, and what it replaces")
    import rca_lib as R
    rows = []
    for case in R.SAMPLE50:
        txt = R.render_step0(case)
        txt = txt["text"] if isinstance(txt, dict) else txt
        rows.append({"case": case, "lines": len(txt.splitlines()), "tokens": R.count_tokens(txt)})
    d = pd.DataFrame(rows)
    print(f"  evidence over {len(d)} cases: lines median {d.lines.median():.0f} "
          f"(IQR {d.lines.quantile(.25):.0f}-{d.lines.quantile(.75):.0f}, range {d.lines.min()}-{d.lines.max()})")
    print(f"  tokens median {d.tokens.median():.0f} (range {d.tokens.min()}-{d.tokens.max()})")
    case = "re3ss_carts_f1_1"
    m = R.load_metrics(case)
    L = pd.read_parquet(f"{R.DATA_DIR}/{case}/logs.parquet")
    txt = R.render_step0(case)
    txt = txt["text"] if isinstance(txt, dict) else txt
    print(f"  {case}: {len(L)} log lines + metrics {m.shape[0]}x{m.shape[1]-1} -> "
          f"{len(txt.splitlines())} evidence lines")


def keyword(t):
    print("\n== keyword: what an Exception|Error filter keeps on the 8 labelled cases")
    import glob, os
    import rca_lib as R
    pat = re.compile(r"Exception|Error", re.I)
    kept = tot = 0
    shares = []
    for f in sorted(glob.glob(f"{R.DATA_DIR}/*/root_cause.txt")):
        case = os.path.basename(os.path.dirname(f))
        marker = open(f).read().strip()
        L = pd.read_parquet(f"{R.DATA_DIR}/{case}/logs.parquet")
        col = "message" if "message" in L.columns else L.columns[-1]
        txt = L[col].astype(str)
        hits = txt[txt.str.contains(pat)]
        tot += 1
        kept += any(marker[:30] in h for h in hits)
        if len(hits):
            share = hits.str.contains("queue-master|SocketException|socket", case=False).mean()
            shares.append(share)
            print(f"    {case:28s} {len(hits):>6} kept, {100*share:3.0f}% socket/queue-master noise")
    print(f"  the labelled root-cause line survived the filter in {kept} of {tot} cases")
    print(f"  noise share across cases: {100*min(shares):.0f}%-{100*max(shares):.0f}% "
          f"(median {100*float(np.median(shares)):.0f}%)")


def rerank(t):
    print("\n== rerank: what re-ranking by step 2 would have done (would_demote)")
    import glob
    tot = {"better": 0, "same": 0, "worse": 0}
    for f in sorted(glob.glob("results/*/step2.parquet")):
        d = pd.read_parquet(f)
        if "would_demote" not in d.columns or "truth" not in d.columns:
            continue
        for (case, src), g in d.groupby(["case", "run_source" if "run_source" in d.columns else "source"]):
            g = g.sort_values("rank")
            truth = g.truth.iat[0]
            before = g.service.iat[0] == truth
            kept = g[~g.would_demote.astype(bool)]
            after = (kept.service.iat[0] == truth) if len(kept) else False
            tot["better" if after and not before else "worse" if before and not after else "same"] += 1
    print(f"  over {sum(tot.values())} case/arm rankings in all committed runs: "
          f"better {tot['better']}, unchanged {tot['same']}, worse {tot['worse']}")


def abstain(t):
    print("\n== abstain: do abstentions land where the baseline is wrong?")
    s = pd.read_csv(f"{STAGED_RUN}/summary.csv")
    ctl = _arm(s, CONTROL)
    llm = s[s.arm.str.contains("llm3", na=False)]
    ab = llm[llm.abstained.astype(bool)]
    hit = sum(1 for r in ab.itertuples() if r.case in ctl.index and not bool(ctl.at[r.case, "correct"]))
    base_wrong = 1 - ctl.correct.mean()
    print(f"  staged run: {len(ab)} abstentions across the LLM deciders")
    print(f"  of those, the baseline would have been wrong on {hit}")
    print(f"  baseline's base rate of being wrong in that run: {100*base_wrong:.0f}%")
    for run, arm in ((CLAUDE_RUN, CLAUDE_ARM), (LOCAL_RUN, "direct:qwen2.5-coder:7b")):
        d = pd.read_csv(f"{run}/summary.csv")
        a = _arm(d, arm)
        for status in ("clear", "weak"):
            g = a[a.status == status]
            print(f"  {arm:26s} {status:5s}: {int(g.abstained.sum())} abstentions in {len(g)} calls")


def reload_cost(t):
    print("\n== reload: wall time against context-size changes")
    s = pd.read_csv(f"{LOCAL_RUN}/summary.csv")
    d = s[s.num_ctx.notna() & s.wall_s.notna() & s.model.notna()].copy()
    for model, g in d.groupby("model"):
        g = g.copy()
        g["changed"] = g.num_ctx != g.num_ctx.shift()
        ch, un = g[g.changed], g[~g.changed]
        print(f"  {model:22s} {len(g)} calls: {len(ch)} changed num_ctx "
              f"(median {ch.wall_s.median():5.1f}s), {len(un)} unchanged (median {un.wall_s.median():5.1f}s)")
    for arm_label, arm in (("plain", "direct:gemma4:26b"), ("capped", "direct-capped:gemma4:26b")):
        g = s[s.arm == arm].copy()
        if not len(g):
            continue
        reuse = (g.num_ctx == g.num_ctx.shift()).mean()
        print(f"  gemma {arm_label:6s}: context reused on {100*reuse:.0f}% of calls")


def borderline(t):
    print("\n== borderline: the demoted-signal cases")
    import rca_lib as R
    idx = R.load_index().set_index("case")
    for case in ("re1ob_adservice_loss_1", "re1ob_cartservice_loss_4"):
        ev, _ = R.step0_metrics(case)
        truth = idx.at[case, "root_cause_service"]
        top = ev.sort_values("z", ascending=False).head(3)
        print(f"  {case} (truth {truth})")
        for r in top.itertuples():
            mark = "  <- truth" if r.service == truth else ""
            print(f"    {r.service:24s} {r.metric:11s} z={r.z:9.1f} x{r.fold:<7} clear={r.clear}"
                  f" {r.evidence}{mark}")


SECTIONS = {"coverage": coverage, "attractor": attractor, "confidence": confidence, "latency": latency,
            "size": size, "keyword": keyword, "rerank": rerank, "abstain": abstain, "reload": reload_cost,
            "borderline": borderline}

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in SECTIONS] or list(SECTIONS)
    t = joined()
    print(f"joined {len(t)} cases from {CLAUDE_RUN} and {LOCAL_RUN} "
          f"({sum(t.status == 'clear')} clear, {sum(t.status == 'weak')} weak)")
    for name in want:
        try:
            SECTIONS[name](t)
        except Exception as e:  # one broken section must not cost the rest
            print(f"\n== {name}: FAILED {type(e).__name__}: {e}")
