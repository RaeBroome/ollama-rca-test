"""Issue #2: score the arms with and without the `diskio_appears` evidence.

`{svc}_diskio` appearing after injection identifies the root-cause service in 17 of 18 cases, including
code-level faults, which looks like the redeploy used to inject the fault rather than the fault itself. If it
is an artifact, every clear-case score may be inflated. This script measures that.

Two different things are reported, and the difference matters:

1. A true counterfactual for the PYTHON arms. `include_artifacts=False` drops diskio-appears symptoms before
   ranking, so the baseline can be re-scored exactly, deterministically, with no model calls and no change to
   the step-0 text any model saw.
2. A conditional decomposition for the LLM arms. Re-scoring a model's answer without the artifact would mean
   re-asking it with different evidence, which changes step 0 and invalidates the committed baselines. So for
   those arms this reports accuracy split by whether the artifact was present on the true root cause, plus how
   often each arm's own stated reason cites a disk signal - evidence about leaning on it, not a counterfactual.

    python analyse_diskio.py                     # the 50-case sample
    python analyse_diskio.py --cases a b c
"""
import argparse
import time

import pandas as pd

from rca_lib import (SAMPLE50, is_correct, load_index, step0_metrics, step1_python, step2_python,
                     step3_python)

CLAUDE_RUN = "results/20260924-192739-claude50-opus"
LOCAL_RUN = "results/20260923-102621-sample50-direct-variants"
CONTROL = "python:strength+py2:rule+py3:top1"


def artifact_facts(case, truth):
    """Which services carry a diskio_appears flag, and whether the truth is one of them.
    step0_metrics() is the same table step 0 renders from, so this asks the question of the real evidence."""
    ev, _ = step0_metrics(case)
    flagged = sorted(set(ev[ev.artifact_flag != ""].service)) if len(ev) else []
    truth_ev = ev[ev.service == truth] if len(ev) else ev
    truth_clear_without = bool(truth_ev[(truth_ev.artifact_flag == "") & truth_ev.clear].shape[0])         if len(truth_ev) else False
    return {"n_services_flagged": len(flagged), "flagged_services": ",".join(flagged),
            "truth_flagged": truth in flagged,
            "truth_still_clear_without_artifact": truth_clear_without}


def python_arms(case, truth):
    """The Python chain scored twice: with the artifact evidence and without it."""
    out = {}
    for tag, keep in (("with", True), ("without", False)):
        s1 = step1_python(case, order="strength", include_artifacts=keep)
        cands = s1["candidates"]
        out[f"step1_top1_{tag}"] = cands.service.iat[0] if len(cands) else None
        s2 = step2_python(case, s1, rule="rule")
        for rule in ("top1", "role"):
            s3 = step3_python(case, s2, rule=rule)
            ans = s3["meta"]["answer"]
            out[f"py3_{rule}_{tag}"] = ans
            out[f"py3_{rule}_{tag}_ok"] = bool(ans) and is_correct(ans, truth)
            out[f"py3_{rule}_{tag}_abstained"] = bool(s3["meta"]["abstained"])
    return out


def main(cases):
    idx = load_index().set_index("case")
    rows = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        truth = idx.at[case, "root_cause_service"]
        r = {"case": case, "truth": truth}
        r.update(artifact_facts(case, truth))
        r.update(python_arms(case, truth))
        rows.append(r)
        print(f"  [{i:>2}/{len(cases)}] {case:32s} flagged={r['n_services_flagged']:>2} "
              f"truth_flagged={str(r['truth_flagged']):5s} "
              f"top1 {r['step1_top1_with']} -> {r['step1_top1_without']}  {time.time()-t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)

    # clear/weak status and the LLM answers come from the committed runs
    c = pd.read_csv(f"{CLAUDE_RUN}/summary.csv")
    st = c[c.arm == CONTROL].drop_duplicates("case").set_index("case")
    d["status"] = d.case.map(st.status)
    d.to_csv("diskio_comparison.csv", index=False)

    print("\n" + "=" * 100)
    print("PART 1  Python arms: exact counterfactual (same cases, artifact evidence dropped before ranking)")
    for status in ("clear", "weak"):
        sel = d[d.status == status]
        if not len(sel):
            continue
        print(f"\n  {status.upper()} cases ({len(sel)})")
        for arm in ("py3_top1", "py3_role"):
            w, wo = sel[f"{arm}_with_ok"].sum(), sel[f"{arm}_without_ok"].sum()
            ab_w, ab_wo = sel[f"{arm}_with_abstained"].sum(), sel[f"{arm}_without_abstained"].sum()
            print(f"    {arm:9s} with {w:>2}/{len(sel)} ({100*w/len(sel):5.1f}%)   "
                  f"without {wo:>2}/{len(sel)} ({100*wo/len(sel):5.1f}%)   "
                  f"delta {wo-w:+d}   abstentions {ab_w} -> {ab_wo}")
        ch = sel[sel[f"py3_top1_with"] != sel[f"py3_top1_without"]]
        if len(ch):
            print(f"    answers that changed ({len(ch)}):")
            print(ch[["case", "truth", "py3_top1_with", "py3_top1_without", "truth_flagged"]]
                  .to_string(index=False).replace("\n", "\n      "))

    print("\n" + "=" * 100)
    print("PART 2  how far the artifact reaches")
    print(f"  cases with any diskio_appears service: {(d.n_services_flagged > 0).sum()}/{len(d)}")
    print(f"  cases where the TRUE root cause is flagged: {d.truth_flagged.sum()}/{len(d)}")
    print(f"  of those, truth still has clear non-artifact evidence: "
          f"{d[d.truth_flagged].truth_still_clear_without_artifact.sum()}/{d.truth_flagged.sum()}")
    print("\n  cases where the truth is flagged AND has no other clear evidence "
          "(the artifact is doing the work):")
    alone = d[d.truth_flagged & ~d.truth_still_clear_without_artifact]
    print(alone[["case", "status", "truth", "py3_top1_with", "py3_top1_without"]].to_string(index=False)
          if len(alone) else "    none")

    print("\n" + "=" * 100)
    print("PART 3  LLM arms: conditional split (NOT a counterfactual - see the module docstring)")
    loc = pd.read_csv(f"{LOCAL_RUN}/summary.csv")
    arms = [("claude-opus", c), ("direct:qwen2.5-coder:7b", loc), ("direct:gemma4:26b", loc),
            ("direct-capped:qwen2.5-coder:7b", loc), ("direct-capped:gemma4:26b", loc)]
    flag = d.set_index("case").truth_flagged
    for arm, src in arms:
        a = src[src.arm == arm].drop_duplicates("case").set_index("case")
        a = a[(a.status == "clear") & a.index.isin(flag.index)]  # only cases this run actually analysed
        f = [x for x in a.index if flag.get(x, False)]
        nf = [x for x in a.index if not flag.get(x, False)]
        if not f:
            continue
        print(f"  {arm:32s} truth flagged: {a.loc[f].correct.sum():>2}/{len(f)} "
              f"({100*a.loc[f].correct.mean():5.1f}%)   not flagged: {a.loc[nf].correct.sum():>2}/{len(nf)} "
              f"({100*a.loc[nf].correct.mean():5.1f}%)")
    print("\n  arms whose stated reason cites a disk signal (text search on reason_for_answer):")
    for arm, src in arms:
        a = src[src.arm == arm].drop_duplicates("case")
        cites = a.reason_for_answer.astype(str).str.contains("diskio|disk i/o|disk io", case=False)
        print(f"    {arm:32s} {cites.sum():>2}/{len(a)} answers mention disk")
    print(f"\nper-case table written to diskio_comparison.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()
    main(a.cases or SAMPLE50)
