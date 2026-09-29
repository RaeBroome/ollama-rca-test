"""Summarise a results/ run folder.

    python analyse_run.py results/<run folder>

Every decider is broken out BY STEP-1 SOURCE. Pooling across step-1 sources is
wrong here: py3:top1 returns step 1's top candidate, so its accuracy is that
chain's step-1 recall@1, and pooling 70% (python) with 40% (qwen) produced a
meaningless 58%.

no_output rows (a model call hit its token limit with empty content, usually a thinking
model that never finished reasoning) are neither wrong nor abstentions: they are left
out of every percentage and counted separately, so a run that produced nothing reads
as "output in 0 of N", not as 0% accuracy.
"""
import sys

import pandas as pd


def block(g):
    n = len(g)
    no = int(g.no_output.sum())
    g = g[~g.no_output]  # percentages are over the calls that produced output
    k = len(g)
    ab = int(g.abstained.sum())
    ans = g[~g.abstained]
    return pd.Series({"decisions": n, "no_output": no, "correct": int(g.correct.sum()),
                      "accuracy_%": round(100 * g.correct.mean()) if k else float("nan"),
                      "abstained": ab, "abstention_%": round(100 * ab / k) if k else float("nan"),
                      "accuracy_when_answered_%": round(100 * ans.correct.mean()) if len(ans) else float("nan")})


def pct(x):
    return round(100 * x.mean()) if len(x) else float("nan")


def main(run):
    s = pd.read_csv(f"{run}/summary.csv")
    # runs written before the no_output outcome existed have no such column: none of their rows are
    s["no_output"] = s["no_output"].fillna(False).astype(bool) if "no_output" in s else False
    pd.set_option("display.width", 250)
    s3 = s[s.stage == "step3"].copy()
    if len(s3):
        s3["step1_source"] = s3.arm.str.split("+").str[0]
        s3["decider"] = s3.arm.str.rsplit("+", n=1).str[1]
        s3["chain"] = s3.arm.str.rsplit("+", n=1).str[0]
    direct = s[s.stage == "direct"].copy()

    for status in ["clear", "weak"]:
        sub = s3[s3.status == status] if len(s3) else s3
        if not len(sub):
            continue
        print(f"\n================ {status.upper()} cases ({sub.case.nunique()} cases)")
        print("--- decider x step-1 source (never pooled across sources)")
        print(sub.groupby(["step1_source", "decider"]).apply(block, include_groups=False).to_string())
        if len(direct):
            d = direct[direct.status == status]
            if len(d):
                print("--- direct: step-0 evidence straight to the model, no Python ranking")
                print(d.groupby("arm").apply(block, include_groups=False).to_string())

    if len(s3):
        print("\n================ vs the control (python:strength + py3:top1), clear cases")
        clear = s3[s3.status == "clear"]
        ctrl = clear[(clear.step1_source == "python:strength") & (clear.decider == "py3:top1")]
        ctrl_acc = pct(ctrl[~ctrl.no_output].correct)
        rows = [{"configuration": "python:strength + py3:top1 (control)", "accuracy_%": ctrl_acc,
                 "decisions": len(ctrl), "no_output": int(ctrl.no_output.sum()), "abstention_%": 0}]
        for (src, dec), g in clear.groupby(["step1_source", "decider"]):
            if (src, dec) == ("python:strength", "py3:top1"):
                continue
            out = g[~g.no_output]
            rows.append({"configuration": f"{src} + {dec}", "accuracy_%": pct(out.correct),
                         "decisions": len(g), "no_output": int(g.no_output.sum()),
                         "abstention_%": pct(out.abstained)})
        if len(direct):
            for arm, g in direct[direct.status == "clear"].groupby("arm"):
                out = g[~g.no_output]
                rows.append({"configuration": f"{arm} (no Python ranking)", "accuracy_%": pct(out.correct),
                             "decisions": len(g), "no_output": int(g.no_output.sum()),
                             "abstention_%": pct(out.abstained)})
        t = pd.DataFrame(rows).sort_values("accuracy_%", ascending=False)
        t["beats_control"] = t["accuracy_%"] > ctrl_acc
        print(t.to_string(index=False))

    if len(s3) and s3.abstained.any():
        print("\n================ abstention quality (is it selective?)")
        top1 = s3[s3.decider == "py3:top1"].set_index(["chain", "case"])[["correct"]].rename(
            columns={"correct": "top1_correct"})
        ab = s3[s3.abstained].set_index(["chain", "case"]).join(top1, how="left").reset_index()
        print(ab.groupby(["decider", "status"]).apply(
            lambda g: pd.Series({"abstentions": len(g),
                                 "top1_would_be_wrong": int((~g.top1_correct.fillna(False)).sum()),
                                 "top1_would_be_right": int(g.top1_correct.fillna(False).sum())}),
            include_groups=False).to_string())
        base = s3[(s3.decider == "py3:top1") & ~s3.no_output]
        print(f"base rate: py3:top1 wrong in {int((~base.correct).sum())}/{len(base)} "
              f"({round(100 * (~base.correct).mean())}%) - an abstention is only selective if it beats this")

    if len(direct) and direct.abstained.any():
        d = direct[direct.abstained]
        print(f"\ndirect-arm abstentions: {len(d)} ({', '.join(sorted(set(d.case)))})")

    scored = pd.concat([s3, direct])
    if len(scored) and scored.no_output.any():
        print("\n================ no output (hit the token limit with empty content; not wrong, not an abstention)")
        for arm, g in scored.groupby("arm"):
            if g.no_output.any():
                print(f"  {arm}: output in {int((~g.no_output).sum())} of {len(g)} "
                      f"({int(g.no_output.sum())} no output: {', '.join(sorted(set(g[g.no_output].case)))})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sorted(__import__("glob").glob("results/*"))[-1])
