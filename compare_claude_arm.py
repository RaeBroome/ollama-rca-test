"""Compare the Claude reference arm with the Python control and the local models, case by case.

    python compare_claude_arm.py results/<claude-run> [results/<local-run>]

The local run defaults to the committed 50-case direct-variants run, which holds qwen's and gemma's answers
for the same cases and the same step-0 version. Clear and weak cases are always reported separately, and
accuracy is never pooled across arms that saw different inputs.
"""
import sys
from pathlib import Path

import pandas as pd

CONTROL = "python:strength+py2:rule+py3:top1"
LOCAL_RUN_DEFAULT = "results/20260923-102621-sample50-direct-variants"


def load(run):
    s = pd.read_csv(Path(run) / "summary.csv")
    s["arm"] = s["arm"].astype(str)
    return s


def answers(s, arm):
    d = s[s.arm == arm].drop_duplicates("case").set_index("case")
    return d


def main(claude_run, local_run=LOCAL_RUN_DEFAULT):
    c = load(claude_run)
    claude_arms = sorted(a for a in c.arm.unique() if a.startswith("claude-"))
    if not claude_arms:
        print(f"no claude arm in {claude_run}")
        return
    ctl = answers(c, CONTROL)
    local = load(local_run) if Path(local_run).exists() else None

    for arm in claude_arms:
        cl = answers(c, arm)
        cases = [x for x in cl.index if x in ctl.index]
        for status in ("clear", "weak"):
            sel = [x for x in cases if cl.at[x, "status"] == status]
            if not sel:
                continue
            n = len(sel)
            cl_ok = sum(bool(cl.at[x, "correct"]) for x in sel)
            ct_ok = sum(bool(ctl.at[x, "correct"]) for x in sel)
            ab = sum(bool(cl.at[x, "abstained"]) for x in sel)
            print(f"\n=== {status.upper()} cases ({n})   arm: {arm}")
            print(f"  {arm:34s} {cl_ok:>3}/{n}  {100 * cl_ok / n:5.1f}%   abstentions {ab}")
            print(f"  {'control (' + CONTROL + ')':34s} {ct_ok:>3}/{n}  {100 * ct_ok / n:5.1f}%")
            if local is not None:
                for la in sorted(a for a in local.arm.unique() if a.startswith("direct:")):
                    lo = answers(local, la)
                    shared = [x for x in sel if x in lo.index]
                    if not shared:
                        continue
                    ok = sum(bool(lo.at[x, "correct"]) for x in shared)
                    note = "" if len(shared) == n else f"   (only {len(shared)} of these cases)"
                    print(f"  {la:34s} {ok:>3}/{len(shared)}  {100 * ok / len(shared):5.1f}%{note}")

        # cases Claude got that nothing else did
        if local is not None:
            locals_ = {la: answers(local, la) for la in local.arm.unique() if la.startswith("direct:")}
            rows = []
            for x in cases:
                others = {"control": bool(ctl.at[x, "correct"])}
                for la, lo in locals_.items():
                    if x in lo.index:
                        others[la] = bool(lo.at[x, "correct"])
                rows.append({"case": x, "status": cl.at[x, "status"], "truth": cl.at[x, "truth"],
                             "claude": cl.at[x, "answer"], "claude_ok": bool(cl.at[x, "correct"]),
                             "control_answer": ctl.at[x, "answer"], **others})
            d = pd.DataFrame(rows)
            only = d[d.claude_ok & ~d[[c for c in d.columns if c in ("control",) or c.startswith("direct:")]].any(axis=1)]
            print(f"\n=== cases {arm} got that the control and every local model missed: {len(only)}")
            if len(only):
                cols = ["case", "status", "truth", "control_answer"] + [c for c in d.columns if c.startswith("direct:")]
                print(only[cols].to_string(index=False))
            miss = d[~d.claude_ok & d["control"]]
            print(f"\n=== cases the control got that {arm} missed: {len(miss)}")
            if len(miss):
                print(miss[["case", "status", "truth", "claude"]].to_string(index=False))
            d.to_csv(Path(claude_run) / f"compare_{arm.replace(':', '-')}.csv", index=False)
            print(f"\nper-case table written to {Path(claude_run) / f'compare_{arm.replace(chr(58), chr(45))}.csv'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
