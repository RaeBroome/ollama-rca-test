"""Issue #2: how often does `{svc}_diskio` appearing after injection point at the root cause?

`explore.ipynb` section 7 claims it identifies the root-cause service in 17 of 18 cases, and FINDINGS repeats
it, but no committed code recomputes it. This scans the whole benchmark cheaply: for each case, only the
`*_diskio` columns are evaluated (not the full step-0 pass), so it is ~1 s per case rather than ~20.

    python scan_diskio.py                 # every case in cases.parquet
    python scan_diskio.py --limit 60      # a quick subset, in index order

Writes diskio_scan.csv: one row per case that has any diskio column, with which services show one appearing.
"""
import argparse
import time

import pandas as pd

from rca_lib import EXCLUDE, load_index, load_inject_time, load_metrics, metric_evidence


def scan_case(case, truth):
    m = load_metrics(case)
    t = load_inject_time(case)
    cols = [c for c in m.columns if c.endswith("_diskio")]
    appears = []
    for c in cols:
        ev = metric_evidence(m, c, t)
        if ev["artifact_flag"] == "diskio_appears":
            appears.append(c[: -len("_diskio")])
    return {"case": case, "truth": truth, "n_diskio_cols": len(cols),
            "n_appears": len(appears), "appears_services": ",".join(sorted(appears)),
            "truth_appears": truth in appears,
            "only_truth_appears": bool(appears) and set(appears) == {truth}}


def main(limit=None):
    idx = load_index()
    if limit:
        idx = idx.head(limit)
    rows, t0 = [], time.time()
    for i, r in enumerate(idx.itertuples(), 1):
        if r.case in EXCLUDE:
            continue
        try:
            rows.append(scan_case(r.case, r.root_cause_service))
        except Exception as e:  # a missing or unreadable case must not lose the scan
            rows.append({"case": r.case, "truth": r.root_cause_service, "n_diskio_cols": None,
                         "n_appears": None, "appears_services": f"ERROR {type(e).__name__}",
                         "truth_appears": None, "only_truth_appears": None})
        if i % 50 == 0:
            print(f"  {i}/{len(idx)} cases, {time.time()-t0:.0f}s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("diskio_scan.csv", index=False)
    ok = d[d.n_appears.notna()]
    any_app = ok[ok.n_appears > 0]

    print("\n" + "=" * 96)
    print(f"scanned {len(ok)} cases ({len(d) - len(ok)} unreadable), {time.time()-t0:.0f}s")
    print(f"  cases with at least one diskio metric column:   {(ok.n_diskio_cols > 0).sum()}")
    print(f"  cases where some service's diskio APPEARS:      {len(any_app)}")
    if not len(any_app):
        return
    print(f"  of those, the true root cause is one of them:   {any_app.truth_appears.sum()}"
          f"  ({100*any_app.truth_appears.mean():.0f}%)   <- the 17-of-18 claim")
    print(f"  of those, the truth is the ONLY one:            {any_app.only_truth_appears.sum()}"
          f"  ({100*any_app.only_truth_appears.mean():.0f}%)")
    svc_total = any_app.appears_services.str.split(",").map(len).sum()
    print(f"  precision over flagged services (not cases):    {any_app.truth_appears.sum()}/{svc_total}"
          f"  ({100*any_app.truth_appears.sum()/svc_total:.0f}%)")

    any_app = any_app.copy()
    any_app["suite"] = any_app.case.str.extract(r"^(re\d)")[0]
    any_app["fault"] = any_app.case.str.extract(r"_(cpu|mem|disk|delay|loss|f\d)_\d+$")[0]
    print("\n  by fault type (does it hold for non-disk and code faults?)")
    g = any_app.groupby("fault", observed=True).agg(cases=("truth_appears", "size"),
                                                   truth_flagged=("truth_appears", "sum"))
    g["rate_%"] = (100 * g.truth_flagged / g.cases).round(0)
    print("    " + g.to_string().replace("\n", "\n    "))
    print("\n  by suite")
    g2 = any_app.groupby("suite", observed=True).agg(cases=("truth_appears", "size"),
                                                    truth_flagged=("truth_appears", "sum"))
    g2["rate_%"] = (100 * g2.truth_flagged / g2.cases).round(0)
    print("    " + g2.to_string().replace("\n", "\n    "))

    print("\n  how much of the benchmark is even exposed to this:")
    print(f"    {len(any_app)}/{len(ok)} cases ({100*len(any_app)/len(ok):.0f}%) contain an appearing diskio")
    print(f"    {any_app.truth_appears.sum()}/{len(ok)} cases ({100*any_app.truth_appears.sum()/len(ok):.0f}%) "
          f"have it on the true root cause")
    print("\nwrote diskio_scan.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
