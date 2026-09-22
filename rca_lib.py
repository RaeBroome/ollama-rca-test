"""Settled, reusable functions for the Ollama RCA evaluation on RCAEval.

Exploration lives in explore.ipynb; anything here has been reviewed and is used from there.
RCAEval-data/ is read-only. Nothing in this module writes files.

CLI:  python rca_lib.py inspect <case> [--no-artifacts] [--max-pat-rows N]
"""
import csv
import os
import re
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("RCA_DATA_DIR", str(Path(__file__).resolve().parent / "RCAEval-data"))


# ============================================================ data access
@lru_cache(maxsize=1)
def load_index():
    return pd.read_parquet(f"{DATA_DIR}/cases.parquet")


def case_info(case):
    return load_index().set_index("case").loc[case]


def load_metrics(case):
    return pd.read_parquet(f"{DATA_DIR}/{case}/metrics.parquet")


def load_inject_time(case):
    with open(f"{DATA_DIR}/{case}/inject_time.txt") as f:
        return int(f.read().strip())


def read_root_cause(case):
    """RE3-SS root_cause.txt: one CSV row (HH:MM, ns timestamp, container, message, pod, node)."""
    f = next(csv.reader(open(f"{DATA_DIR}/{case}/root_cause.txt", encoding="utf-8")))
    return {"hhmm": f[0], "ts": int(f[1]) // 10**9, "container": f[2], "message": f[3], "pod": f[4]}


# ============================================================ cases that must never be scored (static part)
EXCLUDE = {  # broken data: no usable analysis at all
    "re1ob_currencyservice_loss_1": "inject_time in index/txt is 16933142 (missing digits) - no normal period",
    "re1ob_productcatalogservice_cpu_3": "inject_time is after the last metric row - no fault period",
}
BROKEN_LABELS = {  # analysable, but the label evidence is broken
    "re3ss_front-end_f2_2": ("root_cause.txt is a byte-for-byte copy of re3ss_front-end_f2_1's; "
                             "its line is 3 h before inject_time and absent from this case's logs"),
}
# "not diagnosable from available data" cases are computed in explore.ipynb section 7 (diagnosability_status).


# ============================================================ metric columns
COL_FALLBACKS = {
    "cpu": ["cpu"], "mem": ["mem"], "diskio": ["diskio"], "socket": ["socket"], "error": ["error"],
    "latency": ["latency-90", "latency", "latency-50"],
    "load": ["workload", "load"],
}
FAULT_METRIC = {  # fault -> ordered list of metric kinds to try
    "cpu": ["cpu"], "mem": ["mem"], "delay": ["latency"], "loss": ["latency"],
    "disk": ["diskio", "latency"], "socket": ["socket"],
}


def resolve_col(df, svc, kind):
    for suffix in COL_FALLBACKS[kind]:
        c = f"{svc}_{suffix}"
        if c in df.columns:
            return c
    return None


def latency_col(df, service="adservice"):
    return resolve_col(df, service, "latency")


def pick_fault_metric(df, svc, fault):
    """Returns (column, note). note records any fallback used, so it can't go unnoticed."""
    kinds = FAULT_METRIC[fault]
    for i, kind in enumerate(kinds):
        c = resolve_col(df, svc, kind)
        if c:
            return c, ("" if i == 0 else f"fallback: no {kinds[0]}, used {kind}")
    return None, f"none of {kinds} present for {svc}"


# ============================================================ metric shape (explore.ipynb section 3)
# Even an instantaneous fault (tc-netem delay) takes ~36-57 s to fully show in latency-90 (trailing-window
# metric). Anything that settles within ~2x this lag is indistinguishable from a step.
RESPONSE_LAG_S = 60
STEP_PLATEAU_MAX_S = 2 * RESPONSE_LAG_S
STARTUP_SKIP_S = 30  # some series start with a collection spike


def robust_scale(s, level_floor=0.01):
    """Baseline noise: MAD-based std, floored at 1% of the baseline level."""
    s = s.dropna()
    mad_std = 1.4826 * (s - s.median()).abs().median()
    return max(mad_std, level_floor * abs(s.median()), 1e-9)


def shape_signature_v2(metrics_df, col, inject_time, late_frac=0.25, smooth_s=30, plateau_level=0.9,
                       plateau_ref="fault_median"):
    """Unit-free shape of one metric around inject_time. Baseline = whole normal period (minus STARTUP_SKIP_S),
    fault = whole faulty period.

      total_z           final shift in baseline-noise units ("is it real?")
      log2_fold         log2(final / baseline) ("how big?")
      early_share       share of the final shift present in the first RESPONSE_LAG_S
      late_trend_share  trend after RESPONSE_LAG_S as a share of the final shift (~0 step, ~1 ramp, <0 receding)
      late_trend_rho    Spearman rho of value vs time over the same period (signed to the shift direction)
      time_to_plateau_s first time the smoothed series reaches 90% of the fault-period median level
      pre_drift_share   SANITY CHECK on inject_time only
    Returns None if either side has < 10 usable points (presence_signature reports that case).
    """
    df = metrics_df[["time", col]].dropna()
    if df.empty:
        return None
    pre = df[(df.time < inject_time) & (df.time >= df.time.iloc[0] + STARTUP_SKIP_S)]
    post = df[df.time >= inject_time]
    if len(pre) < 10 or len(post) < 10:
        return None

    base = pre[col].median()
    scale = robust_scale(pre[col])
    z_post = (post[col].values - base) / scale
    t_post = (post.time.values - inject_time).astype(float)
    fault_s = t_post[-1]

    n_late = max(int(len(post) * late_frac), 5)
    final_level = np.median(post[col].values[-n_late:])
    total_z = (final_level - base) / scale
    eps = 1e-9 + 1e-6 * abs(base)
    log2_fold = np.log2((abs(final_level) + eps) / (abs(base) + eps))
    change = final_level - base

    early = post[col].values[t_post < RESPONSE_LAG_S]
    early_share = (np.median(early) - base) / change if len(early) and change else np.nan

    after = t_post >= RESPONSE_LAG_S
    late_trend_share = late_trend_rho = np.nan
    if after.sum() > 10 and change:
        slope = np.polyfit(t_post[after], post[col].values[after], 1)[0]
        late_trend_share = slope * (fault_s - RESPONSE_LAG_S) / change
        with warnings.catch_warnings():  # constant series -> rho is NaN, which is handled downstream
            warnings.simplefilter("ignore")
            rho = pd.Series(post[col].values[after]).corr(pd.Series(t_post[after]), method="spearman")
        late_trend_rho = rho * np.sign(change)

    time_to_plateau_s = np.nan
    if plateau_ref == "late":
        plateau_z = total_z
    else:
        plateau_z = (np.median(post[col].values[after]) - base) / scale if after.any() else total_z
    if abs(plateau_z) >= 3:
        smooth = pd.Series(z_post).rolling(smooth_s, min_periods=1, center=True).median().values
        target = plateau_level * plateau_z
        hit = np.nonzero(smooth >= target if plateau_z > 0 else smooth <= target)[0]
        if len(hit):
            time_to_plateau_s = t_post[hit[0]]

    t_pre = (pre.time.values - pre.time.values[0]).astype(float)
    pre_slope = np.polyfit(t_pre, pre[col].values, 1)[0]
    pre_drift_share = pre_slope * t_pre[-1] / change if change else np.nan

    return {
        "total_z": total_z, "log2_fold": log2_fold, "early_share": early_share,
        "late_trend_share": late_trend_share, "late_trend_rho": late_trend_rho,
        "time_to_plateau_s": time_to_plateau_s, "plateau_frac": time_to_plateau_s / fault_s,
        "pre_drift_share": pre_drift_share,
        "normal_s": pre.time.iloc[-1] - pre.time.iloc[0], "fault_s": fault_s,
    }


THRESHOLDS_VERSION = "shape-v2.3"
THRESHOLDS = {  # PROVISIONAL, deliberately not tuned on inspected cases
    "min_z": 5, "min_log2_fold": 0.25, "gradual_trend_share": 0.3, "step_plateau_max_s": STEP_PLATEAU_MAX_S,
    # second route past the size check, for slow leaks still small at the end of the window
    "trend_min_z": 10, "trend_min_rho": 0.5,
}
BORDERLINE_MARGIN = 0.2  # within 20% of a threshold -> "borderline" instead of a forced label


def classify_shape(sig, th=THRESHOLDS, margin=BORDERLINE_MARGIN):
    """Size check: fold route (|z| >= min_z and |log2_fold| >= min_log2_fold) OR trend route (|z| >= trend_min_z,
    late_trend_share >= gradual_trend_share, late_trend_rho >= trend_min_rho). High z alone never passes.
    Shape: step (settles within step_plateau_max_s, |trend| < gradual_trend_share), receding (trend <= -share),
    else gradual. Returns (shape, lean, near); shape is "borderline" when a deciding threshold was within margin."""
    if sig is None:
        return "no_data", "no_data", ""
    z, fold = abs(sig["total_z"]), abs(sig["log2_fold"])
    plateau, trend, rho = sig["time_to_plateau_s"], sig["late_trend_share"], sig["late_trend_rho"]
    close = lambda v, t: abs(v - t) <= margin * t

    fold_near = [k for k, v in [("min_z", z), ("min_log2_fold", fold)] if close(v, th[k])]
    trend_near = [k for k, v in [("trend_min_z", z), ("gradual_trend_share", trend), ("trend_min_rho", rho)]
                  if close(v, th[k])]
    fold_ok = z >= th["min_z"] and fold >= th["min_log2_fold"]
    trend_ok = z >= th["trend_min_z"] and trend >= th["gradual_trend_share"] and rho >= th["trend_min_rho"]

    if fold_ok and trend_ok:
        near = []
    elif fold_ok:
        near = fold_near
    elif trend_ok:
        near = trend_near
    else:
        near = fold_near + trend_near
    if not (fold_ok or trend_ok):
        lean = "no_clear_shift"
    else:
        if close(plateau, th["step_plateau_max_s"]):
            near.append("step_plateau_max_s")
        if close(abs(trend), th["gradual_trend_share"]) and "gradual_trend_share" not in near:
            near.append("gradual_trend_share")
        if plateau <= th["step_plateau_max_s"] and abs(trend) < th["gradual_trend_share"]:
            lean = "step"
        elif trend <= -th["gradual_trend_share"]:
            lean = "receding"
        else:
            lean = "gradual"
    return ("borderline" if near else lean), lean, ",".join(near)


# ============================================================ presence / sparse rate / combined evidence
EVIDENCE_THRESHOLDS = {
    "presence_min_delta": 0.2,  # non-null fraction changes by >= 20 points
    "sparse_min_frac": 0.05,    # nonzero in >= 5% of fault seconds ...
    "sparse_min_ratio": 2.0,    # ... and the mean at least doubles
}


def presence_signature(metrics_df, col, inject_time, th=EVIDENCE_THRESHOLDS):
    pre = metrics_df.loc[metrics_df.time < inject_time, col]
    post = metrics_df.loc[metrics_df.time >= inject_time, col]
    a = pre.notna().mean() if len(pre) else np.nan
    b = post.notna().mean() if len(post) else np.nan
    change = "none"
    if b - a >= th["presence_min_delta"]:
        change = "appears"
    elif a - b >= th["presence_min_delta"]:
        change = "goes_quiet"
    return {"presence_pre": a, "presence_post": b, "presence_change": change}


def sparse_rate_signature(metrics_df, col, inject_time, th=EVIDENCE_THRESHOLDS):
    """For sparse signals like errors. NaN counts as 0 here only; presence_signature reports missing data."""
    pre = metrics_df.loc[metrics_df.time < inject_time, col].fillna(0)
    post = metrics_df.loc[metrics_df.time >= inject_time, col].fillna(0)
    f_pre, f_post = (pre != 0).mean(), (post != 0).mean()
    rises = bool(f_post >= th["sparse_min_frac"] and post.mean() > th["sparse_min_ratio"] * pre.mean() + 1e-12)
    return {"nonzero_frac_pre": f_pre, "nonzero_frac_post": f_post,
            "mean_pre": pre.mean(), "mean_post": post.mean(), "sparse_rise": rises}


def metric_evidence(metrics_df, col, inject_time, sparse=None):
    """Shape + presence + (for *_error) sparse rate. shifted = clear evidence; weak = borderline only.
    artifact_flag marks evidence that may come from the injection mechanism (diskio appearing)."""
    if sparse is None:
        sparse = col.endswith("_error")
    ev = {"metric": col, **presence_signature(metrics_df, col, inject_time)}
    sig = shape_signature_v2(metrics_df, col, inject_time)
    if sig is None:
        ev.update({"shape": "no_data", "lean": "no_data", "near_threshold": ""})
    else:
        ev.update(sig)
        ev["shape"], ev["lean"], ev["near_threshold"] = classify_shape(sig)
    if sparse:
        ev.update(sparse_rate_signature(metrics_df, col, inject_time))
    shape_hit = ev["lean"] not in ("no_clear_shift", "no_data")
    ev["shifted"] = bool((shape_hit and ev["shape"] != "borderline") or ev["presence_change"] != "none"
                         or ev.get("sparse_rise", False))
    ev["weak"] = bool(not ev["shifted"] and ev["shape"] == "borderline")
    reasons = []
    if shape_hit:
        reasons.append(("borderline:" if ev["shape"] == "borderline" else "") + ev["lean"])
    elif ev["shape"] == "borderline":
        reasons.append(f"near-miss({ev['near_threshold']})")
    if ev["presence_change"] != "none":
        reasons.append(ev["presence_change"])
    if ev.get("sparse_rise"):
        reasons.append("sparse_rate_rise")
    ev["evidence"] = ",".join(reasons)
    ev["artifact_flag"] = "diskio_appears" if col.endswith("_diskio") and ev["presence_change"] == "appears" else ""
    return ev


# ============================================================ call graph
STATIC_EDGES = {  # from the systems' published architecture diagrams, NOT derived from this data
    "ob": {("frontend", "adservice"), ("frontend", "cartservice"), ("checkoutservice", "cartservice"),
           ("frontend", "shippingservice"), ("checkoutservice", "shippingservice")},
    "ss": {("front-end", "catalogue"), ("front-end", "carts"), ("front-end", "orders"), ("front-end", "user"),
           ("orders", "user"), ("orders", "carts"), ("orders", "payment"), ("orders", "shipping")},
    "tt": set(),
}
_TRACE_DATASETS = {"ob": ["RE2-OB", "RE3-OB"], "ss": [], "tt": ["RE2-TT", "RE3-TT"]}


def trace_edges(datasets, per_dataset=10):
    """caller -> callee edges from span parent links, sampled from a few cases per dataset."""
    idx = load_index()
    edges = set()
    for ds in datasets:
        for case in idx[(idx.dataset == ds) & idx.has_traces].case.head(per_dataset):
            tr = pd.read_parquet(f"{DATA_DIR}/{case}/traces.parquet", columns=["spanID", "parentSpanID", "serviceName"])
            j = tr.merge(tr[["spanID", "serviceName"]].rename(columns={"spanID": "parentSpanID", "serviceName": "parent"}),
                         on="parentSpanID")
            edges |= set(zip(j.parent, j.serviceName))
    rename = {"frontendservice": "frontend"}  # trace name -> metric name
    return {(rename.get(a, a), rename.get(b, b)) for a, b in edges if a != b}


@lru_cache(maxsize=None)
def _trace_edges_for(system):
    return frozenset(trace_edges(_TRACE_DATASETS[system]))


def callers_of(system, svc):
    """[(caller, "trace"|"static")]. Known gaps: Sock Shop is all static; ts-auth-service has no callers."""
    out = [(a, "trace") for a, b in _trace_edges_for(system) if b == svc]
    out += [(a, "static") for a, b in STATIC_EDGES[system] if b == svc and (a, "trace") not in out]
    return sorted(out)


def service_evidence(case, system, svc, inject_time, m=None):
    """metric_evidence for every own metric of svc plus its callers' latency."""
    m = load_metrics(case) if m is None else m
    own = [c for c in m.columns if c.startswith(svc + "_") and c != "time"]
    callers = [resolve_col(m, c, "latency") for c, _ in callers_of(system, svc)]
    rows = [{"role": "own", **metric_evidence(m, c, inject_time)} for c in own]
    rows += [{"role": "caller", **metric_evidence(m, c, inject_time)} for c in callers if c]
    return pd.DataFrame(rows)


def diagnosability_status(ev):
    """From service_evidence(): 'diagnosable' | 'weak evidence only' | 'not diagnosable from available data'.
    Metrics only - for cases with logs, also check logs before excluding (explore.ipynb section 7)."""
    if len(ev) and ev.shifted.any():
        return "diagnosable"
    if len(ev) and ev.weak.any():
        return "weak evidence only"
    return "not diagnosable from available data"


# ============================================================ logs
_TEMPLATE_RULES = [
    (re.compile(r"^\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(\.\d+)?Z?\s*"), ""),  # leading timestamp
    (re.compile(r"ts=\S+"), "ts=<t>"),
    (re.compile(r"\[([\w-]+),[0-9a-f]*,[0-9a-f]*,(true|false)\]"), r"[\1,<trace>]"),  # spring sleuth ids
    (re.compile(r"\[[\w-]*exec-\d+\]"), "[<thread>]"),
    # keep HTTP status codes: "POST /cart 500" vs "POST /cart 200" is the symptom
    (re.compile(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD) (\S+) (\d{3})\b"), r"\1 \2 status\3"),
    (re.compile(r"(?<![0-9a-f])[0-9a-f]{8,}(?![0-9a-f])"), "<hex>"),
    (re.compile(r"(?<![A-Za-z\d])\d+(\.\d+)?"), "<n>"),  # the lookbehind keeps "status500" intact
]


def template(msg):
    if not isinstance(msg, str):
        return "<null message>"
    for rx, rep in _TEMPLATE_RULES:
        msg = rx.sub(rep, msg)
    return msg.strip()


_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_KV = re.compile(r"\b([A-Za-z_][\w-]*)=(\"[^\"]*\"|[^\s,;]+)")
_JSONKV = re.compile(r'\\?"([A-Za-z_]\w*)\\?":\\?"((?:[^"\\]|\\(?!"))*)\\?"')  # also escaped JSON inside strings


def kv_vocab(messages, min_count=20):
    """(key, value) pairs frequent enough to be vocabulary (method=Register) rather than data (username=...)."""
    c = Counter()
    for m in messages:
        if isinstance(m, str):
            c.update(_KV.findall(m))
            c.update(_JSONKV.findall(m))
    return {kv for kv, n in c.items() if n >= min_count}


def template_v2(msg, vocab):
    """Stronger templating for step 0: masks UUIDs, emails and rare key=value / JSON values, then template()."""
    if not isinstance(msg, str):
        return "<null message>"
    msg = _UUID.sub("<uuid>", msg)
    msg = _EMAIL.sub("<email>", msg)
    msg = _KV.sub(lambda m: m.group(0) if (m.group(1), m.group(2)) in vocab else f"{m.group(1)}=<v>", msg)
    msg = _JSONKV.sub(lambda m: m.group(0) if (m.group(1), m.group(2)) in vocab else f'"{m.group(1)}":"<v>"', msg)
    return template(msg)


def load_logs(case):
    L = pd.read_parquet(f"{DATA_DIR}/{case}/logs.parquet")
    L["rel"] = L.timestamp - load_inject_time(case)
    L["null_message"] = L.message.isna()  # kept as rows, never matched by filters
    L["tmpl"] = L.message.map(template)
    return L


def quiet_periods(L, bin_s=10, window_bins=3, frac=0.25, min_rate=0.2):
    """Per container: flag a 30 s stretch after t=0 below 25% of the median pre-t=0 rate.
    Containers under min_rate lines/s at baseline are reported as too sparse, not dropped."""
    bins = (L.rel // bin_s) * bin_s
    vol = L.groupby([L.container_name, bins]).size().unstack(fill_value=0)
    vol = vol.reindex(columns=range(int(bins.min()), int(bins.max()) + 1, bin_s), fill_value=0)
    out = []
    for c, s in vol.iterrows():
        pre_rate = s[s.index < 0].median() / bin_s
        post = s[s.index >= 0]
        roll = post.rolling(window_bins).mean() / bin_s
        low = roll[roll < frac * pre_rate]
        judged = len(low) and pre_rate >= min_rate
        out.append({"container": c, "pre_lines_per_s": round(pre_rate, 2),
                    "post_lines_per_s": round(post.median() / bin_s, 2),
                    "sparse_baseline": pre_rate < min_rate,
                    "quiet_from_s": (low.index.min() - (window_bins - 1) * bin_s) if judged else None,
                    "quiet_until_s": low.index.max() if judged else None})
    return pd.DataFrame(out), vol


def log_rate_changes(case, min_count=20, ratio=3.0):
    """Existing log patterns whose rate changes >= ratio (with >= min_count lines), plus container volume ratios."""
    L = load_logs(case)
    L["period"] = np.where(L.rel >= 0, "post", "pre")
    pre_s, post_s = -L.rel.min(), L.rel.max() + 1
    cnt = L.groupby(["container_name", "tmpl", "period"]).size().unstack(fill_value=0).reindex(columns=["pre", "post"], fill_value=0)
    rate_pre, rate_post = cnt["pre"] / pre_s, cnt["post"] / post_s
    big = (cnt[["pre", "post"]].max(axis=1) >= min_count) & ((rate_post > ratio * rate_pre) | (rate_pre > ratio * rate_post))
    out = cnt[big].assign(ratio_post_over_pre=(rate_post / rate_pre.replace(0, np.nan))[big].round(2))
    vol = L.groupby(["container_name", "period"]).size().unstack(fill_value=0).reindex(columns=["pre", "post"], fill_value=0)
    vol_ratio = ((vol["post"] / post_s) / (vol["pre"] / pre_s)).round(2)
    return out, vol_ratio


# exception / error class names: never truncated, always kept (subtle-symptom rule)
EXC_CLASS = re.compile(r"\b(?:[a-z_][\w$]*\.)*[A-Z][\w$]*(?:Exception|Error)\b")


def exception_classes(text):
    return list(dict.fromkeys(EXC_CLASS.findall(text if isinstance(text, str) else "")))  # null messages -> none


# ============================================================ step 0: compressed, ground-truth-blind evidence
STEP0_VERSION = "step0-v0.2"
STEP0_PARAMS = {
    "rate_ratio": 3.0,       # clear log rate change
    "weak_rate_ratio": 2.0,  # 2-3x -> weak rate change
    "min_lines": 20,         # rate changes / vanished need >= this many lines on the larger side
    "new_min": 3,            # "new" pattern needs >= 3 lines after t=0; fewer -> one-off
    "rare_max": 20,          # rare recurring: seen before AND after, < 20 lines on each side ...
    "rare_event_max": 3,     # ... of which <= 3 lines in total are "rare events" (e.g. a restart banner) -> examples shown
}


def step0_metrics(case, m=None):
    """Every metric of every service with clear or weak evidence. Returns (table, all services)."""
    m = load_metrics(case) if m is None else m
    t = load_inject_time(case)
    rows = []
    for col in [c for c in m.columns if c != "time"]:
        ev = metric_evidence(m, col, t)
        if not (ev["shifted"] or ev["weak"]):
            continue
        svc, kind = col.rsplit("_", 1)
        fold = ev.get("log2_fold", np.nan)
        rows.append({
            "service": svc, "metric": kind, "evidence": ev["evidence"],
            "dir": "" if pd.isna(fold) else ("up" if fold > 0 else "down"),
            "fold": None if pd.isna(fold) else round(2 ** abs(fold), 2),
            "reached_s": ev.get("time_to_plateau_s"),
            "presence": f'{ev["presence_pre"]:.0%}->{ev["presence_post"]:.0%}' if ev["presence_change"] != "none" else "",
            "errors_nonzero": f'{ev["nonzero_frac_pre"]:.0%}->{ev["nonzero_frac_post"]:.0%}' if ev.get("sparse_rise") else "",
            "clear": bool(ev["shifted"]),
            "artifact_flag": ev["artifact_flag"],
        })
    services = sorted({c.rsplit("_", 1)[0] for c in m.columns if c != "time"})
    return pd.DataFrame(rows), services


def _sample_across_span(grp, k):
    """k rows spread across the time span (earliest and latest always included)."""
    grp = grp.sort_values("first_after_s").reset_index(drop=True)
    if len(grp) <= k:
        return grp
    picked = []
    for target in np.linspace(grp.first_after_s.min(), grp.first_after_s.max(), k):  # evenly spaced in time
        order = (grp.first_after_s - target).abs().sort_values(kind="stable").index
        picked.append(next(i for i in order if i not in picked))
    return grp.loc[sorted(picked)]


def step0_logs(case, p=STEP0_PARAMS):
    L = pd.read_parquet(f"{DATA_DIR}/{case}/logs.parquet")
    L["rel"] = L.timestamp - load_inject_time(case)
    vocab = kv_vocab(L.message)
    L["tmpl"] = L.message.map(lambda s: template_v2(s, vocab))
    L["period"] = np.where(L.rel >= 0, "post", "pre")
    pre_s, post_s = -L.rel.min(), L.rel.max() + 1

    cnt = L.groupby(["container_name", "tmpl", "period"]).size().unstack(fill_value=0).reindex(columns=["pre", "post"], fill_value=0)
    first_post = L[L.period == "post"].groupby(["container_name", "tmpl"]).rel.min()
    last_post = L[L.period == "post"].groupby(["container_name", "tmpl"]).rel.max()
    # one full example message per template, for exception class names that templating/truncation could hide
    example = L.groupby(["container_name", "tmpl"]).message.first()
    rpre, rpost = cnt.pre / pre_s, cnt.post / post_s
    big = cnt[["pre", "post"]].max(axis=1) >= p["min_lines"]
    both = (cnt.pre > 0) & (cnt.post > 0)
    ratio_up, ratio_down = rpost / rpre.replace(0, np.nan), rpre / rpost.replace(0, np.nan)
    ratio = np.fmax(ratio_up, ratio_down)

    kind = np.select(
        [(cnt.pre == 0) & (cnt.post >= p["new_min"]),
         (cnt.post == 0) & (cnt.pre >= p["min_lines"]),
         big & both & (ratio >= p["rate_ratio"]),
         big & both & (ratio >= p["weak_rate_ratio"])],
        ["new", "vanished", "rate_change", "rate_change_weak"], "")
    pats = cnt.assign(kind=kind, first_after_s=first_post, example=example).reset_index()
    pats = pats[pats.kind != ""].copy()
    pats = (pats.sort_values("tmpl").groupby(["container_name", "kind", "pre", "post", "first_after_s"], dropna=False)
            .agg(tmpl=("tmpl", "first"), example=("example", "first"), n_similar=("tmpl", "size")).reset_index())

    def summarise(mask):
        d = cnt[mask].reset_index().merge(first_post.rename("first_after_s").reset_index(), on=["container_name", "tmpl"])
        d = d.merge(last_post.rename("last_after_s").reset_index(), on=["container_name", "tmpl"])
        d = d.merge(example.rename("example").reset_index(), on=["container_name", "tmpl"])
        return {c: g for c, g in d.groupby("container_name")}

    oneoffs = summarise((cnt.pre == 0) & (cnt.post > 0) & (cnt.post < p["new_min"]))
    # rare recurring: seen before AND after t=0 but rarely. Rare events (<= rare_event_max lines in total, e.g. a
    # restart banner repeating an earlier restart) get examples; other infrequent patterns are only counted.
    rare_mask = both & (cnt[["pre", "post"]].max(axis=1) < p["rare_max"])
    rare = summarise(rare_mask & ((cnt.pre + cnt.post) <= p["rare_event_max"]))
    rare_other = cnt[rare_mask & ((cnt.pre + cnt.post) > p["rare_event_max"])].groupby("container_name").size().to_dict()

    qp, _ = quiet_periods(L)
    vol = L.groupby(["container_name", "period"]).size().unstack(fill_value=0).reindex(columns=["pre", "post"], fill_value=0)
    vol = pd.DataFrame({"per_min_pre": (vol.pre / pre_s * 60).round(1), "per_min_post": (vol.post / post_s * 60).round(1)})
    vol = vol.join(qp.set_index("container")[["quiet_from_s", "quiet_until_s", "sparse_baseline"]])
    return pats, oneoffs, rare, rare_other, vol, int(L.message.isna().sum())


_PLAIN = {"step": "step change", "gradual": "still moving", "receding": "jumped then partly recovered",
          "appears": "metric appeared", "goes_quiet": "metric went quiet", "sparse_rate_rise": "more seconds with errors"}


def _plain_evidence(ev):
    out = []
    for part in re.split(r",(?![^(]*\))", ev):  # don't split inside near-miss(a,b)
        if part.startswith("near-miss") or part.startswith("borderline:no_clear"):
            out.append("small change near threshold")
        else:
            out.append(_PLAIN.get(part.replace("borderline:", ""), part))
    return ", ".join(dict.fromkeys(out))


def _fmt_size(fold):
    if fold is None or pd.isna(fold):
        return ""
    return "from ~0" if fold > 1000 else f"{fold:g}"


_SPRING_PREFIX = re.compile(r"^(TRACE|DEBUG|INFO|WARN|ERROR)\s+\[[^\]]*\]\s+<n>\s+---\s+\[[^\]]*\]\s+(\S+)\s*:\s*")


def _compact(s):
    """Display only: collapse whitespace and shorten the Spring prefix
    'WARN [svc,<trace>] <n> --- [<thread>] o.s.web.servlet.PageNotFound   : msg' -> 'WARN o.s.web.servlet.PageNotFound: msg'."""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return _SPRING_PREFIX.sub(r"\1 \2: ", s)


def _cut(s, n, full=None):
    """Truncate to n chars, but never lose an exception/error class name from the full line."""
    s = _compact(s)
    if len(s) <= n:
        cut = s
    else:
        cut = s[: n - 3] + "..."
    missing = [c for c in exception_classes(full if full is not None else s) if c not in cut]
    return cut + (f" [classes: {', '.join(missing)}]" if missing else "")


def render_step0(case, include_artifacts=True, max_pat_rows=None, examples_per_group=5, tmpl_chars=150):
    """Ground-truth-blind evidence text. max_pat_rows=None -> unranked, nothing cut (size set by num_ctx).
    With a cap, pattern rows are taken one per service in turn (clear kinds first) and the rest is
    announced on an OMITTED line - never dropped silently."""
    r = case_info(case)
    met, services = step0_metrics(case)
    omitted = []
    if not include_artifacts and len(met):
        n_art = int((met.artifact_flag != "").sum())
        met = met[met.artifact_flag == ""]
        if n_art:
            omitted.append(f"{n_art} metric rows flagged as possible injection artifacts")
    out = [f"SYSTEM: {r['system_name']}. Services: {', '.join(services)}.",
           f"A fault started at t=0. Data covers {int(r['normal_timesteps'])}s before and {int(r['faulty_timesteps'])}s after t=0.",
           "All tables list services in alphabetical order; nothing is ranked by importance.", "",
           "== METRICS that changed after t=0 (clear = strong evidence, weak = near a detection threshold) ==",
           "service | metric | change | dir | size(x) | settled_at_s | presence | error_seconds | strength"]
    clear = met[met.clear] if len(met) else met
    for x in clear.sort_values(["service", "metric"]).itertuples() if len(clear) else []:
        out.append(" | ".join([x.service, x.metric,
                               _plain_evidence(x.evidence) + (" [possible injection artifact]" if x.artifact_flag else ""),
                               x.dir, _fmt_size(x.fold),
                               "" if x.reached_s is None or pd.isna(x.reached_s) else f"{int(x.reached_s)}",
                               x.presence, x.errors_nonzero, "clear"]))
    if len(met) and (~met.clear).any():
        parts = [svc + ": " + ", ".join(f"{x.metric} {x.dir} x{_fmt_size(x.fold)}" for x in grp.itertuples())
                 for svc, grp in met[~met.clear].sort_values(["service", "metric"]).groupby("service")]
        out.append("Weak changes (near a detection threshold): " + "; ".join(parts))
    quiet_svcs = sorted(set(services) - set(met.service if len(met) else []))
    out.append(f"No metric change: {', '.join(quiet_svcs) if quiet_svcs else '(none)'}")

    if bool(r["has_logs"]):
        pats, oneoffs, rare, rare_other, vol, null_msgs = step0_logs(case)
        out += ["", "== LOG VOLUME per service (lines/min before -> after; quiet = <25% of normal rate for 30s+) =="]
        for c, v in vol.sort_index().iterrows():
            q = ""
            if not pd.isna(v.quiet_from_s):
                q = f" | quiet from {int(v.quiet_from_s)}s to {int(v.quiet_until_s)}s"
            elif v.sparse_baseline:
                q = " | too few lines to judge quiet periods"
            out.append(f"{c} | {v.per_min_pre:g} -> {v.per_min_post:g}{q}")
        out += ["", "== LOG PATTERNS after t=0: new, vanished, rate change >=3x (clear) or 2-3x (weak). All log levels, no keyword filter ==",
                "service | kind | count before -> after | first seen after t=0 (s) | pattern"]
        order = {"new": 0, "vanished": 1, "rate_change": 2, "rate_change_weak": 3}
        pats = pats.assign(_o=pats.kind.map(order)).sort_values(["container_name", "_o", "post"], ascending=[True, True, False])
        shown = pats
        if max_pat_rows is not None and len(pats) > max_pat_rows:
            shown = (pats.assign(_r=pats.groupby("container_name").cumcount())
                     .sort_values(["_r", "_o", "container_name"]).head(max_pat_rows)
                     .sort_values(["container_name", "_o", "post"], ascending=[True, True, False]))
            omitted.append(f"{len(pats) - max_pat_rows} log pattern rows beyond the {max_pat_rows}-row budget")
        label = {"rate_change_weak": "rate change (weak)", "rate_change": "rate change"}
        for x in shown.itertuples():
            fa = "-" if pd.isna(x.first_after_s) or x.post < x.pre else f"{int(x.first_after_s)}"
            dup = f" (+{x.n_similar - 1} similar lines with the same counts)" if x.n_similar > 1 else ""
            out.append(f"{x.container_name} | {label.get(x.kind, x.kind)} | {x.pre} -> {x.post} | {fa} | "
                       f"{_cut(x.tmpl, tmpl_chars, x.example)}{dup}")

        def group_lines(groups, what):
            for c, grp in sorted(groups.items()):
                span = f"{int(grp.first_after_s.min())}s-{int(grp.last_after_s.max())}s"
                ex = "; ".join(f"[{int(e.first_after_s)}s] {_cut(e.tmpl, 90, e.example)}"
                               for e in _sample_across_span(grp, examples_per_group).itertuples())
                classes = exception_classes(" ".join(str(m) for m in grp.example))
                cls = f" | exception/error classes: {', '.join(classes)}" if classes else ""
                out.append(f"{c} | {len(grp)} {what} between {span}{cls} | examples: {ex}")

        group_lines(oneoffs, "one-off new lines (1-2 times, none before t=0)")
        group_lines(rare, "rare events seen both before and after t=0 (<=3 lines in total)")
        if rare_other:
            out.append("Other infrequent patterns seen before and after t=0 (not shown): "
                       + ", ".join(f"{c} {n}" for c, n in sorted(rare_other.items())))
        if null_msgs:
            out.append(f"({null_msgs} log lines have no message text)")
    else:
        out += ["", "== LOGS: none for this case =="]
    if not bool(r["has_traces"]):
        out.append("== TRACES: none for this case ==")
    if omitted:
        out += ["", "OMITTED for size: " + "; ".join(omitted) + "."]
    return "\n".join(out)


# ============================================================ scoring and Ollama context sizing
def normalize_service(s):
    return str(s).strip().lower()


def is_correct(answer, truth):
    """Exact match, case-insensitive, trimmed. Never substring ('carts' must not match 'carts-db')."""
    return normalize_service(answer) == normalize_service(truth)


@lru_cache(maxsize=None)
def _tokenizer(model):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    repo = {"qwen": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "gemma": "unsloth/gemma-3-27b-it"}[model]  # gemma: Gemma 3 tokenizer, approximation for gemma4
    return Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))


def count_tokens(text, model="qwen"):
    """Exact for qwen2.5-coder; approximate for gemma4 (Gemma 3 tokenizer). Falls back to chars/3.5."""
    try:
        return len(_tokenizer(model).encode(text).ids)
    except Exception:
        return round(len(text) / 3.5)


def num_ctx_for(prompt_tokens, answer_reserve=1024, margin=0.15, round_to=512):
    """Per-case Ollama num_ctx: prompt + answer reserve + margin, rounded up, rather than a flat value.
    prompt_tokens must be the FULL prompt (instructions + evidence), not just the evidence."""
    need = prompt_tokens * (1 + margin) + answer_reserve
    return int(np.ceil(need / round_to) * round_to)


def check_prompt_eval(expected_tokens, prompt_eval_count, tolerance=0.1):
    """Ollama truncates silently when the prompt exceeds num_ctx. Returns (ok, message)."""
    if prompt_eval_count is None:
        return False, "no prompt_eval_count in response"
    if prompt_eval_count < expected_tokens * (1 - tolerance):
        return False, f"possible truncation: prompt_eval_count={prompt_eval_count} < expected ~{expected_tokens}"
    return True, "ok"


# ============================================================ CLI (used by the inspect-case skill)
def inspect_case(case, include_artifacts=True, max_pat_rows=None):
    """Human-facing report for one case: labels, exclusions, evidence on the true root cause, and the
    ground-truth-blind step-0 text exactly as a model would get it."""
    idx = load_index()
    if case not in set(idx.case):
        raise SystemExit(f"unknown case: {case}")
    r = case_info(case)
    lines = [f"# {case}",
             f"dataset {r['dataset']} | fault {r['fault']} ({r['fault_description']}) | ground truth: {r['root_cause_service']}",
             f"logs: {bool(r['has_logs'])} | traces: {bool(r['has_traces'])} | root_cause.txt: {bool(r['has_root_cause_file'])}"]
    problems = []
    if case in EXCLUDE:
        problems.append(f"EXCLUDED (broken data): {EXCLUDE[case]}")
    if problems:  # broken data: nothing to analyse
        return "\n".join(lines + ["", "## Problems"] + problems)
    if case in BROKEN_LABELS:  # data is fine, only the label evidence is broken
        problems.append(f"EXCLUDED from scoring (broken label): {BROKEN_LABELS[case]}")

    t = load_inject_time(case)
    ev = service_evidence(case, r["system"], r["root_cause_service"], t)
    status = diagnosability_status(ev)
    if status != "diagnosable":
        problems.append(f"metrics of the root cause + callers: {status}"
                        + (" (logs/traces exist - check them before excluding)" if r["has_logs"] or r["has_traces"] else ""))
    if (ev.artifact_flag != "").any():
        problems.append("root-cause evidence includes diskio_appears (possible injection artifact) - score with and without")
    lines += ["", "## Problems / decisions", *(problems or ["none"])]
    lines += ["", f"## Evidence on the true root cause ({r['root_cause_service']}) and its callers - NOT shown to the model"]
    shown = ev[(ev.shifted) | (ev.weak)]
    for x in shown.itertuples():
        lines.append(f"{x.role:6s} {x.metric}: {x.evidence} ({'clear' if x.shifted else 'weak'}"
                     f"{', ' + x.artifact_flag if x.artifact_flag else ''})")
    if not len(shown):
        lines.append("(no clear or weak evidence)")
    if r["has_root_cause_file"]:
        rc = read_root_cause(case)
        lines += ["", f"## root_cause.txt (at {rc['ts'] - t:+d}s): [{rc['container']}] {rc['message'][:200]}"]

    txt = render_step0(case, include_artifacts=include_artifacts, max_pat_rows=max_pat_rows)
    q, g = count_tokens(txt, "qwen"), count_tokens(txt, "gemma")
    leaks = [s for s in [case, r["fault"] + "_", "root_cause", "ground truth"] if s in txt]
    lines += ["", f"## Step 0 text ({STEP0_VERSION}): qwen {q} tokens (exact), gemma4 ~{g} (approx.) -> "
                  f"num_ctx >= {num_ctx_for(max(q, g))} (evidence only; recompute with the full prompt); "
                  f"leak check: {leaks or 'none'}", "", txt]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("inspect", help="report one case (labels, problems, blind step-0 text)")
    ins.add_argument("case")
    ins.add_argument("--no-artifacts", action="store_true", help="drop artifact-flagged evidence from the step-0 text")
    ins.add_argument("--max-pat-rows", type=int, default=None, help="cap log pattern rows (default: unranked, no cap)")
    a = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(inspect_case(a.case, include_artifacts=not a.no_artifacts, max_pat_rows=a.max_pat_rows))
