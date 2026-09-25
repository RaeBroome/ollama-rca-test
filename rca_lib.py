"""Settled, reusable functions for the Ollama RCA evaluation on RCAEval.

Exploration lives in explore.ipynb; anything here has been reviewed and is used from there.
RCAEval-data/ is read-only. Nothing in this module writes files.

CLI:  python rca_lib.py inspect <case> [--no-artifacts] [--max-pat-rows N]
"""
import csv
import json
import os
import re
import time
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


# Keys whose VALUE is always kept, however rare. kv_vocab only keeps pairs seen >= min_count times, which on a
# sparse container (carts-db: 81 lines) keeps nothing - and masking these destroyed signals our own step-2 rules
# needed ("msg":"connection accepted" -> "msg":"<v>", so the connection-churn rule could never fire).
ALWAYS_KEEP_KEYS = {"msg", "error", "exception", "result", "method", "reason", "err", "level", "severity",
                    "c", "s", "ctx", "status", "statuscode", "code", "caller", "op", "event"}
# 3-digit status codes inside JSON/key=value are kept too (the "METHOD /path 500" rule only covers access logs)
_STATUS_KV = re.compile(r'''(?i)(status|statuscode|code)("?\s*[:=]\s*"?)(\d{3})''')

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
    msg = _STATUS_KV.sub(lambda m: f"{m.group(1)}{m.group(2)}status{m.group(3)}", msg)
    keep = lambda k, v: (k, v) in vocab or k.lower() in ALWAYS_KEEP_KEYS
    msg = _KV.sub(lambda m: m.group(0) if keep(m.group(1), m.group(2)) else f"{m.group(1)}=<v>", msg)
    msg = _JSONKV.sub(lambda m: m.group(0) if keep(m.group(1), m.group(2)) else f'"{m.group(1)}":"<v>"', msg)
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
            "z": abs(ev.get("total_z", np.nan)),  # significance, used for ranking; not shown in the step-0 text
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


def service_order(case, seed=None):
    """Order services appear in. Default alphabetical; with a seed, a per-case shuffle derived from
    (seed, case), so a run is reproducible and the order is recorded. Used for the position-bias test:
    alphabetical order can put the root cause first (Sock Shop "carts") or last ("user") by accident."""
    services = sorted({c.rsplit("_", 1)[0] for c in load_metrics(case).columns if c != "time"})
    if seed is None:
        return services
    import random
    rng = random.Random(f"{seed}:{case}")
    shuffled = list(services)
    rng.shuffle(shuffled)
    return shuffled


def render_step0(case, include_artifacts=True, max_pat_rows=None, examples_per_group=5, tmpl_chars=150,
                 order_seed=None):
    """Ground-truth-blind evidence text. max_pat_rows=None -> unranked, nothing cut (size set by num_ctx).
    With a cap, pattern rows are taken one per service in turn (clear kinds first) and the rest is
    announced on an OMITTED line - never dropped silently.
    order_seed shuffles the service order for the position-bias test (see service_order)."""
    r = case_info(case)
    met, services = step0_metrics(case)
    services = service_order(case, order_seed)
    rank = {svc: i for i, svc in enumerate(services)}
    key = lambda col: col.map(lambda v: rank.get(v, len(rank)))
    omitted = []
    if not include_artifacts and len(met):
        n_art = int((met.artifact_flag != "").sum())
        met = met[met.artifact_flag == ""]
        if n_art:
            omitted.append(f"{n_art} metric rows flagged as possible injection artifacts")
    out = [f"SYSTEM: {r['system_name']}. Services: {', '.join(services)}.",
           f"A fault started at t=0. Data covers {int(r['normal_timesteps'])}s before and {int(r['faulty_timesteps'])}s after t=0.",
           ("All tables list services in an arbitrary order; nothing is ranked by importance." if order_seed is not None
            else "All tables list services in alphabetical order; nothing is ranked by importance."), "",
           "== METRICS that changed after t=0 (clear = strong evidence, weak = near a detection threshold) ==",
           "service | metric | change | dir | size(x) | settled_at_s | presence | error_seconds | strength"]
    clear = met[met.clear] if len(met) else met
    for x in (clear.sort_values(["service", "metric"], key=key) if len(clear) else clear).itertuples():
        out.append(" | ".join([x.service, x.metric,
                               _plain_evidence(x.evidence) + (" [possible injection artifact]" if x.artifact_flag else ""),
                               x.dir, _fmt_size(x.fold),
                               "" if x.reached_s is None or pd.isna(x.reached_s) else f"{int(x.reached_s)}",
                               x.presence, x.errors_nonzero, "clear"]))
    if len(met) and (~met.clear).any():
        parts = [svc + ": " + ", ".join(f"{x.metric} {x.dir} x{_fmt_size(x.fold)}" for x in grp.itertuples())
                 for svc, grp in met[~met.clear].sort_values(["service", "metric"], key=key).groupby("service", sort=False)]
        out.append("Weak changes (near a detection threshold): " + "; ".join(parts))
    quiet_svcs = [x for x in services if x not in set(met.service if len(met) else [])]
    out.append(f"No metric change: {', '.join(quiet_svcs) if quiet_svcs else '(none)'}")

    if bool(r["has_logs"]):
        pats, oneoffs, rare, rare_other, vol, null_msgs = step0_logs(case)
        out += ["", "== LOG VOLUME per service (lines/min before -> after; quiet = <25% of normal rate for 30s+) =="]
        for c, v in vol.reindex([x for x in services if x in vol.index]).iterrows():
            q = ""
            if not pd.isna(v.quiet_from_s):
                q = f" | quiet from {int(v.quiet_from_s)}s to {int(v.quiet_until_s)}s"
            elif v.sparse_baseline:
                q = " | too few lines to judge quiet periods"
            out.append(f"{c} | {v.per_min_pre:g} -> {v.per_min_post:g}{q}")
        out += ["", "== LOG PATTERNS after t=0: new, vanished, rate change >=3x (clear) or 2-3x (weak). All log levels, no keyword filter ==",
                "service | kind | count before -> after | first seen after t=0 (s) | pattern"]
        order = {"new": 0, "vanished": 1, "rate_change": 2, "rate_change_weak": 3}
        pats = pats.assign(_o=pats.kind.map(order), _svc=pats.container_name.map(lambda v: rank.get(v, len(rank))))
        pats = pats.sort_values(["_svc", "_o", "post"], ascending=[True, True, False])
        shown = pats
        if max_pat_rows is not None and len(pats) > max_pat_rows:
            shown = (pats.assign(_r=pats.groupby("container_name").cumcount())
                     .sort_values(["_r", "_o", "_svc"]).head(max_pat_rows)
                     .sort_values(["_svc", "_o", "post"], ascending=[True, True, False]))
            omitted.append(f"{len(pats) - max_pat_rows} log pattern rows beyond the {max_pat_rows}-row budget")
        label = {"rate_change_weak": "rate change (weak)", "rate_change": "rate change"}
        for x in shown.itertuples():
            fa = "-" if pd.isna(x.first_after_s) or x.post < x.pre else f"{int(x.first_after_s)}"
            dup = f" (+{x.n_similar - 1} similar lines with the same counts)" if x.n_similar > 1 else ""
            out.append(f"{x.container_name} | {label.get(x.kind, x.kind)} | {x.pre} -> {x.post} | {fa} | "
                       f"{_cut(x.tmpl, tmpl_chars, x.example)}{dup}")

        def group_lines(groups, what):
            for c, grp in sorted(groups.items(), key=lambda kv: rank.get(kv[0], len(rank))):
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
                       + ", ".join(f"{c} {n}" for c, n in sorted(rare_other.items(), key=lambda kv: rank.get(kv[0], len(rank)))))
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


# Answer reserve per model. Thinking models spend most of their output on reasoning that never reaches
# message.content: gemma4:26b returned an empty answer with num_predict 80-100. PROVISIONAL - the thinking
# reserve was set from a trivial prompt (195 tokens) and should be re-measured on real step-0 prompts.
ANSWER_RESERVE = {"default": 1024, "qwen2.5-coder:7b": 1024, "gemma4:26b": 1024}
# Measured on step-1 prompts: gemma4:26b spent ~6000 thinking chars (~2000 tokens) before answering, so 2048 was
# not enough - it hit the limit with empty content. 4096 leaves room for thinking plus the answer.
THINKING_EXTRA = 4096


def answer_reserve_for(model=None, thinking=False):
    return ANSWER_RESERVE.get(model, ANSWER_RESERVE["default"]) + (THINKING_EXTRA if thinking else 0)


# KNOWN COST, deliberately not optimised: Ollama rebuilds the runner whenever num_ctx changes, which costs a
# full model reload - measured at ~18 s for gemma4:26b and ~6 s for qwen2.5-coder:7b, against 0.2 s for the
# same call at an unchanged num_ctx. Sizing the context per case means most calls pay it, and that reload is
# most of a long run's wall time. Coarse buckets (4096) were tried and reverted: they saved ~26% but changed
# one answer out of 24 (qwen, re3ss_carts_f4_1), and a runtime setting that alters outputs is not worth the
# time saved. Keep the rounding fine; treat the reload as the price of per-case sizing.
def num_ctx_for(prompt_tokens, model=None, thinking=False, answer_reserve=None, margin=0.15, round_to=512):
    """Per-case Ollama num_ctx: prompt + answer reserve + margin, rounded up, rather than a flat value.
    prompt_tokens must be the FULL prompt (instructions + evidence), not just the evidence.
    The reserve is per model and larger when thinking is on (see ANSWER_RESERVE)."""
    reserve = answer_reserve_for(model, thinking) if answer_reserve is None else answer_reserve
    need = prompt_tokens * (1 + margin) + reserve
    return int(np.ceil(need / round_to) * round_to)


def gpu_memory_mb():
    """(free, used, total) MiB, or None when no supported GPU tool is present (CPU-only, Apple silicon, or a
    vendor we do not read). Sample this right before every timed model run and record it with the result:
    /api/ps cannot be trusted for the GPU/CPU split (see ollama_loaded), so a spill is only visible afterwards
    from free memory plus throughput. Missing tools are normal, not an error."""
    return (_gpu_nvidia() or _gpu_amd())


def _gpu_nvidia():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        return tuple(int(x) for x in out.strip().splitlines()[0].split(","))
    except Exception:
        return None


def _gpu_amd():
    """Best effort for AMD via rocm-smi. Untested here (no AMD hardware available); a failure just yields
    None, which callers already handle."""
    import subprocess
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=30).stdout
        data = json.loads(out)
        card = next(iter(data.values()))
        total = used = None
        for k, v in card.items():
            kl = k.lower()
            if "total" in kl and "vram" in kl:
                total = int(v)
            elif "used" in kl and "vram" in kl:
                used = int(v)
        if total is None:
            return None
        used = used or 0
        mib = 1024 * 1024
        return (int((total - used) / mib), int(used / mib), int(total / mib))
    except Exception:
        return None


def check_prompt_eval(expected_tokens, prompt_eval_count, tolerance=0.1):
    """Ollama truncates silently when the prompt exceeds num_ctx. Returns (ok, message)."""
    if prompt_eval_count is None:
        return False, "no prompt_eval_count in response"
    if prompt_eval_count < expected_tokens * (1 - tolerance):
        return False, f"possible truncation: prompt_eval_count={prompt_eval_count} < expected ~{expected_tokens}"
    return True, "ok"



# ============================================================ step 1: anomaly / symptom detection
# One output shape for every version, so configurations are interchangeable:
#   symptoms  : case, source, service, signal, kind, strength, onset_s, size, size_num, artifact_flag, reason
#   candidates: case, source, rank, service, reason, n_signals, first_onset_s, has_clear, artifact_only
#   meta      : dict (versions, config, model, thinking, tokens, timings, free GPU, failures)
# kinds: shape | presence | error_rate | log_new | log_vanished | log_rate | quiet | log_oneoff | log_rare
STEP1_VERSION = "step1-v0.1"
# Service order in the step-0 text is shuffled by default for scoring: alphabetical order is not neutral
# (Sock Shop "carts" sorts first and is often the truth; "user" sorts last). The seed is recorded per run.
DEFAULT_ORDER_SEED = 1234
SYMPTOM_COLS = ["case", "source", "service", "signal", "kind", "strength", "onset_s", "size", "size_num",
                "artifact_flag", "reason"]
CANDIDATE_COLS = ["case", "source", "rank", "service", "reason", "n_signals", "first_onset_s", "has_clear",
                  "artifact_only"]
SIZE_CAP = 1000.0  # cap, so one unbounded value can't dominate a ranking


def _ratio(pre, post):
    if pre and post:
        return max(post / pre, pre / post)
    return SIZE_CAP if post else 1.0


def poisson_z(pre_count, post_count, pre_s, post_s):
    """Significance of a count change: how far the observed count is from what the normal-period rate predicts,
    in sqrt(expected) units. The log-side equivalent of a metric's total_z. Ranking only."""
    expected = (pre_count or 0) * (post_s / pre_s if pre_s else 1)
    return min(abs((post_count or 0) - expected) / max(expected, 1.0) ** 0.5, SIZE_CAP)


def step1_symptoms(case, source="python", include_artifacts=True):
    """Normalise the step-0 evidence into symptom rows. Same evidence as render_step0, no ranking."""
    rows = []
    met, _ = step0_metrics(case)
    for x in met.itertuples():
        if not include_artifacts and x.artifact_flag:
            continue
        ev = x.evidence
        kind = "error_rate" if "sparse_rate_rise" in ev else ("presence" if x.presence else "shape")
        fold = float(x.fold) if x.fold is not None and not pd.isna(x.fold) else 1.0
        z = float(x.z) if not pd.isna(x.z) else 0.0  # rank by significance: a ~0 baseline makes fold unbounded
        rows.append({"service": x.service, "signal": f"{x.service}_{x.metric}", "kind": kind,
                     "strength": "clear" if x.clear else "weak",
                     "onset_s": None if x.reached_s is None or pd.isna(x.reached_s) else float(x.reached_s),
                     "size": _fmt_size(x.fold) + (f" ({x.presence})" if x.presence else "") + (f" errors {x.errors_nonzero}" if x.errors_nonzero else ""),
                     "size_num": min(z, SIZE_CAP), "artifact_flag": x.artifact_flag,
                     "reason": _plain_evidence(ev)})

    if bool(case_info(case)["has_logs"]):
        pats, oneoffs, rare, rare_other, vol, _ = step0_logs(case)
        info = case_info(case)
        pre_s, post_s = float(info["normal_timesteps"]), float(info["faulty_timesteps"])
        kind_map = {"new": "log_new", "vanished": "log_vanished", "rate_change": "log_rate", "rate_change_weak": "log_rate"}
        for x in pats.itertuples():
            rows.append({"service": x.container_name, "signal": _cut(x.tmpl, 120, x.example), "kind": kind_map[x.kind],
                         "strength": "weak" if x.kind == "rate_change_weak" else "clear",
                         "onset_s": None if pd.isna(x.first_after_s) or x.post < x.pre else float(x.first_after_s),
                         "size": f"{x.pre}->{x.post}", "size_num": poisson_z(x.pre, x.post, pre_s, post_s),
                         "artifact_flag": "", "reason": f"log pattern {x.kind}"})
        for c, v in vol.iterrows():
            if not pd.isna(v.quiet_from_s):
                rows.append({"service": c, "signal": f"{c} log volume", "kind": "quiet", "strength": "clear",
                             "onset_s": float(v.quiet_from_s),
                             "size": f"{v.per_min_pre:g}->{v.per_min_post:g} lines/min",
                             "size_num": poisson_z(v.per_min_pre * pre_s / 60, v.per_min_post * post_s / 60, pre_s, post_s),
                             "artifact_flag": "",
                             "reason": f"log volume quiet {int(v.quiet_from_s)}s-{int(v.quiet_until_s)}s"})
        for groups, kind in [(oneoffs, "log_oneoff"), (rare, "log_rare")]:
            for c, grp in groups.items():
                classes = exception_classes(" ".join(str(m) for m in grp.example))
                rows.append({"service": c, "signal": f"{len(grp)} {kind} lines" + (f" [classes: {', '.join(classes)}]" if classes else ""),
                             "kind": kind, "strength": "clear" if classes else "weak",
                             "onset_s": float(grp.first_after_s.min()), "size": f"{len(grp)} patterns",
                             "size_num": poisson_z(0, len(grp), pre_s, post_s), "artifact_flag": "",
                             "reason": f"{kind} between {int(grp.first_after_s.min())}s-{int(grp.last_after_s.max())}s"})
    df = pd.DataFrame(rows, columns=[c for c in SYMPTOM_COLS if c not in ("case", "source")])
    df.insert(0, "source", source)
    df.insert(0, "case", case)
    return df


def rank_candidates(symptoms, order="strength"):
    """order: strength (clear first, then size, one row per service before seconds) | onset (earliest first) | none."""
    s = symptoms.copy()
    if order == "strength":
        s["_clear"] = (s.strength == "clear").astype(int)
        s = s.sort_values(["_clear", "size_num"], ascending=[False, False])
        s["_r"] = s.groupby("service").cumcount()
        s = s.sort_values(["_r", "_clear", "size_num"], ascending=[True, False, False]).drop(columns=["_r", "_clear"])
    elif order == "onset":
        s = s.sort_values("onset_s", na_position="last", kind="stable")
    cands = []
    for svc in dict.fromkeys(s.service):
        g = s[s.service == svc]
        top = g.head(2)
        cands.append({"service": svc, "reason": "; ".join(f"{x.kind} {x.signal[:60]} ({x.size})" for x in top.itertuples()),
                      "n_signals": len(g), "first_onset_s": g.onset_s.min(),
                      "has_clear": bool((g.strength == "clear").any()),
                      "artifact_only": bool((g.artifact_flag != "").all())})
    out = pd.DataFrame(cands)
    out.insert(0, "rank", range(1, len(out) + 1))
    out.insert(0, "source", symptoms.source.iloc[0] if len(symptoms) else "")
    out.insert(0, "case", symptoms.case.iloc[0] if len(symptoms) else "")
    return out[CANDIDATE_COLS], s.reset_index(drop=True)


def step1_python(case, order="strength", include_artifacts=True):
    t0 = time.time()
    sym = step1_symptoms(case, source=f"python:{order}", include_artifacts=include_artifacts)
    cands, sym = rank_candidates(sym, order=order)
    return {"candidates": cands, "symptoms": sym,
            "meta": {"case": case, "source": f"python:{order}", "step1_version": STEP1_VERSION,
                     "step0_version": STEP0_VERSION, "thresholds_version": THRESHOLDS_VERSION,
                     "include_artifacts": include_artifacts, "wall_s": round(time.time() - t0, 2)}}


STEP1_INSTRUCTIONS = """You are analysing telemetry from a microservice system. A fault started at t=0.

Below is a summary of everything that changed after t=0, for every service. Nothing in it is ranked by importance.

Your task is SYMPTOM DETECTION, not a final verdict: list the changes that look like real symptoms, and rank the
services that deserve investigation. A service showing a symptom may be a victim of another service's fault.

Answer with JSON only, no other text, in exactly this shape:
{"symptoms": [{"service": "<name>", "signal": "<copy the metric or log pattern from the evidence>",
               "kind": "shape|presence|error_rate|log_new|log_vanished|log_rate|quiet",
               "first_seen_s": <number or null>, "strength": "clear|weak", "why": "<one short sentence>"}],
 "candidates": [{"service": "<name>", "why": "<one short sentence: why investigate this service>"}]}

Rules: use only service names from the list of services given below; order "candidates" most suspicious first;
include at most 8 symptoms and at most 5 candidates; copy signal names from the evidence rather than inventing them.

EVIDENCE:
"""


KEEP_ALIVE = "5m"  # how long Ollama holds a model after a call; runs unload at the end unless --keep-warm


def ollama_url(path=""):
    """Base URL from OLLAMA_HOST (same variable the ollama CLI uses), accepting `host:port` or a full URL.
    Not everyone runs Ollama on localhost:11434."""
    h = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not h.startswith(("http://", "https://")):
        h = "http://" + h
    return h.rstrip("/") + path


def _ollama_call(path, payload=None, timeout=60):
    import urllib.request
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(ollama_url(path), data=data, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _ollama_try(path, payload=None, timeout=60):
    """None if the endpoint is missing, the server is unreachable, or the response is not JSON.
    Endpoints differ between Ollama versions, so every optional call goes through here."""
    try:
        return _ollama_call(path, payload, timeout)
    except Exception:
        return None


def ollama_version():
    v = _ollama_try("/api/version", timeout=15)
    return (v or {}).get("version")


def ollama_models():
    """[{name, size_gb}] for installed models, [] if unreachable. Tolerates the `name` / `model` key
    difference between Ollama versions."""
    tags = _ollama_try("/api/tags", timeout=30) or {}
    out = []
    for m in tags.get("models", []) or []:
        name = m.get("name") or m.get("model")
        if name:
            out.append({"name": name, "size_gb": round((m.get("size") or 0) / 1e9, 1)})
    return sorted(out, key=lambda d: d["name"])


def ollama_installed():
    return [m["name"] for m in ollama_models()]


def resolve_model(name):
    """The installed model matching `name`, allowing the implicit `:latest` tag. None if absent."""
    have = ollama_installed()
    if name in have:
        return name
    if f"{name}:latest" in have:
        return f"{name}:latest"
    if name.endswith(":latest") and name[: -len(":latest")] in have:
        return name[: -len(":latest")]
    return None


def ollama_unload(model):
    """Free a model's VRAM. Returns True if the request succeeded. A model that is not installed (404) or an
    unreachable server is not an error here: there is nothing to unload."""
    import urllib.request
    body = {"model": model, "prompt": "hi", "stream": False, "keep_alive": 0, "options": {"num_predict": 1}}
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=600).read()
        return True
    except Exception:
        return False


def ollama_loaded():
    """Which models are resident. [] where /api/ps is absent or returns nothing (0.34.2 returned an empty
    list even with a model loaded), so never treat [] as proof that nothing is loaded.

    Its SIZE and CPU/GPU split are NOT reliable: on 0.34.3 it reported 1.3 GB and 24%/76% CPU/GPU for a model
    nvidia-smi showed holding 18.7 GB and generating at full GPU speed. Names only; read memory from
    gpu_memory_mb()."""
    ps = _ollama_try("/api/ps", timeout=30) or {}
    return [m.get("name") or m.get("model") for m in ps.get("models", []) or []]


def ensure_only(model, others=None, warn=True):
    """Unload other RESIDENT models before running `model`. This is an OPTIMISATION - it stops a second model
    holding VRAM and slowing the first call - so any failure warns and continues.

    Eviction is server-wide: on a shared Ollama this takes the model away from anyone else using it, so each
    eviction is announced rather than done silently."""
    installed = set(ollama_installed())
    loaded = set(ollama_loaded())
    candidates = set(others) if others is not None else (installed | set(MODELS_DEFAULT))
    evicted, failed = [], []
    for m in sorted(candidates):
        if m == model or m not in installed:
            continue
        if loaded and m not in loaded:
            continue  # not resident (when the server tells us); nothing to evict
        (evicted if ollama_unload(m) else failed).append(m)
    if warn and evicted:
        print(f"note: unloaded {', '.join(evicted)} to free VRAM for {model}. On a shared Ollama this "
              f"evicts it for other users too.")
    if warn and failed:
        print(f"note: could not unload {', '.join(failed)} (continuing; the first call may be slower "
              f"or spill to CPU if VRAM is tight)")
    return not failed


@lru_cache(maxsize=None)
def model_supports_thinking(model):
    """Ollama rejects `think` with HTTP 400 on models without the capability. Older servers take `name`
    instead of `model`, and older ones still report no `capabilities` at all - in which case we assume no
    thinking, which is the safe direction: the request simply omits `think`."""
    info = _ollama_try("/api/show", {"model": model}, timeout=120) or _ollama_try("/api/show", {"name": model}, timeout=120)
    return "thinking" in ((info or {}).get("capabilities") or [])


def ollama_chat(model, prompt, num_ctx, thinking=True, num_predict=1024, timeout=1800, schema=None,
                keep_alive=None):
    """One chat call at temperature 0. Records free GPU memory before the call (see gpu_memory_mb) and checks
    prompt_eval_count for silent truncation. Answers come from message.content only, never thinking.
    `thinking` is ignored (and recorded as unsupported) for models without the thinking capability."""
    import urllib.error
    import urllib.request
    gpu_before = gpu_memory_mb()
    supported = model_supports_thinking(model)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "keep_alive": KEEP_ALIVE if keep_alive is None else keep_alive,
            "options": {"num_ctx": num_ctx, "temperature": 0, "num_predict": num_predict}}
    if supported:
        body["think"] = thinking
    if schema is not None:  # Ollama structured output: without it qwen copies the table's "-" into numeric fields
        body["format"] = schema
    t0 = time.time()
    fallbacks = []
    # Drop the optional features one at a time if this server rejects them, rather than failing the run:
    # structured output needs a recent Ollama, and `think` only exists on newer builds.
    r = None
    for drop in ([], ["format"], ["format", "think"]):
        attempt = {k: v for k, v in body.items() if k not in drop}
        try:
            r = _ollama_call("/api/chat", attempt, timeout=timeout)
            if drop:
                fallbacks.append("dropped " + ", ".join(f"`{d}`" for d in drop) + " (server rejected it)")
            break
        except urllib.error.HTTPError as e:
            if e.code != 400 or drop == ["format", "think"]:
                raise
        except Exception:
            raise
    wall = time.time() - t0
    fallback = "; ".join(fallbacks)
    msg = r.get("message", {})
    expected = count_tokens(prompt, "qwen" if "qwen" in model else "gemma")
    ok, note = check_prompt_eval(expected, r.get("prompt_eval_count"))
    return {"content": msg.get("content") or "", "thinking_chars": len(msg.get("thinking") or ""),
            "meta": {"model": model, "thinking": thinking and supported, "fallback": fallback,
                     "thinking_supported": supported, "num_ctx": num_ctx, "wall_s": round(wall, 2),
                     "prompt_tokens_expected": expected, "prompt_eval_count": r.get("prompt_eval_count"),
                     "eval_count": r.get("eval_count"), "done_reason": r.get("done_reason"),
                     "truncation_ok": ok, "truncation_note": note,
                     "gpu_free_before_MB": gpu_before[0] if gpu_before else None,
                     "gpu_total_MB": gpu_before[2] if gpu_before else None}}


def _parse_json_block(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        raise ValueError("no JSON object in response")
    return json.loads(t[i:j + 1])


STEP1_SCHEMA = {  # Ollama structured output, so a malformed number can't cost a whole run
    "type": "object",
    "properties": {
        "symptoms": {"type": "array", "items": {"type": "object", "properties": {
            "service": {"type": "string"}, "signal": {"type": "string"}, "kind": {"type": "string"},
            "first_seen_s": {"type": ["number", "null"]}, "strength": {"type": "string"}, "why": {"type": "string"}},
            "required": ["service", "signal", "kind", "strength", "why"]}},
        "candidates": {"type": "array", "items": {"type": "object", "properties": {
            "service": {"type": "string"}, "why": {"type": "string"}}, "required": ["service", "why"]}},
    },
    "required": ["symptoms", "candidates"],
}


def step1_llm(case, model="qwen2.5-coder:7b", thinking=True, include_artifacts=True, max_pat_rows=None,
              num_predict=None, schema=STEP1_SCHEMA, order_seed=DEFAULT_ORDER_SEED):
    """LLM symptom detection on the step-0 text. Same output shape as step1_python.
    Services are validated against the case's own service list; invented names are recorded, not silently kept."""
    thinking = thinking and model_supports_thinking(model)  # qwen2.5-coder has no thinking capability
    evidence = render_step0(case, include_artifacts=include_artifacts, max_pat_rows=max_pat_rows,
                            order_seed=order_seed)
    prompt = STEP1_INSTRUCTIONS + evidence
    services = set(step0_metrics(case)[1])
    n_tok = count_tokens(prompt, "qwen" if "qwen" in model else "gemma")
    num_ctx = num_ctx_for(n_tok, model, thinking)
    source = f"llm:{model}{'-think' if thinking else '-nothink'}{'-shuffled' if order_seed is not None else ''}"

    attempts, data, err = [], None, None
    if num_predict is None:
        num_predict = answer_reserve_for(model, thinking)
    for attempt in range(2):
        r = ollama_chat(model, prompt, num_ctx, thinking=thinking, num_predict=num_predict, schema=schema)
        attempts.append(r["meta"] | {"thinking_chars": r["thinking_chars"], "content_chars": len(r["content"])})
        try:
            data = _parse_json_block(r["content"])
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    meta = {"case": case, "source": source, "step1_version": STEP1_VERSION, "step0_version": STEP0_VERSION,
            "include_artifacts": include_artifacts, "order_seed": order_seed,
            "service_order": service_order(case, order_seed), "evidence_tokens": count_tokens(evidence, "qwen" if "qwen" in model else "gemma"),
            "prompt_tokens": n_tok, "attempts": attempts, "parse_error": err if data is None else None,
            "n_attempts": len(attempts)}
    if data is None:
        empty_s = pd.DataFrame(columns=SYMPTOM_COLS)
        empty_c = pd.DataFrame(columns=CANDIDATE_COLS)
        return {"candidates": empty_c, "symptoms": empty_s, "meta": meta}

    sym_rows, unknown = [], []
    for s in data.get("symptoms", [])[:20]:
        svc = str(s.get("service", ""))
        if svc not in services:
            unknown.append(svc)
        sym_rows.append({"case": case, "source": source, "service": svc, "signal": str(s.get("signal", ""))[:160],
                         "kind": str(s.get("kind", "")), "strength": str(s.get("strength", "")),
                         "onset_s": s.get("first_seen_s"), "size": "", "size_num": np.nan,
                         "artifact_flag": "", "reason": str(s.get("why", ""))[:200]})
    sym = pd.DataFrame(sym_rows, columns=SYMPTOM_COLS)
    cand_rows = []
    for i, c in enumerate(data.get("candidates", [])[:10], start=1):
        svc = str(c.get("service", ""))
        if svc not in services:
            unknown.append(svc)
        g = sym[sym.service == svc]
        cand_rows.append({"case": case, "source": source, "rank": i, "service": svc,
                          "reason": str(c.get("why", ""))[:300],  # the model's STATED reason, kept verbatim
                          "n_signals": len(g), "first_onset_s": pd.to_numeric(g.onset_s, errors="coerce").min(),
                          "has_clear": bool((g.strength == "clear").any()), "artifact_only": False})
    meta["unknown_services"] = sorted(set(unknown))
    return {"candidates": pd.DataFrame(cand_rows, columns=CANDIDATE_COLS), "symptoms": sym, "meta": meta}


# Hand-check expectations: subtle evidence each case must still carry after step 1 (from explore.ipynb parts B/C).
# (label, service, accepted kinds, accepted keywords) - a hit needs the service plus either the kind or a
# keyword, because each arm phrases the same signal differently (the model rewrites signal names).
RETENTION_CHECKS = {
    "re3ss_carts_f1_1": [("carts WARN 'POST not supported'", "carts", {"log_new"}, ["not supported", "pagenotfound", "warn"]),
                         ("carts goes quiet (restart)", "carts", {"quiet"}, ["log volume", "quiet", "restart"]),
                         ("carts error-seconds rise", "carts", {"error_rate"}, ["error"])],
    "re3ss_orders_f3_1": [("orders INFO 'payment response: null'", "orders", {"log_new"}, ["payment response", "null"]),
                          ("front-end 'Not Acceptable' orders", "front-end", {"log_new", "log_rate"}, ["not acceptable", "406"]),
                          ("orders error-seconds rise", "orders", {"error_rate"}, ["error"])],
    "re2ss_user_loss_1": [("user latency goes quiet", "user", {"presence", "shape"}, ["latency", "quiet"]),
                          ("user error metric appears", "user", {"error_rate", "presence"}, ["error"]),
                          ("user log volume quiet", "user", {"quiet"}, ["log volume", "quiet"])],
}


def auto_retention_checks(case, n=3):
    """Checklist derived from the TRUE root cause's own evidence, for cases without a hand-written one.
    Validation only - never part of a model prompt. Prefers the signals the earlier analysis showed are easy
    to lose: presence changes, error-rate rises, new log patterns, then the strongest metric shift."""
    r = case_info(case)
    svc = r["root_cause_service"]
    sym = step1_symptoms(case)
    own = sym[sym.service == svc]
    picks, seen = [], set()
    for kind, label, words in [("presence", "presence change", ["presence", "quiet", "appear"]),
                               ("error_rate", "error-rate rise", ["error"]),
                               ("log_new", "new log pattern", ["new"]),
                               ("quiet", "log volume quiet", ["log volume", "quiet"]),
                               ("log_rate", "log rate change", ["rate"]),
                               ("shape", "metric shift", [])]:
        g = own[own.kind == kind]
        if len(g) and kind not in seen:
            top = g.sort_values("size_num", ascending=False).iloc[0]
            picks.append((f"{svc} {label} ({top.signal[:40]})", svc, {kind}, words))
            seen.add(kind)
        if len(picks) >= n:
            break
    return picks


def step1_sheet(case, results, checks=RETENTION_CHECKS, top_n=5):
    """One-screen hand-check sheet per case. Ground truth is shown for the human; it is never in the model input."""
    r = case_info(case)
    truth = r["root_cause_service"]
    out = [f"=== {case} | truth: {truth} | fault {r['fault']} ({r['dataset']})"]
    for res in results:
        c, s, meta = res["candidates"], res["symptoms"], res["meta"]
        src = meta["source"]
        top = list(c.service.head(top_n))
        pos = (list(c.service).index(truth) + 1) if truth in list(c.service) else None
        louder = [x for x in top[:pos - 1]] if pos else top
        line = f"  [{src}] top{top_n}: {', '.join(top) if top else '(none)'}"
        out.append(line)
        out.append(f"      truth rank: {pos if pos else 'ABSENT'}" + (f" | ranked above truth: {', '.join(louder)}" if louder else "")
                   + (f" | parse_error: {meta['parse_error']}" if meta.get("parse_error") else "")
                   + (f" | invented services: {meta['unknown_services']}" if meta.get("unknown_services") else ""))
        if truth in list(c.service):
            why = c[c.service == truth].reason.iloc[0]
            out.append(f"      stated reason for {truth}: {why[:220]}")
        # retention: is each expected subtle signal still present in this version's symptoms?
        got = []
        text = (s.signal.fillna("") + " " + s.reason.fillna("")).str.lower()
        case_checks = checks.get(case) or auto_retention_checks(case)
        for label, svc, kinds, words in case_checks:
            same_svc = s.service == svc
            by_kind = same_svc & s.kind.isin(kinds)
            by_word = same_svc & text.apply(lambda t: any(w in t for w in words))
            got.append(("KEPT " if bool((by_kind | by_word).any()) else "LOST ") + label)
        if got:
            out.append("      retention: " + " | ".join(got))
        if meta.get("attempts"):
            a = meta["attempts"][-1]
            out.append(f"      cost: {a['prompt_eval_count']} prompt tok, {a['eval_count']} eval tok, "
                       f"{a['thinking_chars']} thinking chars, {a['wall_s']}s, num_ctx {a['num_ctx']}, "
                       f"gpu free before {a['gpu_free_before_MB']} MB, truncation_ok {a['truncation_ok']}")
    return "\n".join(out)


# ============================================================ step 2: symptom -> origin (tracing)
# Step 2 sees ONLY step 1's output plus a dependency block (decided), so step 1's retention differences stay
# visible. Output is the step 1 shape plus role / path / direction_evidence.
STEP2_VERSION = "step2-v0.2-label-only"
STEP3_VERSION = "step3-v0.1"
STEP2_CAND_COLS = CANDIDATE_COLS + ["role", "path", "direction_evidence", "topology_only"]

# A datastore whose only symptoms are connection/socket churn is NOT treated as an origin by the "rule" arm:
# when a service is redeployed its database logs mass connection churn, which made carts-db look like the
# deepest failing node in re3ss_carts_f1_1 (the wrong answer the naive rule gives).
CHURN_WORDS = ["connection", "socket", "conn", "端"]
CHURN_KINDS = {"quiet", "log_oneoff", "log_rare"}


def _is_churn_only(symptoms):
    """True when a service's symptoms are only connection/socket churn or log-volume noise."""
    if not len(symptoms):
        return False
    for x in symptoms.itertuples():
        sig = f"{x.signal} {x.reason}".lower()
        churn = ("socket" in sig) or any(w in sig for w in CHURN_WORDS) or x.kind in CHURN_KINDS
        if not churn:
            return False  # something substantive (cpu/mem/latency/error/new pattern)
    return True


def trace_graph(case, inject_time=None):
    """Per-case edges and per-service exclusive latency from traces. Returns (edges_df, exclusive_df)."""
    t = load_inject_time(case) if inject_time is None else inject_time
    tr = pd.read_parquet(f"{DATA_DIR}/{case}/traces.parquet",
                         columns=["traceID", "spanID", "parentSpanID", "serviceName", "startTime", "duration", "statusCode"])
    # traces use pandas nullable Int64: NA breaks float()/mean(), so coerce first (NA duration -> 0, NA status -> 0)
    for col in ["startTime", "duration", "statusCode"]:
        tr[col] = pd.to_numeric(tr[col], errors="coerce").fillna(0).astype(float)
    tr["after"] = (tr.startTime / 1e6) >= t
    parent = tr[["spanID", "serviceName"]].rename(columns={"spanID": "parentSpanID", "serviceName": "caller"})
    j = tr.merge(parent, on="parentSpanID")
    j = j[j.caller != j.serviceName]
    edges = (j.groupby(["caller", "serviceName", "after"])
             .agg(calls=("spanID", "size"), err_rate=("statusCode", lambda s: float((s != 0).mean())),
                  median_ms=("duration", lambda d: float(d.median() or 0) / 1000.0)).reset_index())
    # exclusive (self) latency: a span's duration minus the time its children took
    child_sum = j.groupby(["parentSpanID", "after"]).duration.sum().rename("child_us").reset_index()
    sp = tr.merge(child_sum.rename(columns={"parentSpanID": "spanID"}), on=["spanID", "after"], how="left")
    sp["child_us"] = sp.child_us.fillna(0)
    sp["exclusive_us"] = (sp.duration - sp.child_us).clip(lower=0)
    excl = sp.groupby(["serviceName", "after"]).exclusive_us.median().rename("exclusive_us").reset_index()
    return edges, excl


def _call_shaped(svc):
    """The service name in a CALL context: a URL path (http://carts/...), a host:port (carts-db:27017), or a
    connection string. A bare mention is not a dependency: carts logs "Creating item for user: <id>", which
    produced 1456 false pieces of evidence for a carts -> user edge that does not exist."""
    return re.compile(rf"(?<![\w-]){re.escape(svc)}(?::\d|/)")


def _bare_mention(svc):
    return re.compile(rf"(?<![\w-]){re.escape(svc)}(?![\w-])")


def log_graph(case, shape="call"):
    """Edges from one service's logs naming another. Sock Shop has no traces, so this is the only per-case
    evidence of who calls whom there. shape='call' requires a call context; 'bare' is the old behaviour,
    kept so the false-edge count can be measured."""
    L = load_logs(case)
    svcs = sorted(set(L.container_name))
    pats = {s: (_call_shaped(s) if shape == "call" else _bare_mention(s)) for s in svcs}
    rows = Counter()
    for c, msg, rel in zip(L.container_name, L.message.fillna(""), L.rel):
        for s, rx in pats.items():
            if s != c and rx.search(msg):
                rows[(c, s, rel >= 0)] += 1
    return pd.DataFrame([{"caller": a, "serviceName": b, "after": aft, "mentions": n}
                         for (a, b, aft), n in rows.items()])


# Asynchronous work has no caller->callee line in logs or traces: shipping publishes to rabbitmq and
# queue-master consumes. Without these edges a consumer looks like a service with no dependencies, and the
# "deepest failing node" rule crowns it the origin (it did, in re3ss_orders_f3_1).
QUEUE_PRODUCER = re.compile(r"(?i)\b(adding|publish\w*|sending|sent|enqueue\w*|added)\b[^.]{0,40}\b(to\s+)?queue\b|\bqueue\b[^.]{0,20}\b(task|message|shipment)\b")
QUEUE_CONSUMER = re.compile(r"(?i)\breceived\b[^.]{0,30}\b(task|message|shipment)\b|\bconsum\w+\b[^.]{0,30}\b(task|message|queue)\b")
BROKERS = ("rabbitmq", "kafka", "redis", "queue")


def queue_edges(case, min_lines=3):
    """producer -> broker -> consumer edges inferred from log phrasing, with the broker named when one of the
    system's services looks like one. Evidence is per-case (the phrases come from this case's own logs)."""
    L = load_logs(case)
    svcs = sorted(set(L.container_name))
    broker = next((s for s in svcs if any(b in s for b in BROKERS)), None)
    prod = Counter()
    cons = Counter()
    for c, msg in zip(L.container_name, L.message.fillna("")):
        if QUEUE_PRODUCER.search(msg):
            prod[c] += 1
        elif QUEUE_CONSUMER.search(msg):
            cons[c] += 1
    rows = []
    for p_svc, n in prod.items():
        for c_svc, m in cons.items():
            if p_svc == c_svc or n < min_lines or m < min_lines:
                continue
            if broker and broker not in (p_svc, c_svc):
                rows.append({"caller": p_svc, "callee": broker, "provenance": "log-queue", "per_case_evidence": True,
                             "queue_lines": n})
                rows.append({"caller": broker, "callee": c_svc, "provenance": "log-queue", "per_case_evidence": True,
                             "queue_lines": m})
            else:
                rows.append({"caller": p_svc, "callee": c_svc, "provenance": "log-queue", "per_case_evidence": True,
                             "queue_lines": min(n, m)})
    return pd.DataFrame(rows).drop_duplicates(subset=["caller", "callee"]) if rows else pd.DataFrame()


@lru_cache(maxsize=64)
def _case_call_graph_cached(case):
    return _case_call_graph(case)


def case_call_graph(case):
    """Cached per case: trace parsing is single-core and was repeated for every arm. Callers must not mutate
    the returned frames (nothing in the pipeline does)."""
    return _case_call_graph_cached(case)


def _case_call_graph(case):
    """Merged graph with per-edge provenance. provenance: trace | log | static | topology.
    per_case_evidence is False for static/topology edges - results that hinge on those are flagged."""
    info = case_info(case)
    system = info["system"]
    rows = []
    if bool(info["has_traces"]):
        edges, excl = trace_graph(case)
        wide = edges.pivot_table(index=["caller", "serviceName"], columns="after",
                                 values=["calls", "err_rate", "median_ms"]).reset_index()
        for _, e in wide.iterrows():
            get = lambda k, a: e.get((k, a), np.nan)
            rows.append({"caller": e["caller"].iloc[0] if hasattr(e["caller"], "iloc") else e["caller"],
                         "callee": e["serviceName"].iloc[0] if hasattr(e["serviceName"], "iloc") else e["serviceName"],
                         "provenance": "trace", "per_case_evidence": True,
                         "calls_pre": get("calls", False), "calls_post": get("calls", True),
                         "err_pre": get("err_rate", False), "err_post": get("err_rate", True),
                         "ms_pre": get("median_ms", False), "ms_post": get("median_ms", True)})
    else:
        excl = pd.DataFrame(columns=["serviceName", "after", "exclusive_us"])
    if bool(info["has_logs"]):
        lg = log_graph(case)
        if len(lg):
            wide = lg.pivot_table(index=["caller", "serviceName"], columns="after", values="mentions").reset_index()
            for _, e in wide.iterrows():
                rows.append({"caller": e["caller"], "callee": e["serviceName"], "provenance": "log",
                             "per_case_evidence": True,
                             "mentions_pre": e.get(False, np.nan), "mentions_post": e.get(True, np.nan)})
    if bool(info["has_logs"]):
        q = queue_edges(case)
        rows += [r for _, r in q.iterrows()] if len(q) else []
    have = {(r["caller"], r["callee"]) for r in rows}
    for a, b in STATIC_EDGES.get(system, set()):
        if (a, b) not in have:
            rows.append({"caller": a, "callee": b, "provenance": "static", "per_case_evidence": False})
            have.add((a, b))
    if not bool(info["has_traces"]):  # RE1: reuse the same system's topology from RE2/RE3 traces
        for a, b in _trace_edges_for(system):
            if (a, b) not in have:
                rows.append({"caller": a, "callee": b, "provenance": "topology", "per_case_evidence": False})
                have.add((a, b))
    g = pd.DataFrame(rows)
    return g, excl


def _edge_note(e):
    if e.get("provenance") == "trace":
        bits = []
        if not pd.isna(e.get("err_post")) and (e.get("err_post") or 0) > (e.get("err_pre") or 0):
            bits.append(f"errors {e['err_pre']:.0%}->{e['err_post']:.0%}")
        if not pd.isna(e.get("ms_post")):
            bits.append(f"{e['ms_pre']:.0f}->{e['ms_post']:.0f} ms")
        if not pd.isna(e.get("calls_post")):
            bits.append(f"{int(e['calls_pre'] or 0)}->{int(e['calls_post'] or 0)} calls")
        return ", ".join(bits) or "trace edge"
    if e.get("provenance") == "log-queue":
        return f"asynchronous queue, {int(e.get('queue_lines') or 0)} log lines"
    if e.get("provenance") == "log":
        num = lambda v: 0 if v is None or pd.isna(v) else int(v)  # an edge seen only on one side gives NaN
        return f"log mentions {num(e.get('mentions_pre'))}->{num(e.get('mentions_post'))}"
    return "architecture only, no per-case evidence"


def step2_dependency_block(case, candidates, graph, excl, max_services=8):
    """Compact per-candidate dependency context: callers, callees, per-edge change, provenance."""
    out = ["== DEPENDENCIES of the candidate services (who calls whom) ==",
           "Edges marked 'architecture only' have no evidence in this case's own data."]
    for svc in list(candidates.service)[:max_services]:
        callees = graph[graph.caller == svc] if len(graph) else graph
        callers = graph[graph.callee == svc] if len(graph) else graph
        def fmt(df, col):
            return "; ".join(f"{r[col]} ({r['provenance']}: {_edge_note(r)})" for _, r in df.iterrows()) or "(none known)"
        out.append(f"{svc}: calls -> {fmt(callees, 'callee')}")
        out.append(f"{svc}: called by <- {fmt(callers, 'caller')}")
    if len(excl):
        w = excl.pivot_table(index="serviceName", columns="after", values="exclusive_us")
        lines = []
        for svc in list(candidates.service)[:max_services]:
            if svc in w.index:
                pre, post = w.loc[svc].get(False, np.nan), w.loc[svc].get(True, np.nan)
                if not pd.isna(pre) and not pd.isna(post):
                    lines.append(f"{svc} {pre/1000:.1f}->{post/1000:.1f} ms")
        if lines:
            out.append("Own (exclusive) time per request, excluding time spent waiting for dependencies: " + "; ".join(lines))
    return "\n".join(out)


def step2_python(case, step1_result, rule="rule"):
    """rule='naive': a service is an origin if no dependency of it is symptomatic (deepest failing node).
    rule='rule':  same, but a dependency whose symptoms are ONLY connection/socket churn does not count -
                  a redeployed service makes its datastore log connection churn (see CHURN_WORDS)."""
    t0 = time.time()
    cands, syms = step1_result["candidates"].copy(), step1_result["symptoms"]
    graph, excl = case_call_graph(case)
    symptomatic = set(cands.service)
    # Which services are symptomatic comes from step 1 (step 2 corrects step 1). What KIND of symptoms a
    # dependency has is read from step 0's evidence, not from step 1's wording: an LLM step 1 paraphrases
    # signals, and the churn rule cannot match paraphrases. A Python arm may read data it already holds.
    base_syms = step1_symptoms(case)

    def counts_as_symptomatic(svc):
        if svc not in symptomatic:
            return False
        if rule == "naive":
            return True
        return not _is_churn_only(base_syms[base_syms.service == svc])

    rows = []
    for x in cands.itertuples():
        callees = list(graph[graph.caller == x.service].callee) if len(graph) else []
        sym_callees = [c for c in callees if counts_as_symptomatic(c)]
        deciding = graph[(graph.caller == x.service) & (graph.callee.isin(sym_callees))] if sym_callees else graph.iloc[0:0]
        topology_only = bool(len(deciding)) and not bool(deciding.per_case_evidence.any())
        if sym_callees:
            role = "victim"
        elif callees:
            role = "origin"
        else:
            role = "unknown"  # no dependency information: absence of evidence is not evidence of origin
        path = " -> ".join([x.service] + sym_callees[:2]) if sym_callees else x.service
        ev = (f"symptomatic dependencies: {', '.join(sym_callees[:3])}" if sym_callees
              else (f"no symptomatic dependency (deps: {', '.join(callees[:3])})" if callees
                    else "NO DEPENDENCY INFORMATION for this service - cannot tell origin from victim"))
        rows.append({**{c: getattr(x, c) for c in CANDIDATE_COLS if c != "rank"},
                     "rank": x.rank, "role": role, "path": path, "direction_evidence": ev,
                     "topology_only": topology_only})
    out = pd.DataFrame(rows, columns=STEP2_CAND_COLS)
    # LABEL-ONLY (measured: re-ranking helped 16 runs and hurt 51 of 120). Step 1's order is left untouched;
    # role / path / direction_evidence are attached for step 3 to weigh. would_demote records what the
    # re-ranking version would have done, so the decision stays measurable.
    out["would_demote"] = (out.role == "victim") & (~out.topology_only)
    out = out.sort_values("rank")
    out["source"] = f"{step1_result['meta']['source']}+py2:{rule}"
    return {"candidates": out.reset_index(drop=True), "symptoms": syms,
            "meta": {"case": case, "source": out.source.iloc[0] if len(out) else f"py2:{rule}",
                     "step2_version": STEP2_VERSION, "rule": rule, "label_only": True,
                     "step1_source": step1_result["meta"]["source"],
                     "graph_edges": len(graph), "edges_without_case_evidence": int((~graph.per_case_evidence).sum()) if len(graph) else 0,
                     "wall_s": round(time.time() - t0, 2)}}


STEP2_INSTRUCTIONS = """A fault started at t=0 in a microservice system. Symptom detection has already run.

Below are the candidate services with their symptoms, and a dependency block showing who calls whom.

Your task: decide which service is the ORIGIN of the fault and which are VICTIMS. A service that calls a
broken dependency shows symptoms too (errors, latency, retries), but it is a victim, not the origin.
Beware the reverse trap: when a service restarts or fails, its datastore logs connection churn - that
does not make the datastore the origin.

If a service's dependencies are unknown (the block says so), you cannot tell whether it is an origin or a
victim: put it in "insufficient_evidence" rather than calling it an origin.

Answer with JSON only:
{"origins": [{"service": "<name>", "why": "<one sentence>", "path": "<victim -> ... -> origin, or the service alone>"}],
 "victims": [{"service": "<name>", "of_service": "<which dependency it is a victim of>", "why": "<one sentence>"}],
 "insufficient_evidence": [{"service": "<name>", "why": "<what is missing>"}]}

Use only service names from the candidates below; order "origins" most likely first; at most 3 origins.

"""

STEP2_SCHEMA = {
    "type": "object",
    "properties": {
        "origins": {"type": "array", "items": {"type": "object", "properties": {
            "service": {"type": "string"}, "why": {"type": "string"}, "path": {"type": "string"}},
            "required": ["service", "why"]}},
        "victims": {"type": "array", "items": {"type": "object", "properties": {
            "service": {"type": "string"}, "of_service": {"type": "string"}, "why": {"type": "string"}},
            "required": ["service", "why"]}},
        "insufficient_evidence": {"type": "array", "items": {"type": "object", "properties": {
            "service": {"type": "string"}, "why": {"type": "string"}}, "required": ["service"]}},
    },
    "required": ["origins", "victims"],
}


def step2_llm(case, step1_result, model="qwen2.5-coder:7b", thinking=False, num_predict=None):
    """Same inputs as step2_python (step 1 output + dependency block only), same output shape."""
    cands, syms = step1_result["candidates"], step1_result["symptoms"]
    graph, excl = case_call_graph(case)
    lines = ["== CANDIDATE SERVICES from symptom detection (in the order symptom detection ranked them) =="]
    for x in cands.itertuples():
        own = syms[syms.service == x.service]
        sig = "; ".join(f"{s.kind} {s.signal[:60]}" for s in own.head(3).itertuples()) or "(no detail)"
        lines.append(f"{x.service}: {str(x.reason)[:160]} | symptoms: {sig}")
    prompt = STEP2_INSTRUCTIONS + "\n".join(lines) + "\n\n" + step2_dependency_block(case, cands, graph, excl)
    services = set(cands.service)
    thinking = thinking and model_supports_thinking(model)
    n_tok = count_tokens(prompt, "qwen" if "qwen" in model else "gemma")
    if num_predict is None:
        num_predict = answer_reserve_for(model, thinking)
    source = f"{step1_result['meta']['source']}+llm2:{model}"

    attempts, data, err = [], None, None
    for _ in range(2):
        r = ollama_chat(model, prompt, num_ctx_for(n_tok, model, thinking), thinking=thinking,
                        num_predict=num_predict, schema=STEP2_SCHEMA)
        attempts.append(r["meta"] | {"thinking_chars": r["thinking_chars"], "content_chars": len(r["content"])})
        try:
            data = _parse_json_block(r["content"])
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    meta = {"case": case, "source": source, "step2_version": STEP2_VERSION, "step1_source": step1_result["meta"]["source"],
            "prompt_tokens": n_tok, "attempts": attempts, "parse_error": err if data is None else None,
            "label_only": True, "graph_edges": len(graph), "edges_without_case_evidence": int((~graph.per_case_evidence).sum()) if len(graph) else 0}
    if data is None:
        return {"candidates": pd.DataFrame(columns=STEP2_CAND_COLS), "symptoms": syms, "meta": meta}

    unknown, seen = [], set()
    labels = {}  # service -> (role, path, why) as judged by the model
    for o in data.get("origins", [])[:5]:
        svc = str(o.get("service", ""))
        if svc not in services:
            unknown.append(svc)
        labels.setdefault(svc, ("origin", str(o.get("path", ""))[:120], str(o.get("why", ""))[:300]))
    for v in data.get("victims", [])[:10]:
        svc = str(v.get("service", ""))
        if svc not in services:
            unknown.append(svc)
        labels.setdefault(svc, ("victim", f"{svc} -> {v.get('of_service', '?')}", str(v.get("why", ""))[:300]))
    for u in data.get("insufficient_evidence", [])[:10]:
        svc = str(u.get("service", ""))
        if svc not in services:
            unknown.append(svc)
        labels.setdefault(svc, ("unknown", svc, str(u.get("why", ""))[:300]))

    # LABEL-ONLY: step 1's order is kept; the model's judgement becomes role / path / direction_evidence.
    rows = []
    for x in cands.itertuples():
        role, path, why = labels.get(x.service, ("unlisted", x.service, "not mentioned by the model"))
        rows.append({**{c: getattr(x, c) for c in CANDIDATE_COLS}, "role": role, "path": path,
                     "direction_evidence": why, "topology_only": False})
    meta["unknown_services"] = sorted(set(unknown))
    return {"candidates": pd.DataFrame(rows, columns=STEP2_CAND_COLS), "symptoms": syms, "meta": meta}


def step2_sheet(case, step1_result, step2_results, known_victims=()):
    """Rank before vs after step 2, plus role/path and whether known victims were demoted."""
    truth = case_info(case)["root_cause_service"]
    before = list(step1_result["candidates"].service)
    r0 = before.index(truth) + 1 if truth in before else None
    out = [f"=== {case} | truth: {truth} | step 1 [{step1_result['meta']['source']}] rank {r0 or 'ABSENT'}: {', '.join(before[:5])}"]
    for res in step2_results:
        c, meta = res["candidates"], res["meta"]
        names = list(c.service)
        r1 = names.index(truth) + 1 if truth in names else None
        move = "=" if r0 == r1 else ("better" if (r1 or 99) < (r0 or 99) else "worse")
        out.append(f"  [{meta['source']}] rank {r1 or 'ABSENT'} ({move}): {', '.join(names[:5])}")
        if truth in names:
            row = c[c.service == truth].iloc[0]
            out.append(f"      role={row.role} path={row.path} | {str(row.direction_evidence)[:110]}"
                       + (" | TOPOLOGY-ONLY EDGE" if row.topology_only else ""))
        for v in known_victims:
            if v in names:
                vr = c[c.service == v].iloc[0]
                out.append(f"      known victim {v}: rank {names.index(v) + 1}, role={vr.role}")
        if meta.get("parse_error"):
            out.append(f"      parse_error: {meta['parse_error']}")
        if meta.get("attempts"):
            a = meta["attempts"][-1]
            out.append(f"      cost: {a['prompt_eval_count']} prompt tok, {a['eval_count']} eval tok, {a['wall_s']}s, "
                       f"gpu free {a['gpu_free_before_MB']} MB")
        if meta.get("edges_without_case_evidence"):
            out.append(f"      graph: {meta['graph_edges']} edges, {meta['edges_without_case_evidence']} without per-case evidence")
    return "\n".join(out)



# ============================================================ step 3: the final root-cause decision
# One answer per case, or an explicit abstention (scored separately - a refusal is not a wrong answer).
# Input is step 1's ranking with step 2's labels attached; no raw evidence, so step 1's retention still governs
# what is visible. P-top1 is the control: if no configuration beats "take step 1's top candidate", that is the
# result of the project.
STEP3_COLS = ["case", "source", "rank", "service", "reason", "role", "path", "confidence"]


def _confidence(row, has_origin_role):
    if row is None:
        return "low"
    if bool(getattr(row, "has_clear", False)) and has_origin_role:
        return "high"
    return "medium" if bool(getattr(row, "has_clear", False)) else "low"


def _step3_result(case, source, service, confidence, justification, row=None, abstained=False, extra=None):
    cols = {"case": case, "source": source, "rank": 1, "service": service,
            "reason": justification, "role": getattr(row, "role", "") if row is not None else "",
            "path": getattr(row, "path", "") if row is not None else "", "confidence": confidence}
    cands = pd.DataFrame([] if abstained else [cols], columns=STEP3_COLS)
    meta = {"case": case, "source": source, "step3_version": STEP3_VERSION, "answer": service,
            "confidence": confidence, "justification": justification, "abstained": abstained, **(extra or {})}
    return {"candidates": cands, "meta": meta}


def step3_python(case, step2_result, rule="top1"):
    """rule='top1': answer = step 1's top candidate (the control).
    rule='role':  answer = the highest-ranked candidate step 2 labelled 'origin'; falls back to top1 and
                  records the fallback. Abstains when no candidate has clear evidence."""
    cands, syms = step2_result["candidates"], step2_result["symptoms"]
    source = f"{step2_result['meta']['source']}+py3:{rule}"
    if not len(cands):
        return {**_step3_result(case, source, None, "none", "no candidates from step 1", abstained=True),
                "symptoms": syms}
    if not bool(cands.has_clear.any()):
        return {**_step3_result(case, source, None, "none",
                                "no candidate has clear evidence; abstaining rather than guessing",
                                abstained=True), "symptoms": syms}
    top = cands.iloc[0]
    pick, fallback = top, False
    if rule == "role":
        origins = cands[cands.role == "origin"] if "role" in cands else cands.iloc[0:0]
        if len(origins):
            pick = origins.iloc[0]
        else:
            fallback = True
    has_origin = "role" in cands and bool((cands.role == "origin").any())
    why = str(pick.reason)[:300]
    if rule == "role":
        why = (f"step 2 labelled it the origin; {why}" if not fallback
               else f"no candidate was labelled origin, fell back to step 1's top candidate; {why}")
    return {**_step3_result(case, source, pick.service, _confidence(pick, has_origin), why, row=pick,
                            extra={"fallback_to_top1": fallback}), "symptoms": syms}


STEP3_INSTRUCTIONS = """A fault started at t=0 in a microservice system. Symptom detection and dependency
tracing have already run. Below are the candidate services, each with its symptoms and its role
(origin = the fault started here; victim = it depends on a broken service; unknown = its dependencies are
not known; unlisted = tracing did not judge it).

Name the ONE service where the fault originated. If the evidence does not support any single service, answer
"none" - an honest abstention is better than a guess, and abstentions are scored separately from wrong answers.

Answer with JSON only:
{"answer": "<service name, or none>", "confidence": "high|medium|low", "justification": "<one sentence>"}

Use only service names from the list below.

"""

STEP3_SCHEMA = {"type": "object",
                "properties": {"answer": {"type": "string"}, "confidence": {"type": "string"},
                               "justification": {"type": "string"}},
                "required": ["answer", "confidence", "justification"]}


def step3_llm(case, step2_result, model="qwen2.5-coder:7b", thinking=False, num_predict=None):
    """Same input as step3_python. Abstention ("none") is allowed and recorded, never scored as a wrong answer."""
    cands, syms = step2_result["candidates"], step2_result["symptoms"]
    thinking = thinking and model_supports_thinking(model)
    lines = ["== CANDIDATES (in the order symptom detection ranked them) =="]
    for x in cands.itertuples():
        own = syms[syms.service == x.service] if len(syms) else syms
        sig = "; ".join(f"{s.kind} {s.signal[:55]}" for s in own.head(3).itertuples()) or "(no detail)"
        role = getattr(x, "role", "")
        path = getattr(x, "path", "")
        lines.append(f"{x.service} [role: {role}{', path: ' + path if path and path != x.service else ''}] "
                     f"| {str(x.reason)[:140]} | symptoms: {sig}")
    prompt = STEP3_INSTRUCTIONS + "\
".join(lines)
    services = set(cands.service)
    n_tok = count_tokens(prompt, "qwen" if "qwen" in model else "gemma")
    if num_predict is None:
        num_predict = answer_reserve_for(model, thinking)
    source = f"{step2_result['meta']['source']}+llm3:{model}"

    attempts, data, err = [], None, None
    for _ in range(2):
        r = ollama_chat(model, prompt, num_ctx_for(n_tok, model, thinking), thinking=thinking,
                        num_predict=num_predict, schema=STEP3_SCHEMA)
        attempts.append(r["meta"] | {"thinking_chars": r["thinking_chars"], "content_chars": len(r["content"])})
        try:
            data = _parse_json_block(r["content"])
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    extra = {"prompt_tokens": n_tok, "attempts": attempts, "parse_error": err if data is None else None}
    if data is None:
        return {**_step3_result(case, source, None, "none", "unparseable answer", abstained=True, extra=extra),
                "symptoms": syms}
    answer = str(data.get("answer", "")).strip()
    conf = str(data.get("confidence", ""))[:10]
    why = str(data.get("justification", ""))[:300]
    if normalize_service(answer) in ("none", "", "unknown"):
        return {**_step3_result(case, source, None, conf or "none", why or "model abstained", abstained=True,
                                extra=extra), "symptoms": syms}
    if answer not in services:  # a name not on the list is not an answer we can score
        extra["unknown_services"] = [answer]
        return {**_step3_result(case, source, None, conf, f"model named a service not in the candidate list: {answer}",
                                abstained=True, extra=extra), "symptoms": syms}
    row = cands[cands.service == answer].iloc[0]
    return {**_step3_result(case, source, answer, conf, why, row=row, extra=extra), "symptoms": syms}


# ============================================================ direct arm: step-0 evidence straight to the model
# No Python ranking, no symptom detection, no tracing: can the model do the whole job from the evidence?
# It is asked for the signals it used, so retention stays measurable and comparable with the staged arms.
DIRECT_VERSION = "direct-v0.1"

DIRECT_INSTRUCTIONS = """You are diagnosing a fault in a microservice system. A fault started at t=0.

Below is a summary of everything that changed after t=0, for every service, in no particular order.

Name the ONE service where the fault ORIGINATED. A service that calls a broken dependency also shows
symptoms (errors, latency, retries) but is a victim, not the origin. When a service restarts or fails, its
datastore logs connection churn - that does not make the datastore the origin.

If the evidence does not support any single service, answer "none": an honest abstention is better than a
guess, and abstentions are scored separately from wrong answers.

Answer with JSON only:
{"answer": "<service name, or none>", "confidence": "high|medium|low",
 "justification": "<one sentence>",
 "signals": [{"service": "<name>", "signal": "<copy the metric or log pattern you used>",
              "kind": "shape|presence|error_rate|log_new|log_vanished|log_rate|quiet"}]}

Use only service names from the evidence. List at most 5 signals.

EVIDENCE:
"""

DIRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"}, "confidence": {"type": "string"}, "justification": {"type": "string"},
        "signals": {"type": "array", "items": {"type": "object", "properties": {
            "service": {"type": "string"}, "signal": {"type": "string"}, "kind": {"type": "string"}},
            "required": ["service", "signal"]}},
    },
    "required": ["answer", "confidence", "justification"],
}


def direct_llm(case, model="qwen2.5-coder:7b", thinking=False, include_artifacts=True, max_pat_rows=None,
               num_predict=None, order_seed=DEFAULT_ORDER_SEED, roles_from=None, graph_facts=False,
               variant=""):
    """One call: step-0 evidence in, final answer out. Same output shape as step3_*.
    roles_from: a step-2 result whose role labels are appended WITHOUT any ranking (services in the same
    shuffled order as the evidence), to separate "ranking anchors the model" from "structure anchors it".
    graph_facts: the call graph as plain facts (who calls whom, with the evidence behind each edge) and NO
    interpretation - no roles, no paths, no conclusions. Separates "facts about the system help" from
    "our conclusions hurt".
    max_pat_rows=CAPPED_PAT_ROWS gives the capped step-0 configuration."""
    thinking = thinking and model_supports_thinking(model)
    evidence = render_step0(case, include_artifacts=include_artifacts, max_pat_rows=max_pat_rows,
                            order_seed=order_seed)
    if graph_facts:
        graph, _ = case_call_graph(case)
        order = {svc: i for i, svc in enumerate(service_order(case, order_seed))}
        g = graph.assign(_o=graph.caller.map(lambda v: order.get(v, len(order)))).sort_values(["_o", "callee"])
        lines = [f"{e.caller} calls {e.callee} ({_edge_note(e._asdict())})" for e in g.itertuples()]
        evidence += ("\n\n== WHO CALLS WHOM (observed in this case's traces and logs, or from the published "
                     "architecture where stated; no interpretation) ==\n" + "\n".join(lines))
    if roles_from is not None:
        c = roles_from["candidates"]
        order = {svc: i for i, svc in enumerate(service_order(case, order_seed))}
        labelled = c.assign(_o=c.service.map(lambda v: order.get(v, len(order)))).sort_values("_o")
        lines = [f"{x.service}: {getattr(x, 'role', '')}"
                 + (f" ({x.path})" if getattr(x, "path", "") and x.path != x.service else "")
                 for x in labelled.itertuples()]
        evidence += ("\n\n== DEPENDENCY ROLES (from tracing; NOT a ranking - services are in the same "
                     "arbitrary order as above) ==\n" + "\n".join(lines))
    prompt = DIRECT_INSTRUCTIONS + evidence
    services = set(step0_metrics(case)[1])
    n_tok = count_tokens(prompt, "qwen" if "qwen" in model else "gemma")
    if num_predict is None:
        num_predict = answer_reserve_for(model, thinking)
    source = f"direct{('-' + variant) if variant else ''}:{model}{'-think' if thinking else ''}"

    attempts, data, err = [], None, None
    for _ in range(2):
        r = ollama_chat(model, prompt, num_ctx_for(n_tok, model, thinking), thinking=thinking,
                        num_predict=num_predict, schema=DIRECT_SCHEMA)
        attempts.append(r["meta"] | {"thinking_chars": r["thinking_chars"], "content_chars": len(r["content"])})
        try:
            data = _parse_json_block(r["content"])
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    extra = {"prompt_tokens": n_tok, "evidence_tokens": count_tokens(evidence, "qwen" if "qwen" in model else "gemma"),
             "attempts": attempts, "parse_error": err if data is None else None, "order_seed": order_seed}
    empty = pd.DataFrame(columns=SYMPTOM_COLS)
    if data is None:
        return {**_step3_result(case, source, None, "none", "unparseable answer", abstained=True, extra=extra),
                "symptoms": empty}

    # the signals it says it used, so retention is comparable with the staged arms
    sym_rows = [{"case": case, "source": source, "service": str(x.get("service", "")),
                 "signal": str(x.get("signal", ""))[:160], "kind": str(x.get("kind", "")), "strength": "",
                 "onset_s": None, "size": "", "size_num": np.nan, "artifact_flag": "", "reason": ""}
                for x in (data.get("signals") or [])[:8]]
    syms = pd.DataFrame(sym_rows, columns=SYMPTOM_COLS)
    answer = str(data.get("answer", "")).strip()
    conf = str(data.get("confidence", ""))[:10]
    why = str(data.get("justification", ""))[:300]
    if normalize_service(answer) in ("none", "", "unknown"):
        return {**_step3_result(case, source, None, conf or "none", why or "model abstained", abstained=True,
                                extra=extra), "symptoms": syms}
    if answer not in services:
        extra["unknown_services"] = [answer]
        return {**_step3_result(case, source, None, conf,
                                f"model named a service that is not in this system: {answer}",
                                abstained=True, extra=extra), "symptoms": syms}
    return {**_step3_result(case, source, answer, conf, why, extra=extra), "symptoms": syms}

# ============================================================ run records (results/<timestamp>-<label>/)
# Every run writes its own folder, so re-runs never overwrite earlier results:
#   step1.parquet      one row per candidate per arm
#   step2.parquet      the same plus role / path / direction_evidence / would_demote
#   step3.parquet      one row per decision; direct.parquet the same for the direct arm
#   symptoms.parquet   the symptom rows behind those candidates
#   retention.parquet  ONE ROW PER EXPECTED ITEM (kept True/False), not just counts
#   summary.csv        one row per case per arm, openable in any spreadsheet
#   metadata.json      step versions, thresholds, seed, models, timings, GPU
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DEFAULT = ("qwen2.5-coder:7b", "gemma4:26b")
SAMPLE12 = [  # the hand-picked inspection set: fault kinds, signal shapes, sizes, and 2 weak-evidence cases
    ("re3ss_carts_f1_1", "RE3 redeploy, WARN signal"),
    ("re3ss_carts_f3_1", "RE3 same signature as f1"),
    ("re3ss_orders_f1_1", "RE3 obvious exception"),
    ("re3ss_orders_f3_1", "RE3 subtle INFO line"),
    ("re3ss_front-end_f1_1", "RE3 rate-only (pattern pre-exists)"),
    ("re2ss_user_loss_1", "loss, latency goes quiet"),
    ("re1ob_adservice_mem_1", "metrics only, small prompt"),
    ("re2ob_checkoutservice_disk_1", "disk, diskio-appears artifact"),
    ("re1tt_ts-order-service_cpu_1", "TrainTicket, metrics only"),
    ("re2tt_ts-order-service_delay_1", "TrainTicket, large prompt"),
    ("re1ob_cartservice_loss_4", "WEAK evidence only"),
    ("re3ss_carts_f4_1", "WEAK evidence only, has logs"),
]


SAMPLE50 = [  # SAMPLE12 plus a stratified spread: 6 cases per RE1/RE2 dataset, 4-6 per RE3, every fault type
    "re1ob_adservice_cpu_1", "re1ob_adservice_delay_1", "re1ob_adservice_disk_1", "re1ob_adservice_loss_1",
    "re1ob_adservice_mem_1", "re1ob_cartservice_loss_4",
    "re1ss_carts_cpu_1", "re1ss_carts_cpu_2", "re1ss_carts_delay_1", "re1ss_carts_disk_1",
    "re1ss_carts_loss_1", "re1ss_carts_mem_1",
    "re1tt_ts-auth-service_cpu_1", "re1tt_ts-auth-service_delay_1", "re1tt_ts-auth-service_disk_1",
    "re1tt_ts-auth-service_loss_1", "re1tt_ts-auth-service_mem_1", "re1tt_ts-order-service_cpu_1",
    "re2ob_checkoutservice_cpu_1", "re2ob_checkoutservice_delay_1", "re2ob_checkoutservice_disk_1",
    "re2ob_checkoutservice_disk_2", "re2ob_checkoutservice_loss_1", "re2ob_checkoutservice_mem_1",
    "re2ss_carts_cpu_1", "re2ss_carts_delay_1", "re2ss_carts_disk_1", "re2ss_carts_loss_1",
    "re2ss_carts_mem_1", "re2ss_user_loss_1",
    "re2tt_ts-auth-service_cpu_1", "re2tt_ts-auth-service_delay_1", "re2tt_ts-auth-service_disk_1",
    "re2tt_ts-auth-service_loss_1", "re2tt_ts-auth-service_mem_1", "re2tt_ts-order-service_delay_1",
    "re3ob_adservice_f3_1", "re3ob_adservice_f4_1", "re3ob_cartservice_f1_1", "re3ob_emailservice_f2_1",
    "re3ss_carts_f1_1", "re3ss_carts_f3_1", "re3ss_carts_f4_1", "re3ss_front-end_f1_1",
    "re3ss_orders_f1_1", "re3ss_orders_f3_1",
    "re3tt_ts-auth-service_f1_1", "re3tt_ts-auth-service_f2_1", "re3tt_ts-auth-service_f3_1",
    "re3tt_ts-auth-service_f4_1",
]
CAPPED_PAT_ROWS = 10  # the capped step-0 configuration


def git_state():
    """Commit and working-tree state at run time, so a result can be traced to the code that produced it."""
    import subprocess
    def g(*args):
        try:
            return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent,
                                  capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    dirty = g("status", "--porcelain")
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(dirty),
            "dirty_files": [l[2:].strip() for l in dirty.splitlines()][:20]}  # porcelain: 2 status chars, then path


def start_run(label, notes=""):
    """Create results/<timestamp>-<label>/ and return a RunWriter that persists every record as it arrives."""
    d = RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
    d.mkdir(parents=True, exist_ok=False)
    return RunWriter(d, notes=notes)


class RunWriter:
    """Writes each record to disk as it is produced, so a crash mid-run keeps everything up to that point.
    During the run: summary.csv plus *.jsonl (append-only, flushed per record). finalize() converts the
    jsonl files to parquet and writes metadata.json; partial runs stay readable as jsonl."""

    def __init__(self, run_dir, notes=""):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.notes = notes
        self.n = 0
        self.t0 = time.time()
        self.git = git_state()  # recorded at run START; finalize records it again in case it changed
        self._summary_header = False

    def _append_jsonl(self, name, rows):
        if not rows:
            return
        with open(self.dir / f"{name}.jsonl", "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def add(self, case, stage, source, result):
        """Persist one record (one arm on one case) immediately."""
        facts = _case_facts(case)
        cands, syms, meta = result["candidates"], result["symptoms"], result["meta"]
        det = retention_detail(case, source, syms)
        names = list(cands.service)
        rank = names.index(facts["truth"]) + 1 if facts["truth"] in names else None
        answer = meta.get("answer", names[0] if names else None)
        truth_row = cands[cands.service == facts["truth"]]
        a = (meta.get("attempts") or [{}])[-1]
        row = {
            "case": case, "dataset": facts["dataset"], "fault": facts["fault"], "status": facts["status"],
            "truth": facts["truth"], "stage": stage, "arm": source,
            "answer": answer, "correct": bool(answer and is_correct(answer, facts["truth"])),
            "abstained": bool(meta.get("abstained", False)), "confidence": meta.get("confidence", ""),
            "rank_of_truth": rank,
            "reason_for_answer": meta.get("justification", cands.reason.iloc[0] if len(cands) else ""),
            "reason_for_truth": (truth_row.reason.iloc[0] if len(truth_row) else ""),
            "role_of_truth": (truth_row.role.iloc[0] if len(truth_row) and "role" in truth_row else ""),
            "retention_kept": "; ".join(x["item"] for x in det if x["kept"]),
            "retention_missed": "; ".join(x["item"] for x in det if not x["kept"]),
            "retention_score": f"{sum(x['kept'] for x in det)}/{len(det)}",
            "prompt_tokens": meta.get("prompt_tokens"), "eval_tokens": a.get("eval_count"),
            "wall_s": a.get("wall_s"), "gpu_free_before_MB": a.get("gpu_free_before_MB"),
            "parse_error": meta.get("parse_error"), "model": a.get("model", ""),
            "thinking": a.get("thinking", ""), "num_ctx": a.get("num_ctx"),
        }
        pd.DataFrame([row]).to_csv(self.dir / "summary.csv", mode="a", index=False,
                                   header=not self._summary_header)
        self._summary_header = True
        c = cands.copy()
        c.insert(0, "stage", stage)
        c.insert(0, "run_source", source)
        c.insert(0, "truth", facts["truth"])
        self._append_jsonl(stage, c.to_dict("records"))
        if stage == "step1":
            sy = syms.copy()
            sy.insert(0, "run_source", source)
            self._append_jsonl("symptoms", sy.to_dict("records"))
        self._append_jsonl("retention", det)
        self.n += 1
        return row

    def finalize(self, extra_meta=None):
        """Convert the jsonl files to parquet and write metadata.json."""
        for name in ["step1", "step2", "step3", "direct", "symptoms", "retention"]:
            f = self.dir / f"{name}.jsonl"
            if f.exists():
                pd.read_json(f, lines=True).to_parquet(self.dir / f"{name}.parquet", index=False)
                f.unlink()
        gpu = gpu_memory_mb()
        meta = {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": self.notes,
                "git": self.git, "git_at_finalize": git_state(),
                "versions": {"step0": STEP0_VERSION, "step1": STEP1_VERSION, "step2": STEP2_VERSION,
                             "step3": STEP3_VERSION, "direct": DIRECT_VERSION, "thresholds": THRESHOLDS_VERSION},
                "thresholds": THRESHOLDS, "evidence_thresholds": EVIDENCE_THRESHOLDS,
                "step0_params": STEP0_PARAMS, "borderline_margin": BORDERLINE_MARGIN,
                "order_seed": DEFAULT_ORDER_SEED, "answer_reserve": ANSWER_RESERVE,
                "thinking_extra": THINKING_EXTRA, "records": self.n,
                "elapsed_s": round(time.time() - self.t0),
                "gpu_free_MB_at_write": gpu[0] if gpu else None,
                "scoring_exclusions_static": {**EXCLUDE, **BROKEN_LABELS},
                **(extra_meta or {})}
        (self.dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        f = self.dir / "summary.csv"
        # a run that died before its first case has no summary.csv; still write metadata rather than
        # raising on top of the original failure
        return pd.read_csv(f) if f.exists() else pd.DataFrame()


def _case_facts(case, _cache={}):
    if case not in _cache:
        info = case_info(case)
        ev = service_evidence(case, info["system"], info["root_cause_service"], load_inject_time(case))
        _cache[case] = {"truth": info["root_cause_service"], "dataset": info["dataset"], "fault": info["fault"],
                        "status": "clear" if diagnosability_status(ev) == "diagnosable" else "weak",
                        "checks": RETENTION_CHECKS.get(case) or auto_retention_checks(case)}
    return _cache[case]


def retention_detail(case, arm, symptoms, checks=None):
    """One row per expected item: which subtle signals this arm kept and which it lost."""
    checks = _case_facts(case)["checks"] if checks is None else checks
    s = symptoms
    text = (s.signal.fillna("") + " " + s.reason.fillna("")).str.lower() if len(s) else pd.Series(dtype=str)
    rows = []
    for label, svc, kinds, words in checks:
        hit = False
        if len(s):
            same = s.service == svc
            hit = bool(((same & s.kind.isin(kinds)) | (same & text.apply(lambda t: any(w in t for w in words)))).any())
        rows.append({"case": case, "arm": arm, "item": label, "service": svc,
                     "kinds": ",".join(sorted(kinds)), "keywords": ",".join(words), "kept": hit})
    return rows


def write_run(run_dir, records, notes="", extra_meta=None):
    """Batch wrapper kept for convenience: prefer RunWriter.add() so a crash mid-run keeps what ran.
    records: list of {case, stage, source, result}."""
    w = run_dir if isinstance(run_dir, RunWriter) else RunWriter(run_dir, notes=notes)
    for r in records:
        w.add(r["case"], r["stage"], r["source"], r["result"])
    return w.finalize(extra_meta)


def read_case(run_dir, case, arm=None):
    """Read one case's records back: summary rows, candidates, symptoms, retention detail."""
    run_dir = Path(run_dir)
    out = {}
    s = pd.read_csv(run_dir / "summary.csv")
    out["summary"] = s[(s.case == case) & (s.arm == arm if arm else True)]
    for name in ["step1", "step2", "step3", "direct", "symptoms", "retention"]:
        f = run_dir / f"{name}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            col = "run_source" if "run_source" in df.columns else "arm"
            df = df[df.case == case]
            out[name] = df[df[col] == arm] if arm else df
    return out

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

    txt = render_step0(case, include_artifacts=include_artifacts, max_pat_rows=max_pat_rows,
                       order_seed=DEFAULT_ORDER_SEED)  # what a scoring run actually sends
    q, g = count_tokens(txt, "qwen"), count_tokens(txt, "gemma")
    leaks = [s for s in [case, r["fault"] + "_", "root_cause", "ground truth"] if s in txt]
    lines += ["", f"## Step 0 text ({STEP0_VERSION}, service order shuffled with seed {DEFAULT_ORDER_SEED}): qwen {q} tokens (exact), gemma4 ~{g} (approx.) -> "
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
