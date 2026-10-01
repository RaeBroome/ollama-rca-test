"""runlog.py - one timestamped log per test run: what the code does + what the machine does.

Writes logs/<run_id>.jsonl (everything, for analysis) and logs/<run_id>.log (readable, rendered at the end).
Does not import rca_lib; rca_lib stays free of file writes.

    with RunLog("glm-think-on", config={"model": m, "num_ctx": n, "num_predict": 100000}) as log:
        log.event("load", "unloading other models")
        ...
        log.event("generate", "request sent")
        r = rca_lib.ollama_chat(...)
        log.result(done_reason=..., answer=..., expected=...)

Needs: pip install psutil   (nvidia-smi on PATH for GPU numbers; missing -> GPU fields are null)
"""
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import psutil

GPU_IDLE_PCT = 10        # warn when GPU util is below this while generating
VRAM_FULL_FRAC = 0.97    # warn when VRAM used is above this share of total
MAX_SAMPLE_LINES = 12    # per phase in the readable view; everything stays in the jsonl


def _hhmmss(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


def _mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def _gpu():
    """(util_pct, used_mb, total_mb) or None. nvidia-smi is the source of truth for VRAM."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return tuple(int(x) for x in out.strip().splitlines()[0].split(","))
    except Exception:
        return None


def _ollama_rss_gb():
    """RAM held by Ollama processes (the runner holds the CPU-side layers)."""
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            if "ollama" in (p.info["name"] or "").lower():
                total += p.info["memory_info"].rss
        except Exception:
            pass
    return round(total / 1e9, 2)


def _ollama_ps():
    """Raw /api/ps, logged as a HINT only: rca_lib documents its CPU/GPU split as unreliable."""
    h = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    h = h if h.startswith(("http://", "https://")) else "http://" + h
    try:
        ps = json.loads(urllib.request.urlopen(h.rstrip("/") + "/api/ps", timeout=5).read())
        return [{"name": m.get("name"), "size_gb": round((m.get("size") or 0) / 1e9, 1),
                 "size_vram_gb": round((m.get("size_vram") or 0) / 1e9, 1),
                 "ctx": m.get("context_length")} for m in ps.get("models", []) or []]
    except Exception:
        return None


class RunLog:
    def __init__(self, label, config=None, log_dir="logs", sample_every=5.0, echo=True):
        # echo=True prints events (and one sample per heartbeat) to the screen as they happen.
        self.echo = echo
        self.t0 = time.time()
        self.run_id = f"{time.strftime('%Y-%m-%d_%H%M%S')}_{label}"
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.dir / f"{self.run_id}.jsonl"
        self.config = config or {}
        self.sample_every = sample_every
        self.phase = "setup"
        self.progress = {}          # streaming code can set e.g. log.progress["tokens"] = n
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._write({"kind": "header", "config": self.config, "git": self._git(),
                     "ollama_server_log": str(Path(os.environ.get("LOCALAPPDATA", "~")) / "Ollama" / "server.log")})

    # ---- writing
    def _write(self, rec):
        rec = {"run_id": self.run_id, "ts": time.time(), "t": round(time.time() - self.t0, 2), **rec}
        with self._lock, open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    @staticmethod
    def _git():
        try:
            g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, timeout=10).stdout.strip()
            return {"commit": g("rev-parse", "--short", "HEAD"), "dirty": bool(g("status", "--porcelain"))}
        except Exception:
            return {}

    def event(self, phase, msg, **kv):
        """A code-side milestone. Sets the current phase, which samples inherit."""
        self.phase = phase
        self._write({"kind": "event", "phase": phase, "msg": msg, **kv})
        if self.echo:
            extra = " " + " ".join(f"{k}={v}" for k, v in kv.items()) if kv else ""
            print(f"  {_mmss(time.time() - self.t0):>5}  [{phase}] {msg}{extra}", flush=True)

    def result(self, **kv):
        self._write({"kind": "result", **kv})

    # ---- sampler
    def _sample_once(self):
        gpu = _gpu()
        vm = psutil.virtual_memory()
        rec = {"kind": "sample", "phase": self.phase,
               "cpu_pct": psutil.cpu_percent(interval=None), "ram_used_gb": round(vm.used / 1e9, 1),
               "ollama_ram_gb": _ollama_rss_gb(),
               "gpu_pct": gpu[0] if gpu else None, "vram_used_mb": gpu[1] if gpu else None,
               "vram_total_mb": gpu[2] if gpu else None,
               "ps_hint": _ollama_ps(), **{f"p_{k}": v for k, v in self.progress.items()}}
        self._write(rec)
        if self.echo:
            heart = f"CPU {rec['cpu_pct']:.0f}%  RAM {rec['ram_used_gb']}G"
            if rec["vram_used_mb"] is not None:
                heart += f"  VRAM {rec['vram_used_mb'] / 1024:.1f}G  GPU {rec['gpu_pct']}%"
            if "p_tokens" in rec:
                heart += f"  tok {rec['p_tokens']}"
            print(f"  {_mmss(time.time() - self.t0):>5}  [{self.phase}] {heart}", flush=True)

    def _loop(self):
        psutil.cpu_percent(interval=None)  # prime: first call always returns 0
        while not self._stop.wait(self.sample_every):
            try:
                self._sample_once()
            except Exception as e:  # the sampler must never kill the test
                self._write({"kind": "event", "phase": self.phase, "msg": f"sampler error: {e}"})

    def start_sampler(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop_sampler(self):
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=15)
            self._thread = None

    # ---- context manager: sampler on for the whole run, readable log always written
    def __enter__(self):
        self.event("setup", "run started")
        self.start_sampler()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc:
            self.event("error", f"{exc_type.__name__}: {exc}")
        self.stop_sampler()
        self.event("done", "run finished")
        print(f"log: {self.render()}")
        return False

    # ---- readable view
    def render(self):
        recs = [json.loads(l) for l in self.jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        head = next(r for r in recs if r["kind"] == "header")
        L = ["=" * 60, f"RUN {self.run_id}", "=" * 60]
        for k, v in head["config"].items():
            L.append(f"{k:<13}{v}")
        g = head.get("git") or {}
        if g:
            L.append(f"{'git':<13}{g.get('commit')}{' (dirty)' if g.get('dirty') else ''}")
        L.append(f"{'server log':<13}{head.get('ollama_server_log')}")

        # group everything after the header by phase, in order of first appearance
        order, by = [], {}
        for r in recs:
            if r["kind"] in ("event", "sample"):
                if r["phase"] not in by:
                    order.append(r["phase"])
                    by[r["phase"]] = []
                by[r["phase"]].append(r)
        for ph in order:
            L += ["", f"-- PHASE: {ph} " + "-" * max(4, 46 - len(ph))]
            items = by[ph]
            samples = [r for r in items if r["kind"] == "sample"]
            shown = self._pick(samples)
            for r in items:
                if r["kind"] == "event":
                    extra = "  " + " ".join(f"{k}={v}" for k, v in r.items()
                                            if k not in ("run_id", "ts", "t", "kind", "phase", "msg"))
                    L.append(f"{_hhmmss(r['ts'])}  {r['msg']}{extra.rstrip()}")
                elif r in shown:
                    L.append(self._sample_line(r))
            hidden = len(samples) - len(shown)
            if hidden > 0:
                L.append(f"         ... {hidden} more samples hidden (full detail in {self.jsonl.name}) ...")
            L += self._warnings(ph, samples)

        res = [r for r in recs if r["kind"] == "result"]
        if res:
            L += ["", "-- RESULT " + "-" * 49]
            for k, v in res[-1].items():
                if k not in ("run_id", "ts", "t", "kind"):
                    L.append(f"{k:<13}{v}")
        L.append(f"{'elapsed':<13}{_mmss(recs[-1]['t'])}")
        out = self.dir / f"{self.run_id}.log"
        out.write_text("\n".join(L) + "\n", encoding="utf-8")
        return str(out)

    @staticmethod
    def _pick(samples):
        if len(samples) <= MAX_SAMPLE_LINES:
            return samples
        step = max(1, len(samples) // (MAX_SAMPLE_LINES - 3))
        keep = samples[:2] + samples[2:-1:step] + samples[-1:]
        return keep[:MAX_SAMPLE_LINES]

    @staticmethod
    def _sample_line(r):
        bits = [f"t+{_mmss(r['t'])}"]
        if "p_tokens" in r:
            bits.append(f"tok {r['p_tokens']:<6}")
        if r.get("p_tok_s") is not None:
            bits.append(f"{r['p_tok_s']} t/s")
        bits.append(f"CPU {r['cpu_pct']:.0f}%  RAM {r['ram_used_gb']}G (ollama {r['ollama_ram_gb']}G)")
        if r["vram_used_mb"] is not None:
            bits.append(f"VRAM {r['vram_used_mb'] / 1024:.1f}G  GPU {r['gpu_pct']}%")
        return "  " + "  ".join(bits)

    @staticmethod
    def _warnings(phase, samples):
        w = []
        g = [s for s in samples if s["gpu_pct"] is not None]
        if g and phase == "generate":
            idle = sum(s["gpu_pct"] < GPU_IDLE_PCT for s in g)
            if idle:
                w.append(f"  !! GPU under {GPU_IDLE_PCT}% in {idle} of {len(g)} samples while generating")
        full = [s for s in g if s["vram_used_mb"] / s["vram_total_mb"] > VRAM_FULL_FRAC]
        if full:
            w.append(f"  !! VRAM over {int(VRAM_FULL_FRAC * 100)}% full in {len(full)} of {len(g)} samples")
        return w
