"""
E1 -- Energy profile of the fleet and the routers.

Implements the protocol in paper Sec. 4.5 exactly:
  idle subtraction, loop-N (never single-query), warm-up, cool-down,
  randomised model order across trials, wrap-around correction, mean +/- 95% CI.
Every model and router is measured at identical BATCH / SEQ_LEN / DTYPE / THREADS.

Usage:
  python e1_profile.py --smoke            # N=20, 1 trial, no cooldown: pipeline check
  python e1_profile.py                    # full: N=1000, 5 trials, 60 s cooldown
Output: experiments/results/e1_energy.csv  (+ per-trial e1_trials.csv)
"""
import os, sys, time, random, argparse, statistics, csv, re
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "energy"))
from models import (TEACHER, STUDENTS, BASELINE, NEURAL_ROUTER, SEQ_LEN, BATCH,
                    DTYPE, THREADS, MODELS_DIR, DATA_DIR, RESULTS)
os.environ["HF_HOME"] = os.path.abspath(MODELS_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, numpy as np
from meter import Meter, measure_idle
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset

torch.set_num_threads(THREADS)
torch.set_grad_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--trials", type=int, default=None)
ap.add_argument("--cooldown", type=float, default=None)
ap.add_argument("--idle-seconds", type=float, default=None)
a = ap.parse_args()
N        = a.n        or (20  if a.smoke else 1000)
TRIALS   = a.trials   or (1   if a.smoke else 5)
COOLDOWN = a.cooldown if a.cooldown is not None else (0 if a.smoke else 60.0)
IDLE_S   = a.idle_seconds or (5 if a.smoke else 60.0)
WARMUP   = 5 if a.smoke else 100

# ------------------------------------------------------------------ queries
# Fixed query set for every configuration: SST-2 validation sentences.
ds = load_dataset("nyu-mll/glue", "sst2", cache_dir=DATA_DIR)["validation"]
QUERIES = [r["sentence"] for r in ds.select(range(min(N, len(ds))))]
print(f"[E1] N={N} trials={TRIALS} cooldown={COOLDOWN}s warmup={WARMUP} seq={SEQ_LEN} "
      f"batch={BATCH} dtype={DTYPE} threads={THREADS}", flush=True)

# ------------------------------------------------------------------ configs
# (label, kind, checkpoint)
CONFIGS = [("teacher", "cls", TEACHER["sst2"]),
           ("teacher_int8", "cls_int8", TEACHER["sst2"])]   # baseline (v): quantised teacher
CONFIGS += [(k, "enc", v) for k, v in STUDENTS.items()]
CONFIGS += [("distilbert_baseline", "enc", BASELINE)]
CONFIGS += [("router_heuristic", "heur", None),
            ("router_linear",    "lin",  None),
            ("router_neural",    "enc",  NEURAL_ROUTER)]

# ---------------------------------------------------------- router features
_WORD = re.compile(r"\w+")
_CUES = ("why", "how", "because", "although", "whereas", "unless", "if", "not",
         "never", "but", "however", "prove", "explain", "compare")
def heuristic_features(text):
    words = _WORD.findall(text.lower()); nw = max(len(words), 1)
    return np.array([
        len(text), nw,
        sum(c.isdigit() for c in text) / max(len(text), 1),
        text.count(",") + text.count(";") + text.count(":"),
        sum(w in _CUES for w in words),
        sum(len(w) for w in words) / nw,
        len(set(words)) / nw,
    ], dtype=np.float32)
_W = np.random.RandomState(0).randn(7).astype(np.float32)   # placeholder weights;
                                                              # cost is what matters here

def make_runner(kind, ckpt):
    if kind == "heur":
        return lambda q: float(heuristic_features(q) @ _W)
    if kind == "lin":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression().fit(np.random.RandomState(0).randn(64, 7),
                                       np.random.RandomState(1).randint(0, 2, 64))
        return lambda q: clf.predict_proba(heuristic_features(q).reshape(1, -1))[0, 1]
    tok = AutoTokenizer.from_pretrained(ckpt)
    cls = AutoModelForSequenceClassification if kind.startswith("cls") else AutoModel
    m = cls.from_pretrained(ckpt, dtype=getattr(torch, DTYPE)).eval()
    if kind == "cls_int8":   # the one deliberate departure from fixed precision
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    def run(q):
        x = tok(q, padding="max_length", truncation=True, max_length=SEQ_LEN,
                return_tensors="pt")
        return m(**x)
    return run, m

# ------------------------------------------------------------------ measure
os.makedirs(RESULTS, exist_ok=True)
print(f"[E1] measuring idle for {IDLE_S}s (close everything else) ...", flush=True)
idle_domains, idle_w = measure_idle(IDLE_S, use_gpu=False)
print(f"[E1] idle = {idle_w:.2f} W  " + ", ".join(f"{k}={v:.2f}" for k, v in idle_domains.items()),
      flush=True)

rows = []          # per trial
for t in range(TRIALS):
    order = CONFIGS[:]; random.Random(t).shuffle(order)      # randomised per trial
    print(f"\n[E1] trial {t+1}/{TRIALS} order: {[c[0] for c in order]}", flush=True)
    for label, kind, ckpt in order:
        r = make_runner(kind, ckpt)
        run = r[0] if isinstance(r, tuple) else r
        nparams = sum(p.numel() for p in r[1].parameters()) if isinstance(r, tuple) else 0
        for q in QUERIES[:WARMUP]: run(q)                     # warm-up, discarded
        m = Meter(use_gpu=False)   # CPU-only study: package + DRAM
        with m:
            for i in range(N):
                run(QUERIES[i % len(QUERIES)])
                if i % 25 == 0: m.sample()
        gross = m.total_joules; net = gross - idle_w * m.seconds
        row = dict(trial=t, label=label, kind=kind, ckpt=ckpt or "", params=nparams,
                   n=N, seconds=m.seconds, gross_J=gross, idle_J=idle_w * m.seconds,
                   net_J=net, net_mJ_per_query=1000 * net / N,
                   ms_per_query=1000 * m.seconds / N,
                   **{f"{k}_J": v for k, v in m.joules.items()})
        rows.append(row)
        print(f"  {label:20s} {row['net_mJ_per_query']:9.3f} mJ/q  {row['ms_per_query']:7.2f} ms/q  "
              f"({nparams/1e6:.1f}M)", flush=True)
        if isinstance(r, tuple): del r
        if COOLDOWN and not (t == TRIALS - 1 and (label, kind, ckpt) == order[-1]):
            time.sleep(COOLDOWN)

# ------------------------------------------------------------------ write
with open(f"{RESULTS}/e1_trials.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

summary = []
for label, kind, ckpt in CONFIGS:
    xs = [r["net_mJ_per_query"] for r in rows if r["label"] == label]
    ms = [r["ms_per_query"]     for r in rows if r["label"] == label]
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    summary.append(dict(label=label, kind=kind, ckpt=ckpt or "",
                        params=next(r["params"] for r in rows if r["label"] == label),
                        mJ_per_query=statistics.mean(xs), sd=sd,
                        ci95=1.96 * sd / len(xs) ** 0.5, ms_per_query=statistics.mean(ms),
                        trials=len(xs), n=N, idle_W=idle_w,
                        seq_len=SEQ_LEN, batch=BATCH, dtype=DTYPE, threads=THREADS))
with open(f"{RESULTS}/e1_energy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

s1 = next(s for s in summary if s["label"] == "s1")["mJ_per_query"]
print("\n[E1] SUMMARY  (net of idle; mean ± 95% CI over trials)")
print(f"  {'config':20s} {'params':>8s} {'mJ/query':>10s} {'±CI':>7s} {'ms/q':>7s} {'ρ vs s1':>8s}")
for s in summary:
    print(f"  {s['label']:20s} {s['params']/1e6:7.1f}M {s['mJ_per_query']:10.3f} {s['ci95']:7.3f} "
          f"{s['ms_per_query']:7.2f} {s['mJ_per_query']/s1:8.2f}")
print(f"\n[E1] wrote {RESULTS}/e1_energy.csv and e1_trials.csv")
