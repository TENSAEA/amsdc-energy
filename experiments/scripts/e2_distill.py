"""
E2 -- Distil the student fleet from the task-fine-tuned teacher.

Implements paper Eq. (3): L = a*T^2*KL(teacher||student) + (1-a)*CE(labels).
Every student sees the COMPLETE task training set -- no partitioning. This is
the "full data" constraint of Sec. 3.3 and the precondition for Prediction 1.

Usage:  python e2_distill.py --task sst2
        python e2_distill.py --task rte
Output: experiments/models/distilled/<task>/<tier>/  + e2_distill_<task>.json
"""
import os, sys, json, time, argparse, random
sys.path.insert(0, os.path.dirname(__file__))
from models import TEACHER, STUDENTS, BASELINE, SEQ_LEN, MODELS_DIR, DATA_DIR, RESULTS
os.environ["HF_HOME"] = os.path.abspath(MODELS_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["sst2", "rte"])
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--alpha", type=float, default=0.7)     # weight on distillation loss
ap.add_argument("--temp", type=float, default=4.0)      # softmax temperature
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--max-train", type=int, default=None,
                help="subsample training set (CPU-only compute budget); test split untouched")
a = ap.parse_args()
EPOCHS = a.epochs or (1 if a.smoke else (3 if a.task == "sst2" else 8))

torch.set_num_threads(a.threads)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

TEXT = {"sst2": ("sentence", None), "rte": ("sentence1", "sentence2")}[a.task]
OUT  = os.path.join(MODELS_DIR, "distilled", a.task)
os.makedirs(OUT, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)

print(f"[E2/{a.task}] epochs={EPOCHS} alpha={a.alpha} T={a.temp} lr={a.lr} bs={a.bs}", flush=True)

# ---------------------------------------------------------------- data
ds = load_dataset("nyu-mll/glue", a.task, cache_dir=DATA_DIR)
train_full, test = ds["train"], ds["validation"]
if a.smoke:
    train_full = train_full.select(range(min(200, len(train_full))))
    test = test.select(range(min(100, len(test))))
elif a.max_train and len(train_full) > a.max_train:
    # Deterministic subsample of TRAIN only. Test split is never touched.
    keep = list(range(len(train_full))); random.Random(SEED).shuffle(keep)
    train_full = train_full.select(sorted(keep[:a.max_train]))
    print(f"[E2/{a.task}] train subsampled to {len(train_full)} (CPU budget)", flush=True)

# 10% of train held out for router calibration (paper Sec. 4.4)
n = len(train_full); idx = list(range(n)); random.Random(SEED).shuffle(idx)
n_cal = max(1, int(0.10 * n))
cal_idx, tr_idx = idx[:n_cal], idx[n_cal:]
train, calib = train_full.select(tr_idx), train_full.select(cal_idx)
print(f"[E2/{a.task}] train={len(train)} calib={len(calib)} test={len(test)}", flush=True)

def encode(tok, split):
    args = ([r[TEXT[0]] for r in split],) if TEXT[1] is None else \
           ([r[TEXT[0]] for r in split], [r[TEXT[1]] for r in split])
    enc = tok(*args, padding="max_length", truncation=True,
              max_length=SEQ_LEN, return_tensors="pt")
    y = torch.tensor([r["label"] for r in split])
    return enc["input_ids"], enc["attention_mask"], y

# ------------------------------------------------- teacher soft targets (cached)
tname = TEACHER[a.task]
print(f"[E2/{a.task}] teacher: {tname}", flush=True)
ttok = AutoTokenizer.from_pretrained(tname)
teacher = AutoModelForSequenceClassification.from_pretrained(tname).eval()
NUM_LABELS = teacher.config.num_labels

@torch.no_grad()
def teacher_logits(split, label):
    ii, am, y = encode(ttok, split)
    outs = []
    t0 = time.time()
    for i in range(0, len(ii), 64):
        outs.append(teacher(input_ids=ii[i:i+64], attention_mask=am[i:i+64]).logits)
        if i % 1280 == 0:
            print(f"    {label} {i}/{len(ii)}  ({time.time()-t0:.0f}s)", flush=True)
    return torch.cat(outs), y

print(f"[E2/{a.task}] caching teacher logits ...", flush=True)
tl_train, y_train = teacher_logits(train, "train")
tl_test,  y_test  = teacher_logits(test,  "test")
tl_cal,   y_cal   = teacher_logits(calib, "calib")
teacher_acc = (tl_test.argmax(-1) == y_test).float().mean().item()
print(f"[E2/{a.task}] TEACHER test acc = {teacher_acc:.4f}", flush=True)
del teacher

# ---------------------------------------------------------------- distillation
def evaluate(model, ii, am, y):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(ii), 64):
            preds.append(model(input_ids=ii[i:i+64], attention_mask=am[i:i+64]).logits.argmax(-1))
    p = torch.cat(preds)
    return (p == y).float().mean().item(), p

results = {"task": a.task, "max_train": a.max_train, "teacher": tname, "teacher_acc": teacher_acc,
           "epochs": EPOCHS, "alpha": a.alpha, "temp": a.temp, "lr": a.lr,
           "bs": a.bs, "seed": SEED, "n_train": len(train), "n_calib": len(calib),
           "n_test": len(test), "students": {}}

fleet = dict(STUDENTS); fleet["distilbert"] = BASELINE   # baseline distilled identically
for tier, ckpt in fleet.items():
    print(f"\n[E2/{a.task}] ==== {tier}: {ckpt} ====", flush=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=NUM_LABELS)
    ii, am, y = encode(tok, train)
    ds_tr = TensorDataset(ii, am, y, tl_train)
    dl = DataLoader(ds_tr, batch_size=a.bs, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    t0 = time.time(); step = 0
    for ep in range(EPOCHS):
        model.train()
        for bi, bam, by, btl in dl:
            opt.zero_grad()
            logits = model(input_ids=bi, attention_mask=bam).logits
            # Eq. (3)
            kd = F.kl_div(F.log_softmax(logits / a.temp, -1),
                          F.softmax(btl / a.temp, -1),
                          reduction="batchmean") * (a.temp ** 2)
            ce = F.cross_entropy(logits, by)
            (a.alpha * kd + (1 - a.alpha) * ce).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(f"    ep{ep+1} step {step}/{steps} kd={kd.item():.4f} ce={ce.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        ii_t, am_t, _ = encode(tok, test)
        acc, _ = evaluate(model, ii_t, am_t, y_test)
        print(f"    ep{ep+1} test acc = {acc:.4f}", flush=True)

    # final: save per-example correctness on test AND calibration (needed for oracle + router)
    ii_t, am_t, _ = encode(tok, test);  acc_t, pred_t = evaluate(model, ii_t, am_t, y_test)
    ii_c, am_c, _ = encode(tok, calib); acc_c, pred_c = evaluate(model, ii_c, am_c, y_cal)
    d = os.path.join(OUT, tier); model.save_pretrained(d); tok.save_pretrained(d)
    results["students"][tier] = {
        "ckpt": ckpt, "dir": d, "test_acc": acc_t, "calib_acc": acc_c,
        "params": sum(p.numel() for p in model.parameters()),
        "train_seconds": time.time() - t0,
        "test_correct":  (pred_t == y_test).int().tolist(),
        "calib_correct": (pred_c == y_cal).int().tolist(),
    }
    print(f"  {tier}: test={acc_t:.4f} calib={acc_c:.4f} "
          f"(teacher {teacher_acc:.4f})  [{time.time()-t0:.0f}s]", flush=True)

# teacher correctness too (it is the top tier of the fleet)
results["teacher_test_correct"]  = (tl_test.argmax(-1) == y_test).int().tolist()
results["teacher_calib_correct"] = (tl_cal.argmax(-1) == y_cal).int().tolist()
results["calib_idx"], results["test_labels"] = cal_idx, y_test.tolist()

# raw query text -- needed by the router feature extractor in E4
def raw(split):
    if TEXT[1] is None:
        return [r[TEXT[0]] for r in split]
    return [r[TEXT[0]] + " " + r[TEXT[1]] for r in split]
results["test_text"]  = raw(test)
results["calib_text"] = raw(calib)

with open(f"{RESULTS}/e2_distill_{a.task}.json", "w") as f:
    json.dump(results, f)
print(f"\n[E2/{a.task}] SUMMARY")
print(f"  {'tier':12s} {'params':>10s} {'test acc':>9s} {'vs teacher':>11s}")
print(f"  {'teacher':12s} {335.1:9.1f}M {teacher_acc:9.4f} {0.0:+10.4f}")
for t, r in results["students"].items():
    print(f"  {t:12s} {r['params']/1e6:9.1f}M {r['test_acc']:9.4f} {r['test_acc']-teacher_acc:+10.4f}")
print(f"[E2/{a.task}] wrote {RESULTS}/e2_distill_{a.task}.json")
