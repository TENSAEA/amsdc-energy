"""
E8 -- Confidence-gated capacity cascade.

E7 established that the capacity signal lives in the small model's output, not in
the query surface. The deployable system this implies runs s1 first and escalates
on low confidence. Crucially, s1's inference is CHARGED as the routing cost:

    E(q) = E(s1) + 1[conf(q) < t] * E(tier_k)

The confidence threshold t is calibrated on the CALIBRATION split and evaluated on
TEST. We sweep t and the escalation target k, and report the full operating curve
plus the per-query energy distribution (median / p95 / max), because a bounded tail
is the property the paper argues cascades lack.
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from models import MODELS_DIR, RESULTS, SEQ_LEN, STUDENTS
os.environ["HF_HOME"] = os.path.abspath(MODELS_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(6); torch.set_grad_enabled(False)

task = "sst2"
D  = json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
E  = {r["label"]: float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
LADDER = list(STUDENTS.keys()) + ["teacher"]
EN = np.array([E[t] for t in LADDER])

def cv(tier, split):
    if tier == "teacher": return np.array(D[f"teacher_{split}_correct"])
    return np.array(D["students"][tier][f"{split}_correct"])
C_te = np.stack([cv(t,"test")  for t in LADDER])
C_ca = np.stack([cv(t,"calib") for t in LADDER])
def orc(C):
    o=np.full(C.shape[1],len(LADDER)-1)
    for k in range(len(LADDER)-1,-1,-1): o[C[k]==1]=k
    return o
o_te = orc(C_te)
E_T  = EN[-1]*C_te.shape[1]; RIE = E_T - EN[o_te].sum()

@torch.no_grad()
def conf_of(tier, texts):
    d=os.path.join(MODELS_DIR,"distilled",task,tier)
    tok=AutoTokenizer.from_pretrained(d); m=AutoModelForSequenceClassification.from_pretrained(d).eval()
    o=[]
    for i in range(0,len(texts),64):
        e=tok(texts[i:i+64],padding="max_length",truncation=True,max_length=SEQ_LEN,return_tensors="pt")
        o.append(torch.softmax(m(**e).logits,-1).max(-1).values)
    return torch.cat(o).numpy()

print("computing s1 confidence on calib + test ...", flush=True)
conf_ca, conf_te = conf_of("s1", D["calib_text"]), conf_of("s1", D["test_text"])

def run(conf, C, t, k):
    esc = conf < t
    e = EN[0] + esc*EN[k]
    ok = np.where(esc, C[k], C[0])
    return e, ok

rows=[]
for k in range(1,len(LADDER)):
    for t in np.round(np.arange(0.50,1.001,0.01),2):
        e_ca, ok_ca = run(conf_ca, C_ca, t, k)
        e_te, ok_te = run(conf_te, C_te, t, k)
        rows.append(dict(escalate_to=LADDER[k], thresh=float(t),
                         calib_acc=float(ok_ca.mean()), acc=float(ok_te.mean()),
                         mJ_per_query=float(e_te.mean()), median=float(np.median(e_te)),
                         p95=float(np.percentile(e_te,95)), max=float(e_te.max()),
                         escalated_pct=float(100*(conf_te<t).mean()),
                         RR=float((E_T-e_te.sum())/RIE)))

with open(f"{RESULTS}/e8_cascade_{task}.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# Operating points selected on CALIBRATION accuracy, reported on TEST (no test leakage)
print(f"\n{'='*94}\n  E8: confidence-gated cascade  (s1 always runs; its energy is the routing cost)\n{'='*94}")
print(f"  {'target':>7s} {'calib floor':>11s} {'thresh':>7s} {'escal%':>7s} {'mJ/query':>10s} "
      f"{'test acc':>9s} {'median':>9s} {'p95':>10s} {'max':>10s} {'RR':>7s}")
for floor in [0.90, 0.92, 0.94, 0.95]:
    cands=[r for r in rows if r["calib_acc"]>=floor]
    if not cands: print(f"  floor {floor:.2f}: unreachable"); continue
    b=min(cands,key=lambda r:r["mJ_per_query"])
    print(f"  {b['escalate_to']:>7s} {floor:11.2f} {b['thresh']:7.2f} {b['escalated_pct']:7.1f} "
          f"{b['mJ_per_query']:10.2f} {b['acc']:9.4f} {b['median']:9.2f} {b['p95']:10.2f} "
          f"{b['max']:10.2f} {b['RR']:7.3f}")
print(f"\n  reference:  teacher {EN[-1]:.2f} mJ @ {C_te[-1].mean():.4f}   "
      f"oracle {EN[o_te].mean():.2f} mJ @ {C_te[o_te,np.arange(len(o_te))].mean():.4f}")
print(f"  wrote {RESULTS}/e8_cascade_{task}.csv")
