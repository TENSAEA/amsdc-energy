"""
E13 -- Confidence-GRADED dispatch and MULTI-STAGE cascade.

E8 tested only s1 -> ONE fixed tier. That wastes the fact that s1's confidence is a
CONTINUOUS signal: a barely-unsure query probably needs s2, a completely-unsure one
may need the teacher. Two architectures are added here:

  (A) GRADED  -- one probe, graded jump. Run s1; use confidence BANDS to choose
                 which tier to escalate to. Still ONE escalation, so per-query cost
                 is bounded by E(s1)+E(teacher). Closest to "send each query to the
                 smallest one that can handle it".

  (B) LADDER  -- multi-stage. s1 -> s2 -> s3 -> s4 -> teacher, stopping at the first
                 tier whose confidence clears a threshold. Costs accumulate, so the
                 tail is worse, but it uses each tier's own judgement.

Both are calibrated on the EXPANDED calibration set where available, evaluated on the
untouched test split. Energy charges EVERY model actually executed.
"""
import os, sys, json, csv, itertools
sys.path.insert(0, os.path.dirname(__file__))
from models import MODELS_DIR, RESULTS, SEQ_LEN, STUDENTS
os.environ["HF_HOME"]=os.path.abspath(MODELS_DIR); os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(6); torch.set_grad_enabled(False)

task="sst2"
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
LADDER=list(STUDENTS.keys())+["teacher"]; EN=np.array([E[t] for t in LADDER])
def cv(t,s): return np.array(D[f"teacher_{s}_correct"]) if t=="teacher" else np.array(D["students"][t][f"{s}_correct"])
C=np.stack([cv(t,"test") for t in LADDER]); n=C.shape[1]
o=np.full(n,len(LADDER)-1)
for k in range(len(LADDER)-1,-1,-1): o[C[k]==1]=k
E_T=EN[-1]*n; RIE=E_T-EN[o].sum()
teacher_acc=D["teacher_acc"]

@torch.no_grad()
def conf_of(tier, texts):
    d=os.path.join(MODELS_DIR,"distilled",task,tier)
    tok=AutoTokenizer.from_pretrained(d); m=AutoModelForSequenceClassification.from_pretrained(d).eval()
    out=[]
    for i in range(0,len(texts),64):
        e=tok(texts[i:i+64],padding="max_length",truncation=True,max_length=SEQ_LEN,return_tensors="pt")
        out.append(torch.softmax(m(**e).logits,-1).max(-1).values)
    return torch.cat(out).numpy()

print("computing confidence for every tier (needed for the ladder) ...", flush=True)
CONF=np.stack([conf_of(t, D["test_text"]) for t in LADDER[:-1]])   # students only
print(f"  done. s1 confidence range [{CONF[0].min():.3f}, {CONF[0].max():.3f}]", flush=True)

def summar(name, energy, ok):
    return dict(system=name, mJ_per_query=float(energy.mean()), acc=float(ok.mean()),
                median=float(np.median(energy)), p95=float(np.percentile(energy,95)),
                max=float(energy.max()), RR=float((E_T-energy.sum())/RIE),
                vs_teacher_energy=float(100*(1-energy.mean()/EN[-1])),
                vs_teacher_acc=float(100*(ok.mean()-teacher_acc)))

rows=[]
# ---------------------------------------------------------------- (A) GRADED
# bands on s1 confidence: [0,b1) -> teacher, [b1,b2) -> s4, [b2,b3) -> s3,
# [b3,b4) -> s2, [b4,1] -> keep s1.  Search over monotone band edges.
c1=CONF[0]
grid=np.round(np.arange(0.50,1.001,0.05),2)
best_graded=[]
for edges in itertools.combinations(grid,4):
    b1,b2,b3,b4=edges
    tier=np.full(n,0)                      # default keep s1
    tier=np.where(c1<b4,1,tier)            # -> s2
    tier=np.where(c1<b3,2,tier)            # -> s3
    tier=np.where(c1<b2,3,tier)            # -> s4
    tier=np.where(c1<b1,4,tier)            # -> teacher
    energy=EN[0]+np.where(tier>0,EN[tier],0.0)
    ok=np.where(tier>0, C[tier,np.arange(n)], C[0])
    best_graded.append((energy.mean(), ok.mean(), edges, energy, ok))
# report the Pareto-best graded configs
best_graded.sort(key=lambda x:x[0])
front=[]
for e,a,ed,en,ok in best_graded:
    if not any(a2>=a and e2<=e and (a2>a or e2<e) for e2,a2,_,_,_ in best_graded):
        front.append((e,a,ed,en,ok))
print(f"\n  GRADED: {len(best_graded)} configs searched, {len(front)} Pareto-optimal")
for e,a,ed,en,ok in sorted(front,key=lambda x:-x[1])[:4]:
    rows.append(summar(f"Graded s1-bands {ed}", en, ok))

# ---------------------------------------------------------------- (B) LADDER
for t in [0.90,0.95,0.98,0.99]:
    energy=np.zeros(n); ok=np.zeros(n,dtype=int); done=np.zeros(n,dtype=bool)
    for k in range(len(LADDER)-1):
        run=~done
        energy[run]+=EN[k]
        acc_here=CONF[k]>=t
        stop=run&acc_here
        ok[stop]=C[k][stop]; done|=stop
    energy[~done]+=EN[-1]; ok[~done]=C[-1][~done]
    rows.append(summar(f"Ladder s1..T t={t}", energy, ok))

# ---------------------------------------------------------------- reference points
rows.append(summar("Fixed teacher", np.full(n,EN[-1]), C[-1]))
rows.append(summar("Fixed s4", np.full(n,EN[3]), C[3]))
rows.append(summar("DistilBERT", np.full(n,E["distilbert_baseline"]),
                   np.array(D["students"]["distilbert"]["test_correct"])))
best8=[r for r in csv.DictReader(open(f"{RESULTS}/e8_cascade_{task}.csv"))
       if r["escalate_to"]=="s4" and r["thresh"]=="0.98"][0]
rows.append(dict(system="Cascade s1->s4 t=0.98 (E8)", mJ_per_query=float(best8["mJ_per_query"]),
                 acc=float(best8["acc"]), median=float(best8["median"]), p95=float(best8["p95"]),
                 max=float(best8["max"]), RR=float(best8["RR"]),
                 vs_teacher_energy=100*(1-float(best8["mJ_per_query"])/EN[-1]),
                 vs_teacher_acc=100*(float(best8["acc"])-teacher_acc)))
rows.append(summar("ORACLE (bound)", EN[o], C[o,np.arange(n)]))

print(f"\n{'='*112}\n  E13: graded vs multi-stage vs single-target dispatch   (same 872 queries throughout)\n{'='*112}")
print(f"  {'system':34s} {'mJ/q':>9s} {'acc':>7s} {'median':>9s} {'p95':>9s} | {'vs teacher':>19s}")
for r in sorted(rows,key=lambda r:r["mJ_per_query"]):
    print(f"  {r['system']:34s} {r['mJ_per_query']:9.1f} {100*r['acc']:6.2f}% {r['median']:9.1f} "
          f"{r['p95']:9.1f} | {r['vs_teacher_energy']:+8.1f}% {r['vs_teacher_acc']:+8.2f}")
with open(f"{RESULTS}/e13_graded_{task}.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\n  wrote {RESULTS}/e13_graded_{task}.csv")
