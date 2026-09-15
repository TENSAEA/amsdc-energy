"""
E15 -- Systematic search for ANY deployable configuration reaching teacher accuracy.

Goal: replace the teacher. That requires acc >= teacher at lower energy.
E14 found none among simple votes/agreement. This does two things:

 1. Tests the DIVERSITY hypothesis directly. DistilBERT has a DIFFERENT pre-training
    lineage from the Turc miniatures. If error correlation is the barrier, DistilBERT
    should decorrelate from them more than they do from each other -- and mixed-lineage
    ensembles should outperform same-lineage ones at equal energy.

 2. Exhaustively searches acceptance rules: k-of-n agreement, confidence gates, and
    combinations, escalating to the teacher when the rule abstains. Reports every
    configuration reaching teacher accuracy at lower energy.
"""
import os, sys, json, csv, itertools
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS
import numpy as np

task="sst2"
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
S=json.load(open(f"{RESULTS}/e7_signal_{task}.json"))
MEM=["s1","s2","s3","s4","distilbert"]
EN={**{t:E[t] for t in STUDENTS}, "distilbert":E["distilbert_baseline"], "teacher":E["teacher"]}
C={t:np.array(D["students"][t]["test_correct"]) for t in MEM}
C["teacher"]=np.array(D["teacher_test_correct"])
n=len(C["s1"]); T_E,T_A=EN["teacher"],C["teacher"].mean()
conf=np.array(S["conf"])

print("="*88); print("  PART 1 -- DOES A DIFFERENT PRE-TRAINING LINEAGE DECORRELATE ERRORS?"); print("="*88)
def ratio(a,b):
    ea,eb=1-C[a],1-C[b]; return (ea&eb).mean()/max(ea.mean()*eb.mean(),1e-12)
print(f"\n  same lineage (Turc miniatures):")
tur=[("s1","s2"),("s1","s3"),("s1","s4"),("s2","s3"),("s2","s4"),("s3","s4")]
rt=[ratio(a,b) for a,b in tur]
for (a,b),r in zip(tur,rt): print(f"    {a}&{b:12s} {r:5.2f}x")
print(f"    mean {np.mean(rt):.2f}x")
print(f"\n  cross lineage (DistilBERT vs Turc):")
cro=[("distilbert",t) for t in ["s1","s2","s3","s4"]]
rc=[ratio(a,b) for a,b in cro]
for (a,b),r in zip(cro,rc): print(f"    {a}&{b:12s} {r:5.2f}x")
print(f"    mean {np.mean(rc):.2f}x")
print(f"\n  --> cross-lineage errors are {100*(1-np.mean(rc)/np.mean(rt)):+.1f}% "
      f"{'LESS' if np.mean(rc)<np.mean(rt) else 'MORE'} correlated than same-lineage")

print("\n"+"="*88); print("  PART 2 -- EXHAUSTIVE SEARCH FOR acc >= TEACHER AT LOWER ENERGY"); print("="*88)
rows=[]
def add(name, e, ok):
    e=np.asarray(e,float); ok=np.asarray(ok,float)
    rows.append(dict(system=name, mJ=float(e.mean()), acc=float(ok.mean()),
        p95=float(np.percentile(e,95)), dE=float(100*(1-e.mean()/T_E)),
        dA=float(100*(ok.mean()-T_A))))

# k-of-n agreement; abstain -> teacher.  (binary task: correctness determines the vote)
for r in range(2,6):
    for combo in itertools.combinations(MEM,r):
        base=sum(EN[t] for t in combo)
        if base>=T_E: continue
        votes=np.stack([C[t] for t in combo]).sum(0)
        for k in range(1,r+1):
            accept=(votes>=k)|(votes<=r-k)           # k-of-n one way or the other
            maj=(votes> r/2).astype(int)
            e=base+np.where(accept,0.0,EN["teacher"])
            ok=np.where(accept,maj,C["teacher"])
            add(f"{k}of{r}({'+'.join(combo)})->T", e, ok)
        # unanimous-only variant
        unan=(votes==0)|(votes==r)
        add(f"unan({'+'.join(combo)})->T", base+np.where(unan,0.0,EN["teacher"]),
            np.where(unan,(votes>r/2).astype(int),C["teacher"]))

# confidence gate combined with agreement
for combo in itertools.combinations(MEM,2):
    base=sum(EN[t] for t in combo)
    agree=C[combo[0]]==C[combo[1]]
    for th in [0.90,0.95,0.98,0.99,0.995]:
        accept=agree&(conf>=th)
        e=base+np.where(accept,0.0,EN["teacher"])
        ok=np.where(accept,C[combo[0]],C["teacher"])
        add(f"agree({'+'.join(combo)})&conf>={th}->T", e, ok)

hit=[r for r in rows if r["dA"]>=0 and r["dE"]>0]
near=[r for r in rows if -0.6<=r["dA"]<0 and r["dE"]>0]
print(f"\n  configurations searched: {len(rows)}")
print(f"  reaching teacher accuracy at lower energy: {len(hit)}")
if hit:
    print(f"\n  {'system':44s} {'mJ/q':>9s} {'acc':>7s} | {'energy':>9s} {'acc':>8s}")
    for r in sorted(hit,key=lambda r:r["mJ"])[:12]:
        print(f"  {r['system']:44s} {r['mJ']:9.1f} {100*r['acc']:6.2f}% | {r['dE']:+8.1f}% {r['dA']:+8.2f}")
print(f"\n  within 0.6 points of the teacher: {len(near)}")
for r in sorted(near,key=lambda r:r["mJ"])[:8]:
    print(f"  {r['system']:44s} {r['mJ']:9.1f} {100*r['acc']:6.2f}% | {r['dE']:+8.1f}% {r['dA']:+8.2f}")
with open(f"{RESULTS}/e15_search_{task}.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(sorted(rows,key=lambda r:-r["dA"]))
print(f"\n  wrote {RESULTS}/e15_search_{task}.csv")
