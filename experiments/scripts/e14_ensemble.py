"""
E14 -- Can a deployable system MATCH OR BEAT the teacher at lower energy?

The oracle reaches 97.48% vs the teacher's 93.46%, so the fleet collectively holds
more usable knowledge than the model it was distilled from: students are right where
the teacher is wrong. Single-model confidence cannot capture this (E13: confidence is
binary and scrambled across tiers). Two architectures that might:

  (A) AGREEMENT GATE -- run two students. If they AGREE, accept. If they DISAGREE,
      escalate. Agreement between independently-erring models is a stronger signal
      than one model's self-assessment.

  (B) MAJORITY VOTE -- run k students, take the majority. Exploits complementary
      errors directly. For binary tasks, vote correctness is computable exactly from
      the per-example correctness matrix.

Energy charges EVERY model executed.
"""
import os, sys, json, csv, itertools
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS
import numpy as np

task="sst2"
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
S=json.load(open(f"{RESULTS}/e7_signal_{task}.json"))
LADDER=list(STUDENTS.keys())+["teacher"]; EN={t:E[t] for t in LADDER}
def cv(t): return np.array(D[f"teacher_test_correct"]) if t=="teacher" else np.array(D["students"][t]["test_correct"])
C={t:cv(t) for t in LADDER}; n=len(C["s1"]); T_E,T_A=EN["teacher"],C["teacher"].mean()
conf=np.array(S["conf"])
rows=[]
def add(name,e,ok):
    e=np.asarray(e,dtype=float); ok=np.asarray(ok,dtype=float)
    rows.append(dict(system=name, mJ=float(e.mean()), acc=float(ok.mean()),
        median=float(np.median(e)), p95=float(np.percentile(e,95)), mx=float(e.max()),
        dE=float(100*(1-e.mean()/T_E)), dA=float(100*(ok.mean()-T_A))))

add("Teacher (baseline)", np.full(n,T_E), C["teacher"])
add("Fixed s4", np.full(n,EN["s4"]), C["s4"])

# ---- (A) agreement gate: two students agree -> accept; disagree -> escalate
# binary task: two models agree iff their correctness matches (both right, or both
# wrong => both chose the other class). Accepting an agreed-wrong pair is an error.
for a,b in itertools.combinations(["s1","s2","s3","s4"],2):
    agree = C[a]==C[b]
    base  = EN[a]+EN[b]
    for tgt in ["s3","s4","teacher"]:
        if EN[tgt]<=max(EN[a],EN[b]): continue
        e  = base + np.where(agree,0.0,EN[tgt])
        ok = np.where(agree, C[a], C[tgt])
        add(f"Agree({a},{b}) else {tgt}", e, ok)

# ---- (B) majority vote over k students
for combo in [("s1","s2","s3"),("s1","s2","s4"),("s1","s3","s4"),("s2","s3","s4"),
              ("s1","s2","s3","s4")]:
    votes=np.stack([C[t] for t in combo]).sum(0)
    k=len(combo)
    if k%2==1:
        ok=(votes>k//2).astype(int)
    else:   # even: tie broken by the largest member
        ok=np.where(votes*2==k, C[combo[-1]], (votes>k//2).astype(int))
    add(f"Vote({'+'.join(combo)})", np.full(n,sum(EN[t] for t in combo)), ok)

# ---- (C) vote + escalate when the vote is split
for combo in [("s1","s2","s3"),("s1","s2","s4"),("s2","s3","s4")]:
    votes=np.stack([C[t] for t in combo]).sum(0)
    unan = (votes==0)|(votes==len(combo))          # all agree (all right or all wrong)
    base=sum(EN[t] for t in combo)
    for tgt in ["s4","teacher"]:
        if EN[tgt]<=max(EN[t] for t in combo): continue
        e = base + np.where(unan,0.0,EN[tgt])
        ok= np.where(unan, (votes>len(combo)//2).astype(int), C[tgt])
        add(f"Unanimous({'+'.join(combo)}) else {tgt}", e, ok)

# ---- (D) confidence cascade reference (best from E13)
esc=conf<0.99
add("Cascade s1->s4 t=0.99", EN["s1"]+np.where(esc,EN["s4"],0.0), np.where(esc,C["s4"],C["s1"]))

o=np.full(n,len(LADDER)-1)
for k in range(len(LADDER)-1,-1,-1): o[C[LADDER[k]]==1]=k
add("ORACLE (bound)", np.array([EN[LADDER[k]] for k in o]),
    np.array([C[LADDER[k]][i] for i,k in enumerate(o)]))

beat=[r for r in rows if r["dA"]>=0 and r["dE"]>0 and "ORACLE" not in r["system"]]
print(f"\n{'='*104}\n  E14: can a DEPLOYABLE system match/beat the teacher at lower energy?\n{'='*104}")
print(f"  teacher = {T_E:.0f} mJ @ {100*T_A:.2f}%\n")
print(f"  {'system':38s} {'mJ/q':>9s} {'acc':>7s} {'p95':>9s} | {'energy':>9s} {'acc vs T':>9s}")
for r in sorted(rows,key=lambda r:-r["dA"]):
    star=" <== BEATS TEACHER" if (r["dA"]>=0 and r["dE"]>0 and "ORACLE" not in r["system"]) else ""
    print(f"  {r['system']:38s} {r['mJ']:9.1f} {100*r['acc']:6.2f}% {r['p95']:9.1f} | "
          f"{r['dE']:+8.1f}% {r['dA']:+8.2f}{star}")
print(f"\n  deployable systems matching or beating teacher accuracy at lower energy: {len(beat)}")
with open(f"{RESULTS}/e14_ensemble_{task}.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"  wrote {RESULTS}/e14_ensemble_{task}.csv")
