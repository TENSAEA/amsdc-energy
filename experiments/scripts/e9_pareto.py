"""
E9 -- Consolidated Pareto analysis across EVERY system and operating point.

Answers the question the paper must answer honestly: given 91.3% recoverable
energy, does ANY deployable policy reach the frontier, or do fixed tiers dominate?
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS
import numpy as np

task="sst2"
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
rie=list(csv.DictReader(open(f"{RESULTS}/e3_rie_{task}.csv")))[0]
LADDER=list(STUDENTS.keys())+["teacher"]; EN=np.array([E[t] for t in LADDER])

def cv(t,s):
    return np.array(D[f"teacher_{s}_correct"]) if t=="teacher" else np.array(D["students"][t][f"{s}_correct"])
C=np.stack([cv(t,"test") for t in LADDER]); n=C.shape[1]
o=np.full(n,len(LADDER)-1)
for k in range(len(LADDER)-1,-1,-1): o[C[k]==1]=k
E_T=EN[-1]*n; RIE=E_T-EN[o].sum()

S=[]   # (name, mJ/query, acc, deployable, median, p95, max)
for k,t in enumerate(LADDER):
    S.append((f"Fixed {t}", EN[k], C[k].mean(), True, EN[k], EN[k], EN[k]))
S.append(("DistilBERT only", E["distilbert_baseline"], D["students"]["distilbert"]["test_acc"], True,
          E["distilbert_baseline"], E["distilbert_baseline"], E["distilbert_baseline"]))
S.append(("Teacher int8", E["teacher_int8"], D["teacher_acc"], True,
          E["teacher_int8"], E["teacher_int8"], E["teacher_int8"]))
for r in csv.DictReader(open(f"{RESULTS}/e8_cascade_{task}.csv")):
    S.append((f"Cascade s1->{r['escalate_to']} t={r['thresh']}", float(r["mJ_per_query"]),
              float(r["acc"]), True, float(r["median"]), float(r["p95"]), float(r["max"])))
for r in csv.DictReader(open(f"{RESULTS}/e5_endtoend_{task}.csv")):
    if r["system"].startswith("AMSD-C"):
        S.append((r["system"]+" (surface)", float(r["mJ_per_query"]), float(r["acc"]), True,
                  float(r["median"]), float(r["p95"]), float(r["max"])))
oe=EN[o]
S.append(("ORACLE (not deployable)", oe.mean(), C[o,np.arange(n)].mean(), False,
          float(np.median(oe)), float(np.percentile(oe,95)), float(oe.max())))

dep=[s for s in S if s[3]]
front=[s for s in dep if not any(t[1]<=s[1] and t[2]>=s[2] and (t[1]<s[1] or t[2]>s[2]) for t in dep if t is not s)]
front.sort(key=lambda s:s[1])

print(f"\n{'='*96}\n  E9: Pareto frontier over {len(dep)} deployable operating points  (task={task}, n={n})\n{'='*96}")
print(f"  RIE = {float(rie['RIE_pct']):.1f}% of teacher energy   oracle {oe.mean():.1f} mJ @ {C[o,np.arange(n)].mean():.4f}\n")
print(f"  {'system':34s} {'mJ/query':>10s} {'acc':>8s} {'median':>9s} {'p95':>10s} {'max':>10s} {'RR':>7s}")
for nm,e,ac,_,md,p95,mx in front:
    print(f"  {nm:34s} {e:10.2f} {ac:8.4f} {md:9.2f} {p95:10.2f} {mx:10.2f} {(E_T-e*n)/RIE:7.3f}")

kinds={}
for nm,e,ac,_,_,_,_ in front:
    k=("fixed tier" if nm.startswith("Fixed") else "DistilBERT" if "Distil" in nm
       else "int8" if "int8" in nm else "cascade" if "Cascade" in nm else "surface router")
    kinds.setdefault(k,0); kinds[k]+=1
print(f"\n  frontier composition: " + ", ".join(f"{v}x {k}" for k,v in sorted(kinds.items(), key=lambda x:-x[1])))

casc=[s for s in front if "Cascade" in s[0]]
print(f"\n  cascades on the frontier: {len(casc)}")
if casc:
    print(f"  {'':2s}best-accuracy cascade:")
    b=max(casc,key=lambda s:s[2])
    print(f"    {b[0]}  ->  {b[1]:.1f} mJ @ {b[2]:.4f}   median {b[4]:.0f}  p95 {b[5]:.0f}  max {b[6]:.0f}"
          f"   tail ratio p95/median = {b[5]/max(b[4],1e-9):.1f}x")
fx=[s for s in front if s[0].startswith("Fixed")]
print(f"\n  fixed tiers on the frontier: {[s[0] for s in fx]}")
print(f"  surface routers on the frontier: {[s[0] for s in front if 'surface' in s[0]]}")
json.dump({"frontier":[{"system":s[0],"mJ":s[1],"acc":s[2],"median":s[4],"p95":s[5],"max":s[6]} for s in front],
           "n_deployable":len(dep),"RIE_pct":float(rie['RIE_pct'])},
          open(f"{RESULTS}/e9_pareto_{task}.json","w"), indent=1)
print(f"\n  wrote {RESULTS}/e9_pareto_{task}.json")
