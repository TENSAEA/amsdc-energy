"""E25 -- THE authoritative teacher-matching table (supersedes E20/E22 numbers).

Difference from E20: this requires the selected configuration to ACTUALLY SAVE ENERGY
(E(cheap) + P(escalate)*E(teacher) < E(teacher)). Without that guard the search can fall
back on the degenerate 'accept nothing' threshold, which trivially equals teacher accuracy
while spending MORE than the teacher. E20 reported a few cells contaminated by that.

Protocol: threshold selected on m examples, evaluated on the disjoint 872-m, 200 random
splits per cell. Splits with no feasible (saving) threshold are dropped and counted.
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS
import numpy as np

task="sst2"; SEEDS=200
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
TE=E["teacher"]
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
CF={k:np.array(v) for k,v in json.load(open(f"{RESULTS}/e19_conf_{task}.json")).items()}
T=list(STUDENTS.keys())
C={t:np.array(D["students"][t]["test_correct"]) for t in T}
C["teacher"]=np.array(D["teacher_test_correct"]); n=len(C["s1"])

def cell(path,m):
    Es=E[path]; ds=[];svs=[];drop=0
    for seed in range(SEEDS):
        ix=np.arange(n); np.random.RandomState(seed).shuffle(ix); sel,ev=ix[:m],ix[m:]
        bar=C["teacher"][sel].mean(); best=None
        for t in np.unique(np.round(CF[path][sel],4)):
            esc=CF[path][sel]<t
            e=Es+esc.mean()*TE
            if e>=TE: continue                                   # must actually save
            if np.where(esc,C["teacher"][sel],C[path][sel]).mean()>=bar:
                if best is None or e<best[0]: best=(e,t)
        if best is None: drop+=1; continue
        t=best[1]; esc=CF[path][ev]<t
        ds.append(100*(np.where(esc,C["teacher"][ev],C[path][ev]).mean()-C["teacher"][ev].mean()))
        svs.append(100*(1-(Es+esc.mean()*TE)/TE))
    ds=np.array(ds); svs=np.array(svs)
    ci=1.96*ds.std()/np.sqrt(len(ds))
    return dict(m=m,delta=float(ds.mean()),ci95=float(ci),saved=float(svs.mean()),
                saved_sd=float(svs.std()),matched=int((ds>=0).sum()),trials=len(ds),
                dropped=drop,matches=bool(abs(ds.mean())<=ci))

out={"teacher_acc":float(C["teacher"].mean()),"n":n,"cells":{}}
print(f"teacher {100*C['teacher'].mean():.2f}%  n={n}   (200 splits/cell; guard: energy < teacher)\n")
print(f"{'path':>5} {'E/ET':>6} {'m':>5} {'delta (pts)':>17} {'saved':>15} {'matched':>9} {'drop':>5}")
for path in T:
    for m in (130,261,392,523,654):
        r=cell(path,m); out["cells"][f"{path}_{m}"]=r
        star=" MATCHES" if r["matches"] else ""
        print(f"{path:>5} {E[path]/TE:6.3f} {m:5d} {r['delta']:+8.2f} +-{r['ci95']:5.2f} "
              f"{r['saved']:8.1f} +-{r['saved_sd']:4.1f}% {r['matched']:4d}/{r['trials']:<4d} {r['dropped']:5d}{star}")
    print()
json.dump(out,open(f"{RESULTS}/e25_table_match_{task}.json","w"),indent=1)
print(f"[E25] wrote {RESULTS}/e25_table_match_{task}.json")
