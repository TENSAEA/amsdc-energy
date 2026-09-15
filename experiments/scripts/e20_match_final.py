"""E20 -- AMSD-C matches teacher accuracy at lower energy, given enough calibration.

E14/E15 searched ENSEMBLES and found none beating the teacher. That was the wrong
architecture: running all four students costs 5,414.6 mJ = 54% of the teacher BEFORE
any escalation, capping the ensemble route at ~46% even if perfect.

A CASCADE matches the teacher under a much weaker condition:
    on the ACCEPTED set, the cheap path need only be >= the teacher.
The teacher is itself wrong on 57/872, so the bar is reachable.

E18/E19 then appeared to fail -- but for a statistical reason, not an architectural one.
Matching means resolving an accuracy difference of ~0 points, while the standard error
at n=436 is +-1.18 points. The threshold was being fitted to noise.

This experiment measures the calibration requirement directly: select the threshold on
m examples, evaluate on the disjoint remainder, sweep m, and report all four cheap paths
so that neither threshold nor architecture is chosen on the evaluation data.
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS
import numpy as np

task="sst2"; SEEDS=200
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
CF={k:np.array(v) for k,v in json.load(open(f"{RESULTS}/e19_conf_{task}.json")).items()}
T=list(STUDENTS.keys()); TE=E["teacher"]
C={t:np.array(D["students"][t]["test_correct"]) for t in T}
C["teacher"]=np.array(D["teacher_test_correct"]); n=len(C["s1"])

print(f"SST-2 test n={n}; teacher {C['teacher'].sum()}/{n} = {100*C['teacher'].mean():.2f}%")
print(f"Threshold selected on m examples, evaluated on the disjoint {n}-m. {SEEDS} random splits.\n")
out={}
for m_ in T:
    print(f"--- cheap path {m_}  (E={E[m_]:.1f} mJ = {100*E[m_]/TE:.1f}% of teacher) ---")
    print(f"  {'m':>5} {'eval':>5} {'acc delta (pts)':>20} {'energy saved':>17} {'matched':>9}")
    out[m_]=[]
    for frac in (0.15,0.30,0.45,0.60,0.75):
        m=int(n*frac); ds=[];svs=[]
        for seed in range(SEEDS):
            ix=np.arange(n); np.random.RandomState(seed).shuffle(ix)
            sel,ev=ix[:m],ix[m:]
            bar=C["teacher"][sel].mean()
            cand=[t for t in np.unique(np.round(CF[m_][sel],4))
                  if np.where(CF[m_][sel]<t,C["teacher"][sel],C[m_][sel]).mean()>=bar]
            if not cand: continue
            t=min(cand); esc=CF[m_][ev]<t
            acc=np.where(esc,C["teacher"][ev],C[m_][ev]).mean()
            ds.append(100*(acc-C["teacher"][ev].mean()))
            svs.append(100*(1-(E[m_]+esc.mean()*TE)/TE))
        ds=np.array(ds); svs=np.array(svs)
        ci=1.96*ds.std()/np.sqrt(len(ds))
        out[m_].append(dict(m=m,n_eval=n-m,delta=float(ds.mean()),delta_sd=float(ds.std()),
                            delta_ci95=float(ci),saved=float(svs.mean()),saved_sd=float(svs.std()),
                            matched=int((ds>=0).sum()),trials=len(ds)))
        star=" <-- matches" if abs(ds.mean())<=ci else ""
        print(f"  {m:5d} {n-m:5d} {ds.mean():+8.2f} +-{ci:5.2f}(95%CI) {svs.mean():9.1f} +-{svs.std():4.1f}% "
              f"{int((ds>=0).sum()):4d}/{len(ds)}{star}")
    print()
json.dump(dict(teacher_acc=float(C["teacher"].mean()),n=n,paths=out),
          open(f"{RESULTS}/e20_match_final_{task}.json","w"),indent=1)
print(f"[E20] wrote {RESULTS}/e20_match_final_{task}.json")
