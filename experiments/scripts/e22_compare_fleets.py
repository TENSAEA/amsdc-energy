"""E22 -- Does a properly-trained fleet close the gap to the 81.9% ceiling?

Compares two fleets on the IDENTICAL 872 SST-2 validation queries with the IDENTICAL
E20 protocol (threshold selected on m examples, evaluated on the disjoint remainder,
200 random splits, all four cheap paths reported):

    fleet A -- E2  : distilled on 7,200 training examples  (10.7% of SST-2 train)
    fleet B -- E21 : distilled on the full ~67,349 examples

Energy is UNCHANGED between fleets: it depends on architecture, sequence length, batch
and precision, never on training data, so e1_energy.csv applies to both. Only accuracy
differs -- and accuracy is what sets the accept rate, hence the saving:

    energy saved = 1 - E(cheap)/E(teacher) - P(escalate)
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

task="sst2"; SEEDS=200
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
T=list(STUDENTS.keys()); TE=E["teacher"]

def load_fleet(path, conf_path=None):
    D=json.load(open(path))
    C={t:np.array(D["students"][t]["test_correct"]) for t in T}
    C["teacher"]=np.array(D["teacher_test_correct"])
    if "conf" in D["students"][T[0]]:
        CF={t:np.array(D["students"][t]["conf"]) for t in T}
    else:
        CF={k:np.array(v) for k,v in json.load(open(conf_path)).items()}
    return D,C,CF

def sweep(C,CF,label):
    n=len(C["s1"]); out={}
    print(f"\n{'='*82}\n  {label}   teacher {100*C['teacher'].mean():.2f}%  (n={n})\n{'='*82}")
    for m_ in T:
        print(f"  {m_}  acc={100*C[m_].mean():5.2f}%  E={E[m_]:7.1f} mJ ({100*E[m_]/TE:4.1f}% of teacher)")
    print(f"\n  {'path':>5} {'m':>5} {'delta vs teacher':>20} {'energy saved':>16} {'matched':>9}")
    for m_ in T:
        out[m_]=[]
        for frac in (0.30,0.45,0.60,0.75):
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
            if not ds: continue
            ds=np.array(ds); svs=np.array(svs); ci=1.96*ds.std()/np.sqrt(len(ds))
            out[m_].append(dict(m=m,delta=float(ds.mean()),ci=float(ci),saved=float(svs.mean()),
                                saved_sd=float(svs.std()),matched=int((ds>=0).sum()),trials=len(ds)))
            star=" <-- MATCHES" if abs(ds.mean())<=ci else ""
            print(f"  {m_:>5} {m:5d} {ds.mean():+8.2f} +-{ci:5.2f}(95%CI) {svs.mean():9.1f} +-{svs.std():4.1f}% "
                  f"{int((ds>=0).sum()):4d}/{len(ds)}{star}")
    # architectural ceiling
    print(f"\n  ceiling (perfect gate):")
    for m_ in T:
        A=~((C[m_]==0)&(C["teacher"]==1)); p=A.mean(); e=E[m_]+(1-p)*TE
        print(f"    {m_}: accept {100*p:.1f}%  -> {100*(1-e/TE):.1f}% saving")
    return out

A=f"{RESULTS}/e2_distill_{task}.json"; B=f"{RESULTS}/e21_full_{task}.json"
res={}
_,Ca,CFa=load_fleet(A,f"{RESULTS}/e19_conf_{task}.json"); res["fleet_7200"]=sweep(Ca,CFa,"FLEET A -- 7,200 training examples (E2)")
if os.path.exists(B):
    _,Cb,CFb=load_fleet(B); res["fleet_full"]=sweep(Cb,CFb,"FLEET B -- full SST-2 train (E21)")
else:
    print(f"\n[E22] {B} not present yet -- fleet A only.")
json.dump(res,open(f"{RESULTS}/e22_compare_{task}.json","w"),indent=1)
print(f"\n[E22] wrote {RESULTS}/e22_compare_{task}.json")
