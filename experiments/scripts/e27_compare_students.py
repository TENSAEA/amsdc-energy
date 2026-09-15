"""E27 -- General comparison of s2 variants under the E25 protocol + opposition diagnostics.

Usage: e27_compare_students.py <task> <label>=<results.json> [<label>=<results.json> ...]
  e.g.  e27 qnli plain7200=experiments/results/e21_q7200_qnli.json plain20k=experiments/results/e21_q20k_qnli.json
        e27 sst2 plain20k=...e21_d20k_sst2.json cad_g1=...e21_cad_g1_sst2.json cad_g5=...e21_cad_g5_sst2.json

For every variant: accuracy, gate AUC, perfect-gate ceiling, teacher-wrong wins (and how
many sit in the student's top-30% confidence), top-k curve vs teacher, and the E25
protocol (threshold on m, evaluated on disjoint remainder, 200 splits, energy guard).
The teacher is asserted identical across variants so the comparison isolates the student.
Energy is architecture-invariant, so e1_energy.csv applies to every variant.
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS
import numpy as np
from sklearn.metrics import roc_auc_score

task=sys.argv[1]; variants=[a.split("=",1) for a in sys.argv[2:]]
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
TE=E["teacher"]; ES=E["s2"]; SEEDS=200

def load(path):
    D=json.load(open(path)); s=D["students"]["s2"]
    if "conf" in s: cf=np.array(s["conf"])
    else:   # E2-era result: confidences were computed later into e19_conf_<task>.json
        cf=np.array(json.load(open(f"{RESULTS}/e19_conf_{task}.json"))["s2"])
    return np.array(s["test_correct"]), cf, np.array(D["teacher_test_correct"]), D
CT=None; out={"task":task,"variants":{}}
for lab,path in variants:
    if not os.path.exists(path): print(f"  [{lab}] missing: {path}"); continue
    C,CF,T,D=load(path)
    if CT is None: CT=T; n=len(CT); fr=[int(n*f) for f in (0.45,0.60,0.75)]
    assert np.array_equal(CT,T), f"teacher differs for {lab}"
    tw=(CT==0); win=tw&(C==1); top30=CF>=np.quantile(CF,0.70)
    ceil=~((C==0)&(CT==1)); p=ceil.mean()
    d=dict(n_train=D.get("n_train"),mode=D.get("mode","plain"),gamma=D.get("gamma"),
           acc=float(C.mean()),auc=float(roc_auc_score(C,CF)),
           ceiling_saved=float(100*(1-(ES+(1-p)*TE)/TE)),
           teacher_wrong=int(tw.sum()),tw_wins=int(win.sum()),tw_wins_top30=int((win&top30).sum()),
           topk={},protocol={})
    o=np.argsort(-CF)
    for k in (10,20,30,50,70,90,100):
        i=o[:int(n*k/100)]; d["topk"][k]=int(C[i].sum()-CT[i].sum())
    for m in fr:
        ds=[];svs=[];drop=0
        for seed in range(SEEDS):
            ix=np.arange(n); np.random.RandomState(seed).shuffle(ix); sel,ev=ix[:m],ix[m:]
            bar=CT[sel].mean(); best=None
            for t in np.unique(np.round(CF[sel],4)):
                esc=CF[sel]<t; e=ES+esc.mean()*TE
                if e>=TE: continue
                if np.where(esc,CT[sel],C[sel]).mean()>=bar and (best is None or e<best[0]): best=(e,t)
            if best is None: drop+=1; continue
            t=best[1]; esc=CF[ev]<t
            ds.append(100*(np.where(esc,CT[ev],C[ev]).mean()-CT[ev].mean()))
            svs.append(100*(1-(ES+esc.mean()*TE)/TE))
        ds=np.array(ds); svs=np.array(svs); ci=1.96*ds.std()/np.sqrt(max(len(ds),1))
        d["protocol"][m]=dict(delta=float(ds.mean()),ci95=float(ci),saved=float(svs.mean()),
                              matched=int((ds>=0).sum()),trials=len(ds),dropped=drop,
                              matches=bool(abs(ds.mean())<=ci))
    out["variants"][lab]=d
print(f"task={task}  teacher={100*CT.mean():.2f}%  n={n}  teacher wrong on {int((CT==0).sum())}\n")
hdr=f"{'variant':>12} {'acc':>7} {'AUC':>7} {'ceil':>6} {'tw-wins':>8} {'top30':>6} | " + " ".join(f"{k:>4}%" for k in (10,20,30,50,70,90,100))
print(hdr)
for lab,d in out["variants"].items():
    print(f"{lab:>12} {100*d['acc']:6.2f}% {d['auc']:7.4f} {d['ceiling_saved']:5.1f}% {d['tw_wins']:4d}/{d['teacher_wrong']:<3d} {d['tw_wins_top30']:6d} | "
          + " ".join(f"{d['topk'][k]:+5d}" for k in (10,20,30,50,70,90,100)))
print(f"\n{'variant':>12} {'m':>5} {'delta vs teacher':>18} {'saved':>7} {'matched':>9}")
for lab,d in out["variants"].items():
    for m,r in d["protocol"].items():
        star=" MATCHES" if r["matches"] else ""
        print(f"{lab:>12} {m:5d} {r['delta']:+8.2f} +-{r['ci95']:4.2f} {r['saved']:6.1f}% {r['matched']:4d}/{r['trials']:<4d}{star}")
tag="_".join(l for l,_ in variants)
json.dump(out,open(f"{RESULTS}/e27_{task}_{tag}.json","w"),indent=1)
print(f"\n[E27] wrote {RESULTS}/e27_{task}_{tag}.json")
