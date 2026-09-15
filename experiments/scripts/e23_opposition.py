"""E23 -- Does a better-distilled cheap path make a better cascade? (No.)

Compares s2 distilled on 7,200 examples (E2) against s2 distilled on 20,000 (E21) under
the identical E20 protocol, and diagnoses the difference. Saves everything it prints.

The teacher is byte-identical across both pipelines (0.9346 on the same 872 queries), so
the comparison isolates the student.
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS
import numpy as np
from sklearn.metrics import roc_auc_score

task="sst2"; SEEDS=200
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
TE=E["teacher"]; ES=E["s2"]
A=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
B=json.load(open(f"{RESULTS}/e21_d20k_{task}.json"))
CFa=np.array(json.load(open(f"{RESULTS}/e19_conf_{task}.json"))["s2"])
Ca=np.array(A["students"]["s2"]["test_correct"]); CTa=np.array(A["teacher_test_correct"])
Cb=np.array(B["students"]["s2"]["test_correct"]); CFb=np.array(B["students"]["s2"]["conf"])
CTb=np.array(B["teacher_test_correct"])
assert np.array_equal(CTa,CTb), "teacher differs between pipelines -- comparison invalid"
CT=CTa; n=len(CT)
FL={"s2_7200":(Ca,CFa,A["students"]["s2"].get("n_train",7200)),
    "s2_20000":(Cb,CFb,B["n_train"])}
out={"teacher_acc":float(CT.mean()),"n_test":n,"E_s2_mJ":ES,"E_teacher_mJ":TE,"fleets":{}}

def protocol(C,CF,m):
    ds=[];svs=[]
    for seed in range(SEEDS):
        ix=np.arange(n); np.random.RandomState(seed).shuffle(ix); sel,ev=ix[:m],ix[m:]
        bar=CT[sel].mean()
        cand=[t for t in np.unique(np.round(CF[sel],4))
              if np.where(CF[sel]<t,CT[sel],C[sel]).mean()>=bar
              and ES+(CF[sel]<t).mean()*TE < TE]
        if not cand: continue
        t=min(cand); esc=CF[ev]<t
        ds.append(100*(np.where(esc,CT[ev],C[ev]).mean()-CT[ev].mean()))
        svs.append(100*(1-(ES+esc.mean()*TE)/TE))
    ds=np.array(ds); svs=np.array(svs)
    return dict(m=m,delta=float(ds.mean()),ci95=float(1.96*ds.std()/np.sqrt(len(ds))),
                saved=float(svs.mean()),saved_sd=float(svs.std()),
                matched=int((ds>=0).sum()),trials=len(ds))

print(f"teacher {100*CT.mean():.2f}% on {n} queries; s2 costs {100*ES/TE:.1f}% of teacher\n")
for k,(C,CF,ntr) in FL.items():
    ceil_mask=~((C==0)&(CT==1)); p=ceil_mask.mean()
    d=dict(n_train=ntr, acc=float(C.mean()), auc=float(roc_auc_score(C,CF)),
           conf_wrong=float(((CF>=0.99)&(C==0)).mean()),
           prec99=float(((CF>=0.99)&(C==1)).sum()/max(((CF>=0.99)).sum(),1)),
           ceiling_saved=float(100*(1-(ES+(1-p)*TE)/TE)),
           teacher_wrong_wins=int(((CT==0)&(C==1)).sum()),
           wins_in_top30=int((((CT==0)&(C==1))&(CF>=np.quantile(CF,0.70))).sum()),
           topk={}, protocol={})
    for kk in (10,20,30,50,70,90,100):
        idx=np.argsort(-CF)[:int(n*kk/100)]
        d["topk"][kk]=int(C[idx].sum()-CT[idx].sum())
    for m in (392,523,654): d["protocol"][m]=protocol(C,CF,m)
    out["fleets"][k]=d
    print(f"{k}: acc={100*d['acc']:.2f}%  AUC={d['auc']:.4f}  ceiling={d['ceiling_saved']:.1f}%")
    print(f"   teacher-wrong wins {d['teacher_wrong_wins']}/57, of which {d['wins_in_top30']} in top-30% conf")
    print(f"   top-k diff vs teacher: " + "  ".join(f"{kk}%:{v:+d}" for kk,v in d["topk"].items()))
    for m,r in d["protocol"].items():
        star=" MATCHES" if abs(r['delta'])<=r['ci95'] else ""
        print(f"   m={m}: {r['delta']:+.2f}+-{r['ci95']:.2f}  saved {r['saved']:.1f}%  {r['matched']}/{r['trials']}{star}")
    print()
json.dump(out,open(f"{RESULTS}/e23_opposition_{task}.json","w"),indent=1)
print(f"[E23] wrote {RESULTS}/e23_opposition_{task}.json")
