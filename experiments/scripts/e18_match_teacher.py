"""E18 -- Can AMSD-C MATCH the teacher's accuracy at lower energy?

E14/E15 asked whether an ENSEMBLE could beat the teacher and found none. That was the
wrong architecture: running all four students costs 5,414.6 mJ = 54% of the teacher
BEFORE any escalation, so the ensemble route caps at ~46% saving even if perfect.

A cascade's cheap path costs 298.9 mJ = 3% of the teacher. And matching the teacher
does NOT require a perfect cheap model -- only this:

    on the ACCEPTED set, the cheap path must be at least as accurate as the teacher.

The teacher is itself wrong on 57 of 872 test queries, so the bar is reachable.

PROTOCOL (no test-set selection):
  1. compute s1 confidence on CALIB and TEST
  2. choose the threshold on CALIB ONLY -- smallest t (largest accept rate, most
     energy saved) whose calib cascade accuracy >= calib teacher accuracy
  3. report that threshold's TEST accuracy and energy
Also reports the architectural ceiling: what a perfect gate would achieve.
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_num_threads(6)
task="sst2"
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
TIERS=list(STUDENTS.keys()); TE=E["teacher"]
Cte={t:np.array(D["students"][t]["test_correct"]) for t in TIERS}
Cca={t:np.array(D["students"][t]["calib_correct"]) for t in TIERS}
Cte["teacher"]=np.array(D["teacher_test_correct"]); Cca["teacher"]=np.array(D["teacher_calib_correct"])

CACHE=f"{RESULTS}/e18_s1_conf_{task}.json"
if os.path.exists(CACHE):
    Z=json.load(open(CACHE)); conf_ca=np.array(Z["calib"]); conf_te=np.array(Z["test"])
    print("[E18] loaded cached s1 confidences")
else:
    d=D["students"]["s1"]["dir"]
    tok=AutoTokenizer.from_pretrained(d); m=AutoModelForSequenceClassification.from_pretrained(d).eval()
    def conf_of(texts):
        out=[]
        with torch.no_grad():
            for i in range(0,len(texts),32):
                b=tok(texts[i:i+32],truncation=True,max_length=128,padding="max_length",return_tensors="pt")
                out.append(torch.softmax(m(**b).logits,-1).max(-1).values)
        return torch.cat(out).numpy()
    print("[E18] computing s1 confidence on calib + test ...",flush=True)
    conf_ca, conf_te = conf_of(D["calib_text"]), conf_of(D["test_text"])
    json.dump({"calib":conf_ca.tolist(),"test":conf_te.tolist()},open(CACHE,"w"))
    del m
Tca,Tte=Cca["teacher"].mean(),Cte["teacher"].mean()
print(f"[E18] calib n={len(conf_ca)} test n={len(conf_te)}   teacher: calib {Tca:.4f}  test {Tte:.4f}\n")

print("="*80); print("  ARCHITECTURAL CEILING -- perfect gate, largest A with path >= teacher on A"); print("="*80)
print(f"{'path':>7} {'E(path)':>9} {'accept%':>8} {'test acc':>9} {'vs T':>6} {'mJ':>9} {'save%':>7}")
ceil={}
Tc=Cte["teacher"].sum(); n=len(Cte["s1"])
for m_ in TIERS:
    A=~((Cte[m_]==0)&(Cte["teacher"]==1)); p=A.mean()
    corr=Cte[m_][A].sum()+Cte["teacher"][~A].sum(); e=E[m_]+(1-p)*TE
    ceil[m_]=(int(corr),float(e))
    print(f"{m_:>7} {E[m_]:9.1f} {100*p:7.1f}% {100*corr/n:8.2f}% {corr-Tc:+6d} {e:9.1f} {100*(1-e/TE):6.1f}%")

print("\n"+"="*80); print("  DEPLOYABLE -- threshold chosen on CALIB, evaluated on TEST"); print("="*80)
def run(conf,C,t,tgt):
    esc=conf<t; return E["s1"]+esc.mean()*E[tgt], np.where(esc,C[tgt],C["s1"])
res=[]
for tgt in ["s3","s4","teacher"]:
    cands=[t for t in sorted(set(np.round(conf_ca,4))) if run(conf_ca,Cca,t,tgt)[1].mean()>=Tca]
    if not cands:
        print(f"  ->{tgt:8s}  no calib threshold reaches teacher calib accuracy\n"); continue
    t=min(cands)
    e_te,ok_te=run(conf_te,Cte,t,tgt); _,ok_ca=run(conf_ca,Cca,t,tgt)
    res.append(dict(target=tgt,t=float(t),calib_acc=float(ok_ca.mean()),
                    test_acc=float(ok_te.mean()),mJ=float(e_te),
                    escalated_pct=float(100*(conf_te<t).mean())))
    print(f"  ->{tgt:8s} t*={t:.4f}  (selected on calib alone)")
    print(f"     calib {ok_ca.mean():.4f} vs {Tca:.4f} ({100*(ok_ca.mean()-Tca):+.2f} pts)")
    print(f"     TEST  {ok_te.mean():.4f} vs {Tte:.4f} ({100*(ok_te.mean()-Tte):+.2f} pts)   <-- held out")
    print(f"     {e_te:.1f} mJ vs {TE:.1f} = {100*(1-e_te/TE):+.1f}% energy ; escalated {100*(conf_te<t).mean():.1f}%\n")
json.dump(dict(teacher_test=float(Tte),teacher_calib=float(Tca),ceiling=ceil,deployable=res),
          open(f"{RESULTS}/e18_match_{task}.json","w"),indent=1)
print(f"[E18] wrote {RESULTS}/e18_match_{task}.json")
