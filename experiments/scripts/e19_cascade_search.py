"""E19 -- Find a cascade that MATCHES teacher accuracy at lower energy, honestly.

Fixes two flaws in E18:
  (1) the calibration split is contaminated (teacher scores 0.9888 there vs 0.9346 on
      test) because it comes from the teacher's own fine-tuning corpus, so it sets an
      impossible bar. We select on a held-out HALF OF TEST instead.
  (2) choosing the smallest threshold that clears the bar sits on a knife edge and does
      not transfer. We select with a MARGIN and take the most economical threshold that
      clears teacher + margin.

Architectures searched (energy charges every model executed):
  2-stage  s_i -> teacher            gated on conf(s_i)
  3-stage  s_i -> s_j -> teacher     gated on conf(s_i) then conf(s_j)
Evaluation: 5 seeds x 2 directions = 10 held-out evaluations; report mean +- sd.
"""
import os, sys, json, csv, itertools
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_num_threads(6); task="sst2"
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json")); TIERS=list(STUDENTS.keys()); TE=E["teacher"]
C={t:np.array(D["students"][t]["test_correct"]) for t in TIERS}
C["teacher"]=np.array(D["teacher_test_correct"]); n=len(C["s1"])

CACHE=f"{RESULTS}/e19_conf_{task}.json"
if os.path.exists(CACHE):
    CF={k:np.array(v) for k,v in json.load(open(CACHE)).items()}; print("[E19] cached confidences")
else:
    CF={}
    for t in TIERS:
        d=D["students"][t]["dir"]
        tok=AutoTokenizer.from_pretrained(d); m=AutoModelForSequenceClassification.from_pretrained(d).eval()
        o=[]
        with torch.no_grad():
            for i in range(0,n,32):
                b=tok(D["test_text"][i:i+32],truncation=True,max_length=128,padding="max_length",return_tensors="pt")
                o.append(torch.softmax(m(**b).logits,-1).max(-1).values)
        CF[t]=torch.cat(o).numpy(); del m
        print(f"[E19] conf({t}) done  mean={CF[t].mean():.4f}",flush=True)
    json.dump({k:v.tolist() for k,v in CF.items()},open(CACHE,"w"))

def two(i,t,idx):
    esc=CF[i][idx]<t
    return E[i]+esc.mean()*TE, np.where(esc,C["teacher"][idx],C[i][idx])
def three(i,j,t1,t2,idx):
    e1=CF[i][idx]<t1; e2=e1&(CF[j][idx]<t2)
    en=E[i]+e1.mean()*E[j]+e2.mean()*TE
    ok=np.where(~e1,C[i][idx],np.where(~e2,C[j][idx],C["teacher"][idx]))
    return en,ok

ARCH=[]
for i in TIERS: ARCH.append(("2",(i,)))
for i,j in itertools.permutations(TIERS,2):
    if E[j]>E[i]: ARCH.append(("3",(i,j)))

def best_on(idx,margin):
    """most economical config clearing teacher+margin on idx"""
    bar=C["teacher"][idx].mean()+margin; best=None
    for kind,ms in ARCH:
        if kind=="2":
            i=ms[0]
            for t in np.unique(np.round(CF[i][idx],3)):
                e,ok=two(i,t,idx)
                if ok.mean()>=bar and (best is None or e<best[0]): best=(e,kind,ms,(t,))
        else:
            i,j=ms
            for t1 in np.unique(np.round(CF[i][idx],2)):
                for t2 in np.unique(np.round(CF[j][idx],2)):
                    e,ok=three(i,j,t1,t2,idx)
                    if ok.mean()>=bar and (best is None or e<best[0]): best=(e,kind,ms,(t1,t2))
    return best
def apply(cfg,idx):
    _,kind,ms,ts=cfg
    return two(ms[0],ts[0],idx) if kind=="2" else three(ms[0],ms[1],ts[0],ts[1],idx)

for margin in (0.0,0.005,0.010,0.015,0.020):
    ds=[];svs=[];picks=[]
    for seed in range(5):
        ix=np.arange(n); np.random.RandomState(seed).shuffle(ix)
        for a,b in ((ix[:n//2],ix[n//2:]),(ix[n//2:],ix[:n//2])):
            cfg=best_on(a,margin)
            if cfg is None: continue
            e,ok=apply(cfg,b)
            ds.append(100*(ok.mean()-C["teacher"][b].mean())); svs.append(100*(1-e/TE))
            picks.append("+".join(cfg[2]))
    if not ds: print(f"margin {margin:+.3f}: no config clears the bar"); continue
    ds=np.array(ds); svs=np.array(svs)
    from collections import Counter
    print(f"margin {margin:+.3f} | held-out acc vs teacher {ds.mean():+.2f} +- {ds.std():.2f} pts "
          f"| energy saved {svs.mean():5.1f} +- {svs.std():4.1f}% | matched {int((ds>=0).sum())}/{len(ds)} "
          f"| picks {Counter(picks).most_common(2)}")
