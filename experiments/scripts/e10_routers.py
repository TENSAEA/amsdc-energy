"""
E10 -- Search for a routing signal that is BOTH predictive and cheap.

E4a tested 12 hand-crafted SURFACE statistics (length, punctuation, cue words,
lexical density) and found |r| < 0.03 even under a gradient-boosting ceiling.
That rules out surface FORM. It does NOT rule out lexical CONTENT, which is also
cheap: a sparse dot product over a hashed vocabulary costs microjoules, orders of
magnitude below a transformer forward pass.

This tests, in increasing cost order:
  R1  hand-crafted surface features            (E4a baseline, ~0 J)
  R2  bag-of-words / TF-IDF + linear           (sparse dot product, ~0 J)
  R3  char n-gram TF-IDF + linear              (~0 J)
  R4  static-embedding mean + linear           (one lookup per token, ~0 J)
  R5  s1 confidence                            (requires running s1)
Each is scored on the SAME target (oracle tier) with the SAME protocol:
fit on calibration, evaluate on test, no leakage.
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

task="sst2"
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
LADDER=list(STUDENTS.keys())+["teacher"]
def cv(t,s): return np.array(D[f"teacher_{s}_correct"]) if t=="teacher" else np.array(D["students"][t][f"{s}_correct"])
C_te=np.stack([cv(t,"test") for t in LADDER]); C_ca=np.stack([cv(t,"calib") for t in LADDER])
def orc(C):
    o=np.full(C.shape[1],len(LADDER)-1)
    for k in range(len(LADDER)-1,-1,-1): o[C[k]==1]=k
    return o
o_te,o_ca=orc(C_te),orc(C_ca)
Tca,Tte=D["calib_text"],D["test_text"]
esc_te=(o_te>0).astype(int); esc_ca=(o_ca>0).astype(int)
print(f"  calibration: {len(Tca)}   test: {len(Tte)}   escalation base rate: {esc_te.mean():.3f}")

def report(name, score_te, pred_tier=None):
    r=np.corrcoef(score_te,o_te)[0,1]
    auc=roc_auc_score(esc_te, score_te)
    ra=float((pred_tier==o_te).mean()) if pred_tier is not None else float("nan")
    print(f"  {name:38s} r={r:+.4f}  AUC={auc:.4f}" + (f"  route-acc={ra:.4f}" if pred_tier is not None else ""))
    return dict(router=name, r=float(r), auc=float(auc), route_acc=ra)

res=[]
print(f"\n{'='*78}\n  E10: which signals predict required capacity?\n{'='*78}\n")

# R1 -- surface (E4a baseline, reproduced here for comparability)
import re
W=re.compile(r"\w+")
def surf(t):
    w=W.findall(t.lower()); nw=max(len(w),1)
    return [len(t),nw,t.count(",")+t.count(";"),sum(len(x) for x in w)/nw,len(set(w))/nw]
Xc,Xt=np.array([surf(t) for t in Tca]),np.array([surf(t) for t in Tte])
m=Ridge(alpha=1.0).fit(Xc,o_ca); res.append(report("R1 surface statistics", m.predict(Xt)))

# R2 -- word TF-IDF
for nf in (2000, 10000):
    v=TfidfVectorizer(ngram_range=(1,2), max_features=nf, sublinear_tf=True)
    Zc=v.fit_transform(Tca); Zt=v.transform(Tte)
    m=Ridge(alpha=1.0).fit(Zc,o_ca)
    clf=LogisticRegression(max_iter=2000).fit(Zc,o_ca)
    res.append(report(f"R2 word TF-IDF 1-2gram ({nf}) + ridge", m.predict(Zt), clf.predict(Zt)))

# R3 -- char n-gram TF-IDF
v=TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=10000, sublinear_tf=True)
Zc=v.fit_transform(Tca); Zt=v.transform(Tte)
m=Ridge(alpha=1.0).fit(Zc,o_ca); clf=LogisticRegression(max_iter=2000).fit(Zc,o_ca)
res.append(report("R3 char 3-5gram TF-IDF + ridge", m.predict(Zt), clf.predict(Zt)))

# R3b -- direct escalation target (binary is easier than 5-way ordinal)
v=TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)
Zc=v.fit_transform(Tca); Zt=v.transform(Tte)
clf=LogisticRegression(max_iter=2000).fit(Zc,esc_ca)
p=clf.predict_proba(Zt)[:,1]
print(f"  {'R3b word TF-IDF -> P(escalate)':38s} r={np.corrcoef(p,o_te)[0,1]:+.4f}  AUC={roc_auc_score(esc_te,p):.4f}")
res.append(dict(router="R3b word TF-IDF -> P(escalate)", r=float(np.corrcoef(p,o_te)[0,1]),
                auc=float(roc_auc_score(esc_te,p)), route_acc=float("nan")))

# R5 -- s1 confidence (reference; requires running s1)
s7=json.load(open(f"{RESULTS}/e7_signal_{task}.json"))
conf=np.array(s7["conf"])
print(f"  {'R5 s1 confidence (runs s1)':38s} r={np.corrcoef(conf,o_te)[0,1]:+.4f}  AUC={roc_auc_score(esc_te,-conf):.4f}")
res.append(dict(router="R5 s1 confidence", r=float(np.corrcoef(conf,o_te)[0,1]),
                auc=float(roc_auc_score(esc_te,-conf)), route_acc=float("nan")))

json.dump(res, open(f"{RESULTS}/e10_routers_{task}.json","w"), indent=1)
print(f"\n  (AUC 0.5 = no signal.  s1 confidence is the reference at {roc_auc_score(esc_te,-conf):.3f})")
print(f"  wrote {RESULTS}/e10_routers_{task}.json")
