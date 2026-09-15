"""E26 -- Response-space router: route on s1's FULL hidden state, not its max-softmax.

Finding 3 showed query features give AUC 0.605 and s1's max-softmax gives 0.784. But
max-softmax discards 255 of the 256 dimensions s1 already computed in the same forward
pass. This asks: does the full [CLS] representation carry more routing signal, at ZERO
extra energy?

Target: "does s1 suffice" (s1 correct). Honest protocol: 5-fold CV on the 872 test
queries, logistic regression on the hidden state, AUC reported out-of-fold.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, MODELS_DIR
os.environ["HF_HOME"]=os.path.abspath(MODELS_DIR); os.environ["HF_HUB_OFFLINE"]="1"
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

torch.set_num_threads(4); task="sst2"
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
CF=json.load(open(f"{RESULTS}/e19_conf_{task}.json"))
y=np.array(D["students"]["s1"]["test_correct"]); texts=D["test_text"]
d=D["students"]["s1"]["dir"]; tok=AutoTokenizer.from_pretrained(d)
m=AutoModelForSequenceClassification.from_pretrained(d,output_hidden_states=True).eval()
H=[];L=[]
with torch.no_grad():
    for i in range(0,len(texts),32):
        b=tok(texts[i:i+32],truncation=True,max_length=128,padding="max_length",return_tensors="pt")
        o=m(**b); H.append(o.hidden_states[-1][:,0,:].numpy()); L.append(o.logits.numpy())
H=np.concatenate(H); L=np.concatenate(L)
conf=np.array(CF["s1"]); margin=np.abs(L[:,0]-L[:,1])
print(f"s1 hidden dim={H.shape[1]}, n={len(y)}, s1 acc={y.mean():.4f}\n")
def cv_auc(X,label):
    skf=StratifiedKFold(5,shuffle=True,random_state=0); p=np.zeros(len(y))
    for tr,te in skf.split(X,y):
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000))
        clf.fit(X[tr],y[tr]); p[te]=clf.predict_proba(X[te])[:,1]
    return roc_auc_score(y,p),p
res={}
for name,X in [("max-softmax only (current gate)",conf[:,None]),
               ("logit margin",margin[:,None]),
               ("full hidden state (256-d)",H),
               ("hidden state + conf + margin",np.hstack([H,conf[:,None],margin[:,None]]))]:
    a,p=cv_auc(X,name); res[name]=float(a); print(f"  {name:36s} AUC = {a:.4f}")
raw=roc_auc_score(y,conf); print(f"  {'(raw max-softmax, no fitting)':36s} AUC = {raw:.4f}")
json.dump(res,open(f"{RESULTS}/e26_hidden_router_{task}.json","w"),indent=1)
print(f"\n[E26] wrote {RESULTS}/e26_hidden_router_{task}.json")
