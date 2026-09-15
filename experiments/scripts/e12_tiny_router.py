"""
E12 -- A correctly-sized learned router.

E4a's "neural router" was DistilBERT (66.4M), 6x LARGER than the smallest student
it dispatches to (11.2M), giving rho = 5.13. That tests whether an OVERSIZED router
pays for itself -- not whether a router can. A router predicts one ordinal value;
it does not need a full language model.

This trains bert-tiny (4.39M, rho ~ 0.39) and bert-mini (9.59M, rho ~ 0.86) to
predict required capacity directly from the query, fitted on the EXPANDED
calibration set from E11 and evaluated on the untouched test split.

Targets:
  binary   -- does s1 suffice?          (the decision that matters most)
  ordinal  -- which tier is required?   (the full dispatch)
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from models import STUDENTS, MODELS_DIR, RESULTS, SEQ_LEN
os.environ["HF_HOME"]=os.path.abspath(MODELS_DIR); os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
torch.set_num_threads(6)

ap=argparse.ArgumentParser()
ap.add_argument("--router", default="google/bert_uncased_L-2_H-128_A-2")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--target", default="binary", choices=["binary","ordinal"])
a=ap.parse_args()
task="sst2"; SEED=42
torch.manual_seed(SEED); np.random.seed(SEED)

LADDER=list(STUDENTS.keys())+["teacher"]
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
X=json.load(open(f"{RESULTS}/e11_calib_{task}.json"))

def orc(C):
    o=np.full(C.shape[1],len(LADDER)-1)
    for k in range(len(LADDER)-1,-1,-1): o[C[k]==1]=k
    return o
# expanded calibration (E11) + original calibration split (E2)
C_new=np.stack([np.array(X[f"{t}_correct"]) for t in LADDER])
o_new=orc(C_new); T_new=X["text"]
def cv(t,s): return np.array(D[f"teacher_{s}_correct"]) if t=="teacher" else np.array(D["students"][t][f"{s}_correct"])
o_old=orc(np.stack([cv(t,"calib") for t in LADDER])); T_old=D["calib_text"]
T_tr=T_new+T_old; o_tr=np.concatenate([o_new,o_old])
C_te=np.stack([cv(t,"test") for t in LADDER]); o_te=orc(C_te); T_te=D["test_text"]
print(f"[E12] router={a.router}  target={a.target}")
print(f"[E12] train={len(T_tr)} (expanded)  test={len(T_te)}   escalation rate train={100*(o_tr>0).mean():.1f}% test={100*(o_te>0).mean():.1f}%")

y_tr = (o_tr>0).astype(int) if a.target=="binary" else o_tr
y_te = (o_te>0).astype(int) if a.target=="binary" else o_te
NL = 2 if a.target=="binary" else len(LADDER)

tok=AutoTokenizer.from_pretrained(a.router)
m=AutoModelForSequenceClassification.from_pretrained(a.router, num_labels=NL)
nparams=sum(p.numel() for p in m.parameters())
def enc(ts):
    e=tok(ts,padding="max_length",truncation=True,max_length=SEQ_LEN,return_tensors="pt")
    return e["input_ids"], e["attention_mask"]
ii,am=enc(T_tr); it,at=enc(T_te)
# class weights: escalation is the minority class
cw=torch.tensor((len(y_tr)/(NL*np.bincount(y_tr,minlength=NL).clip(1))),dtype=torch.float32)
dl=DataLoader(TensorDataset(ii,am,torch.tensor(y_tr)),batch_size=a.bs,shuffle=True,
              generator=torch.Generator().manual_seed(SEED))
opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=0.01)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.lr,total_steps=a.epochs*len(dl),pct_start=0.1)

@torch.no_grad()
def probs():
    m.eval(); o=[]
    for i in range(0,len(it),128): o.append(torch.softmax(m(input_ids=it[i:i+128],attention_mask=at[i:i+128]).logits,-1))
    return torch.cat(o).numpy()

t0=time.time()
for ep in range(a.epochs):
    m.train()
    for bi,bam,by in dl:
        opt.zero_grad()
        F.cross_entropy(m(input_ids=bi,attention_mask=bam).logits, by, weight=cw).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sched.step()
    P=probs()
    if a.target=="binary":
        auc=roc_auc_score((o_te>0).astype(int), P[:,1]); acc=(P.argmax(1)==y_te).mean()
        print(f"  ep{ep+1} AUC={auc:.4f} acc={acc:.4f} ({time.time()-t0:.0f}s)",flush=True)
    else:
        pred=P.argmax(1); exp=(P*np.arange(NL)).sum(1)
        auc=roc_auc_score((o_te>0).astype(int), 1-P[:,0])
        print(f"  ep{ep+1} route-acc={(pred==o_te).mean():.4f} AUC(esc)={auc:.4f} r={np.corrcoef(exp,o_te)[0,1]:+.4f} ({time.time()-t0:.0f}s)",flush=True)

P=probs()
score = P[:,1] if a.target=="binary" else 1-P[:,0]
res=dict(router=a.router, params=int(nparams), target=a.target,
         auc=float(roc_auc_score((o_te>0).astype(int), score)),
         score=score.tolist(), train_seconds=time.time()-t0,
         n_train=len(T_tr))
if a.target=="ordinal":
    res["expected_tier"]=(P*np.arange(NL)).sum(1).tolist()
    res["pred_tier"]=P.argmax(1).tolist()
tag=a.router.split("/")[-1]+"_"+a.target
d=os.path.join(MODELS_DIR,"router",task,tag); m.save_pretrained(d); tok.save_pretrained(d)
json.dump(res, open(f"{RESULTS}/e12_router_{tag}.json","w"))
print(f"[E12] {nparams/1e6:.2f}M params  AUC={res['auc']:.4f}  saved -> {d}")
