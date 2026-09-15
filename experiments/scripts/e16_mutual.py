"""
E16 -- Deep Mutual Learning fleet: can decorrelated students replace the teacher?

DIAGNOSIS (E14/E15): every student is distilled toward the SAME teacher, so they
inherit the same misconceptions. Measured error correlation 3.2x-7.6x above
independence. A 4-student vote reaches 90.02% against a 95.76% ceiling -- 5.74
points destroyed purely by correlation.

HYPOTHESIS (Deep Mutual Learning, Zhang et al. 2018): students that learn from the
data AND FROM EACH OTHER are pushed apart rather than together. If correlation falls,
the vote should climb toward the ceiling and cross the teacher's 93.46%.

LOSS per student i:
    L_i = a*T^2*KL(teacher || s_i)          keep teacher knowledge (coverage)
        + (1-a)*CE(y, s_i)                  ground truth
        + b * mean_j!=i [ -KL(s_j || s_i) ] DIVERGENCE from peers (negative weight
                                            => actively decorrelate)
b=0 reproduces the E2 baseline exactly, so the comparison is controlled.
"""
import os, sys, json, time, argparse, random
sys.path.insert(0, os.path.dirname(__file__))
from models import TEACHER, STUDENTS, MODELS_DIR, DATA_DIR, RESULTS, SEQ_LEN
os.environ["HF_HOME"]=os.path.abspath(MODELS_DIR); os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
torch.set_num_threads(6)

ap=argparse.ArgumentParser()
ap.add_argument("--beta", type=float, default=0.5, help="peer-divergence weight (0 = E2 baseline)")
ap.add_argument("--alpha", type=float, default=0.7)
ap.add_argument("--temp", type=float, default=4.0)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max-train", type=int, default=8000)
ap.add_argument("--tag", default="mutual")
a=ap.parse_args()
task="sst2"; SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"[E16] beta={a.beta} alpha={a.alpha} T={a.temp} epochs={a.epochs}", flush=True)

ds=load_dataset("nyu-mll/glue",task,cache_dir=DATA_DIR)
tr_full, test = ds["train"], ds["validation"]
keep=list(range(len(tr_full))); random.Random(SEED).shuffle(keep)
tr_full=tr_full.select(sorted(keep[:a.max_train]))
idx=list(range(len(tr_full))); random.Random(SEED).shuffle(idx)
ncal=int(0.10*len(idx)); train=tr_full.select(idx[ncal:]); calib=tr_full.select(idx[:ncal])
print(f"[E16] train={len(train)} calib={len(calib)} test={len(test)}", flush=True)

tname=TEACHER[task]; ttok=AutoTokenizer.from_pretrained(tname)
teacher=AutoModelForSequenceClassification.from_pretrained(tname).eval()
NL=teacher.config.num_labels
def enc(tok,split):
    e=tok([r["sentence"] for r in split],padding="max_length",truncation=True,
          max_length=SEQ_LEN,return_tensors="pt")
    return e["input_ids"],e["attention_mask"],torch.tensor([r["label"] for r in split])
@torch.no_grad()
def tlogits(split,lab):
    ii,am,y=enc(ttok,split); o=[]; t0=time.time()
    for i in range(0,len(ii),64):
        o.append(teacher(input_ids=ii[i:i+64],attention_mask=am[i:i+64]).logits)
        if i%1600==0: print(f"    {lab} {i}/{len(ii)} ({time.time()-t0:.0f}s)",flush=True)
    return torch.cat(o),y
print("[E16] caching teacher logits ...", flush=True)
TL,Ytr=tlogits(train,"train"); TLte,Yte=tlogits(test,"test"); TLca,Yca=tlogits(calib,"calib")
tacc=(TLte.argmax(-1)==Yte).float().mean().item()
print(f"[E16] TEACHER test acc = {tacc:.4f}", flush=True); del teacher

TIERS=list(STUDENTS.keys())
toks={t:AutoTokenizer.from_pretrained(STUDENTS[t]) for t in TIERS}
models={t:AutoModelForSequenceClassification.from_pretrained(STUDENTS[t],num_labels=NL) for t in TIERS}
# identical tokenizer family -> one encoding shared by all students
ii,am,y=enc(toks["s1"],train); iit,amt,_=enc(toks["s1"],test); iic,amc,_=enc(toks["s1"],calib)
dl=DataLoader(TensorDataset(ii,am,y,TL),batch_size=a.bs,shuffle=True,
              generator=torch.Generator().manual_seed(SEED))
opts={t:torch.optim.AdamW(models[t].parameters(),lr=a.lr,weight_decay=0.01) for t in TIERS}
steps=a.epochs*len(dl)
scheds={t:torch.optim.lr_scheduler.OneCycleLR(opts[t],max_lr=a.lr,total_steps=steps,pct_start=0.1) for t in TIERS}

@torch.no_grad()
def evaluate(m,I,A,Y):
    m.eval(); p=[]
    for i in range(0,len(I),64): p.append(m(input_ids=I[i:i+64],attention_mask=A[i:i+64]).logits.argmax(-1))
    p=torch.cat(p); return (p==Y).float().mean().item(), (p==Y).int().tolist()

t0=time.time(); step=0
for ep in range(a.epochs):
    for t in TIERS: models[t].train()
    for bi,bam,by,btl in dl:
        # one forward per student, then a joint backward with peer divergence
        logits={t:models[t](input_ids=bi,attention_mask=bam).logits for t in TIERS}
        for t in TIERS:
            kd=F.kl_div(F.log_softmax(logits[t]/a.temp,-1),F.softmax(btl/a.temp,-1),
                        reduction="batchmean")*(a.temp**2)
            ce=F.cross_entropy(logits[t],by)
            loss=a.alpha*kd+(1-a.alpha)*ce
            if a.beta>0:
                div=torch.stack([F.kl_div(F.log_softmax(logits[t],-1),
                                          F.softmax(logits[o].detach(),-1),
                                          reduction="batchmean") for o in TIERS if o!=t]).mean()
                loss=loss - a.beta*div          # NEGATIVE => push apart
            opts[t].zero_grad(); loss.backward(retain_graph=(t!=TIERS[-1]))
            torch.nn.utils.clip_grad_norm_(models[t].parameters(),1.0)
            opts[t].step(); scheds[t].step()
        step+=1
        if step%50==0: print(f"    ep{ep+1} step {step}/{steps} ({time.time()-t0:.0f}s)",flush=True)
    accs={t:evaluate(models[t],iit,amt,Yte)[0] for t in TIERS}
    print(f"  ep{ep+1}: "+"  ".join(f"{t}={accs[t]:.4f}" for t in TIERS)+f"  ({time.time()-t0:.0f}s)",flush=True)

res={"task":task,"beta":a.beta,"alpha":a.alpha,"temp":a.temp,"epochs":a.epochs,
     "teacher_acc":tacc,"students":{}}
for t in TIERS:
    at,ct=evaluate(models[t],iit,amt,Yte); ac,cc=evaluate(models[t],iic,amc,Yca)
    d=os.path.join(MODELS_DIR,a.tag,task,t); models[t].save_pretrained(d); toks[t].save_pretrained(d)
    res["students"][t]={"test_acc":at,"calib_acc":ac,"test_correct":ct,"calib_correct":cc,
                        "params":sum(p.numel() for p in models[t].parameters())}
res["teacher_test_correct"]=(TLte.argmax(-1)==Yte).int().tolist()
res["teacher_calib_correct"]=(TLca.argmax(-1)==Yca).int().tolist()
res["test_text"]=[r["sentence"] for r in test]; res["calib_text"]=[r["sentence"] for r in calib]
res["n_test"]=len(test); res["n_calib"]=len(calib)
json.dump(res,open(f"{RESULTS}/e16_{a.tag}_{task}.json","w"))
print(f"\n[E16] SUMMARY  (teacher {tacc:.4f})")
for t in TIERS: print(f"  {t:4s} {res['students'][t]['test_acc']:.4f}")
print(f"[E16] wrote {RESULTS}/e16_{a.tag}_{task}.json")
