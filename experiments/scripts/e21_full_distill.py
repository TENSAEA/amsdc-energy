"""E21 -- Re-distil the fleet on the FULL SST-2 training set, on the GPU.

Why: E2 trained every student on 7,200 of SST-2's ~67,349 training examples (10.7%),
an artefact of a 10-hour compute budget rather than a principled choice. s4 (bert-base)
reached 91.40% where bert-base fine-tuned normally on SST-2 reaches ~92.7%. The students
are undertrained, and student accuracy is what sets the cascade's accept rate:

    energy saved = 1 - E(cheap)/E(teacher) - P(escalate)

A more accurate cheap path accepts more queries at teacher-level accuracy, so it cuts
P(escalate) directly. This is the most direct route to closing the gap between the
23.2% we measure and the 81.9% a perfect gate would achieve.

IMPORTANT -- measurement integrity:
  * Training runs on the GPU in a SEPARATE venv (.venv-gpu). All ENERGY measurement
    stays on CPU in the original .venv, byte-identical to E1. Energy depends on
    architecture, sequence length, batch and precision -- never on training data -- so
    e1_energy.csv remains valid unchanged. Only accuracy is re-measured.
  * Evaluation is the same 872 SST-2 validation queries, same order, as every other
    experiment. Calibration uses a split of those 872 (per E20), never the training
    corpus, because the teacher was fine-tuned on train and scores 0.9888 there.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, MODELS_DIR, STUDENTS, TEACHER, SEQ_LEN, DATA_DIR
# must precede the transformers import: every other script in this project points the
# hub cache at experiments/models, where all checkpoints are already downloaded
os.environ["HF_HOME"] = os.path.abspath(MODELS_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1" 
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset

ap=argparse.ArgumentParser()
ap.add_argument("--epochs",type=int,default=2); ap.add_argument("--bs",type=int,default=32)
ap.add_argument("--lr",type=float,default=5e-5); ap.add_argument("--alpha",type=float,default=0.7)
ap.add_argument("--temp",type=float,default=4.0); ap.add_argument("--tag",default="full")
ap.add_argument("--max_train",type=int,default=0)   # 0 = all
ap.add_argument("--max_test",type=int,default=0)    # smoke-test only
ap.add_argument("--tiers",default="")              # comma list, e.g. "s2"; empty = all
ap.add_argument("--task",default="sst2")
ap.add_argument("--mode",default="plain",choices=["plain","cad"])
ap.add_argument("--gamma",type=float,default=3.0)   # cad: weight on teacher-WRONG examples
a=ap.parse_args()
task=a.task; SEED=42
dev="cuda" if torch.cuda.is_available() else "cpu"
# Pascal (compute capability < 7.0) has no fp16 tensor cores and runs fp16 at a small
# fraction of fp32 throughput, so half precision is a pessimisation there, not a speedup.
USE_FP16 = (dev=="cuda" and torch.cuda.get_device_capability(0)[0] >= 7)
print(f"[E21] device={dev} torch={torch.__version__}",flush=True)
if dev=="cuda": print(f"[E21] {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} fp16={USE_FP16}",flush=True)

ds=load_dataset("nyu-mll/glue",task,cache_dir=DATA_DIR)
VALSPLIT={"mnli":"validation_matched"}
train=list(ds["train"]); test=list(ds[VALSPLIT.get(task,"validation")])
if a.max_test: test=test[:a.max_test]
if a.max_train: train=train[:a.max_train]
print(f"[E21] train={len(train)}  test={len(test)}  (E2 used 7200)",flush=True)

tname=TEACHER[task]
ttok=AutoTokenizer.from_pretrained(tname)
teacher=AutoModelForSequenceClassification.from_pretrained(tname).to(dev).eval()
if USE_FP16: teacher=teacher.half()

FIELDS={"sst2":("sentence",None),"qnli":("question","sentence"),"rte":("sentence1","sentence2"),"mnli":("premise","hypothesis")}
F1,F2=FIELDS[task]
def texts_of(rows):
    return [r[F1] for r in rows] if F2 is None else ([r[F1] for r in rows],[r[F2] for r in rows])
def enc(tok,texts):
    b=(tok(texts,truncation=True,max_length=SEQ_LEN,padding="max_length",return_tensors="pt")
       if not isinstance(texts,tuple) else
       tok(texts[0],texts[1],truncation=True,max_length=SEQ_LEN,padding="max_length",return_tensors="pt"))
    return b["input_ids"],b["attention_mask"]

Ytr=torch.tensor([r["label"] for r in train]); Yte=torch.tensor([r["label"] for r in test])
print("[E21] caching teacher logits ...",flush=True); t0=time.time()
TB=64   # teacher inference batch; bert-large fp32 on a 6 GB card
def teacher_logits(texts):
    ii,am=enc(ttok,texts); N=ii.shape[0]; out=[]
    with torch.no_grad():
        for i in range(0,N,TB):
            out.append(teacher(input_ids=ii[i:i+TB].to(dev),attention_mask=am[i:i+TB].to(dev)).logits.float().cpu())
            if i%(TB*100)==0: print(f"    {i}/{N} ({time.time()-t0:.0f}s)",flush=True)
    return torch.cat(out)
TLCACHE=f"{RESULTS}/e21_teacherlogits_{task}_{len(train)}_{len(test)}.pt"
import glob as _g
_bigger=[f for f in _g.glob(f"{RESULTS}/e21_teacherlogits_{task}_*_{len(test)}.pt")
         if int(f.split("_")[-2])>=len(train)]
if os.path.exists(TLCACHE):
    z=torch.load(TLCACHE); TLtr,TLte=z["train"],z["test"]
    print(f"[E21] reusing cached teacher logits: {TLCACHE}",flush=True)
elif _bigger:
    z=torch.load(sorted(_bigger)[0]); TLtr,TLte=z["train"][:len(train)],z["test"]
    print(f"[E21] sliced train logits from larger cache: {sorted(_bigger)[0]}",flush=True)
else:
    TLtr=teacher_logits(texts_of(train)); TLte=teacher_logits(texts_of(test))
    torch.save({"train":TLtr,"test":TLte},TLCACHE)
    print(f"[E21] saved teacher logits -> {TLCACHE}",flush=True)
NUM_LABELS=teacher.config.num_labels
tacc=(TLte.argmax(-1)==Yte).float().mean().item()
print(f"[E21] teacher test acc = {tacc:.4f}  (E2: 0.9346)",flush=True)
del teacher; torch.cuda.empty_cache()

res={"task":task,"tag":a.tag,"mode":a.mode,"gamma":a.gamma,"n_train":len(train),"n_test":len(test),"epochs":a.epochs,
     "alpha":a.alpha,"temp":a.temp,"lr":a.lr,"bs":a.bs,"teacher_acc":tacc,"students":{},
     "teacher_test_correct":(TLte.argmax(-1)==Yte).int().tolist(),
     "test_text":[str(r[F1]) for r in test],"test_labels":Yte.tolist()}

TIERS_RUN=[t.strip() for t in a.tiers.split(",") if t.strip()] or list(STUDENTS.keys())
print(f"[E21] training tiers: {TIERS_RUN}  mode={a.mode}" + (f" gamma={a.gamma}" if a.mode=="cad" else ""),flush=True)
SAFE=os.path.join(MODELS_DIR,"safetensors")
for tier in TIERS_RUN:
    ck=os.path.join(SAFE,tier) if os.path.isdir(os.path.join(SAFE,tier)) else STUDENTS[tier]
    print(f"\n[E21] === {tier}: {ck} ===",flush=True); t0=time.time()
    tok=AutoTokenizer.from_pretrained(ck)
    m=AutoModelForSequenceClassification.from_pretrained(ck,num_labels=NUM_LABELS,ignore_mismatched_sizes=True).to(dev)
    itr,atr=enc(tok,texts_of(train)); ite,ate=enc(tok,texts_of(test))
    dl=DataLoader(TensorDataset(itr,atr,Ytr,TLtr),batch_size=a.bs,shuffle=True,drop_last=False)
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr)
    steps=a.epochs*len(dl)
    sch=get_linear_schedule_with_warmup(opt,int(0.1*steps),steps)
    scaler=torch.amp.GradScaler("cuda",enabled=USE_FP16)
    step=0
    for ep in range(a.epochs):
        m.train()
        for bi,bam,by,btl in dl:
            bi,bam,by,btl=bi.to(dev),bam.to(dev),by.to(dev),btl.to(dev)
            with torch.amp.autocast("cuda",enabled=USE_FP16):
                lg=m(input_ids=bi,attention_mask=bam).logits
                if a.mode=="plain":
                    kd=F.kl_div(F.log_softmax(lg/a.temp,-1),F.softmax(btl/a.temp,-1),reduction="batchmean")*(a.temp**2)
                    loss=a.alpha*kd+(1-a.alpha)*F.cross_entropy(lg,by)
                else:
                    # Complement-Aware Distillation: imitate the teacher only where it is RIGHT.
                    # Where the teacher is WRONG, learn the hard label alone, upweighted by gamma,
                    # so the student keeps a confident disagreement on the teacher's error set.
                    tright=(btl.argmax(-1)==by)
                    kd_i=F.kl_div(F.log_softmax(lg/a.temp,-1),F.softmax(btl/a.temp,-1),reduction="none").sum(-1)*(a.temp**2)
                    ce_i=F.cross_entropy(lg,by,reduction="none")
                    loss=torch.where(tright, a.alpha*kd_i+(1-a.alpha)*ce_i, a.gamma*ce_i).mean()
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            scaler.step(opt); scaler.update(); sch.step(); step+=1
            if step%500==0: print(f"    step {step}/{steps} ({time.time()-t0:.0f}s)",flush=True)
    m.eval(); pr=[]; cf=[]
    with torch.no_grad():
        for i in range(0,len(test),128):
            lg=m(input_ids=ite[i:i+128].to(dev),attention_mask=ate[i:i+128].to(dev)).logits.float().cpu()
            pr.append(lg.argmax(-1)); cf.append(torch.softmax(lg,-1).max(-1).values)
    pr=torch.cat(pr); cf=torch.cat(cf); acc=(pr==Yte).float().mean().item()
    d=os.path.join(MODELS_DIR,a.tag,task,tier); os.makedirs(d,exist_ok=True)
    m.save_pretrained(d); tok.save_pretrained(d)
    res["students"][tier]={"ckpt":STUDENTS[tier],"dir":d,"test_acc":acc,
        "test_correct":(pr==Yte).int().tolist(),"conf":cf.tolist(),
        "params":sum(p.numel() for p in m.parameters()),"train_seconds":time.time()-t0}
    print(f"[E21] {tier}: acc={acc:.4f}  ({time.time()-t0:.0f}s)",flush=True)
    json.dump(res,open(f"{RESULTS}/e21_{a.tag}_{task}.json","w"))   # save after EVERY tier
    print(f"[E21] checkpointed results after {tier}",flush=True)
    del m; torch.cuda.empty_cache()

json.dump(res,open(f"{RESULTS}/e21_{a.tag}_{task}.json","w"))
print(f"\n[E21] SUMMARY (teacher {tacc:.4f})")
for t in TIERS_RUN: print(f"  {t:3s} {res['students'][t]['test_acc']:.4f}")
print(f"[E21] wrote {RESULTS}/e21_{a.tag}_{task}.json")
