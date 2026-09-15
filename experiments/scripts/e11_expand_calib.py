"""
E11 -- Expand the calibration set.

The router in E10 was fitted on 800 examples, which is far too few to fit any
learned routing function fairly. This runs the full ladder (all students +
teacher) over previously-UNUSED SST-2 training examples to build a calibration
set an order of magnitude larger. The test split is never touched.
"""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(__file__))
from models import TEACHER, STUDENTS, MODELS_DIR, DATA_DIR, RESULTS, SEQ_LEN
os.environ["HF_HOME"]=os.path.abspath(MODELS_DIR); os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
torch.set_num_threads(6); torch.set_grad_enabled(False)

task="sst2"; SEED=42; N_NEW=8000
D=json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
ds=load_dataset("nyu-mll/glue",task,cache_dir=DATA_DIR)["train"]

# reproduce E2's split to find UNUSED examples
n=len(ds); keep=list(range(n)); random.Random(SEED).shuffle(keep)
used=set(sorted(keep[:8000]))                      # E2 used these 8000
unused=[i for i in range(n) if i not in used]
random.Random(SEED+1).shuffle(unused)
sel=sorted(unused[:N_NEW])
sub=ds.select(sel)
texts=[r["sentence"] for r in sub]; y=torch.tensor([r["label"] for r in sub])
print(f"[E11] {len(texts)} new calibration examples from unused train (test never touched)", flush=True)

def run(name, path, is_teacher=False):
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(path)
    m=AutoModelForSequenceClassification.from_pretrained(path).eval()
    preds=[]
    for i in range(0,len(texts),64):
        e=tok(texts[i:i+64],padding="max_length",truncation=True,max_length=SEQ_LEN,return_tensors="pt")
        preds.append(m(**e).logits.argmax(-1))
        if i%1600==0: print(f"    {name} {i}/{len(texts)} ({time.time()-t0:.0f}s)",flush=True)
    p=torch.cat(preds); acc=(p==y).float().mean().item()
    print(f"  {name}: acc={acc:.4f} [{time.time()-t0:.0f}s]",flush=True)
    return (p==y).int().tolist()

out={"text":texts,"labels":y.tolist(),"n":len(texts)}
for tier in STUDENTS:
    out[f"{tier}_correct"]=run(tier, os.path.join(MODELS_DIR,"distilled",task,tier))
out["teacher_correct"]=run("teacher", TEACHER[task], True)
json.dump(out, open(f"{RESULTS}/e11_calib_{task}.json","w"))
print(f"[E11] wrote {RESULTS}/e11_calib_{task}.json")
