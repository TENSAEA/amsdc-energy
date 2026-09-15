"""
E7 -- Where does the capacity signal live?

E4a showed surface features carry no information about which tier a query needs
(|r| < 0.03, including a capacity-unconstrained gradient-boosting ceiling).
This asks the complementary question: does the SMALLEST STUDENT'S OWN OUTPUT
carry that signal? If it does, capacity routing is not a query-classification
problem at all -- it requires executing a model, which turns single-shot routing
into a cascade and makes the router-model energy trade-off decisive.
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(__file__))
from models import MODELS_DIR, DATA_DIR, RESULTS, SEQ_LEN, STUDENTS
os.environ["HF_HOME"] = os.path.abspath(MODELS_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(6); torch.set_grad_enabled(False)

task = "sst2"
D = json.load(open(f"{RESULTS}/e2_distill_{task}.json"))
LADDER = list(STUDENTS.keys()) + ["teacher"]

def corr_vec(tier, split):
    if tier == "teacher": return np.array(D[f"teacher_{split}_correct"])
    return np.array(D["students"][tier][f"{split}_correct"])
C_test = np.stack([corr_vec(t,"test") for t in LADDER])
o_test = np.full(C_test.shape[1], len(LADDER)-1)
for k in range(len(LADDER)-1,-1,-1): o_test[C_test[k]==1]=k

@torch.no_grad()
def probs(tier, texts):
    d = os.path.join(MODELS_DIR, "distilled", task, tier)
    tok = AutoTokenizer.from_pretrained(d)
    m = AutoModelForSequenceClassification.from_pretrained(d).eval()
    out=[]
    for i in range(0,len(texts),64):
        e = tok(texts[i:i+64], padding="max_length", truncation=True,
                max_length=SEQ_LEN, return_tensors="pt")
        out.append(torch.softmax(m(**e).logits, -1))
    return torch.cat(out).numpy()

P = probs("s1", D["test_text"])
conf   = P.max(1)                                   # max softmax probability
margin = np.sort(P,1)[:,-1] - np.sort(P,1)[:,-2]    # top-1 minus top-2
ent    = -(P*np.log(P+1e-12)).sum(1)                # predictive entropy

print(f"\n{'='*72}\n  E7: does s1's own output predict required capacity?\n{'='*72}")
print(f"\n  {'signal':28s} {'r vs oracle tier':>17s}   {'|r|':>6s}")
for nm, v in [("s1 max-softmax confidence", conf), ("s1 top-2 margin", margin),
              ("s1 predictive entropy", ent)]:
    r = np.corrcoef(v, o_test)[0,1]
    print(f"  {nm:28s} {r:+17.4f}   {abs(r):6.4f}")
print(f"\n  for reference, best surface feature on TEST: |r| = 0.0253")

# how separable is "s1 is enough" from "escalate"?
need_more = (o_test > 0).astype(int)
from sklearn.metrics import roc_auc_score
print(f"\n  AUC for predicting 'escalation needed' (o_test > s1):")
for nm, v in [("s1 confidence", conf), ("s1 margin", margin), ("s1 entropy", ent)]:
    print(f"    {nm:16s} AUC = {roc_auc_score(need_more, -v if nm!='s1 entropy' else v):.4f}")

json.dump({"conf": conf.tolist(), "margin": margin.tolist(), "entropy": ent.tolist(),
           "oracle": o_test.tolist(),
           "r_conf": float(np.corrcoef(conf,o_test)[0,1]),
           "r_margin": float(np.corrcoef(margin,o_test)[0,1]),
           "r_entropy": float(np.corrcoef(ent,o_test)[0,1]),
           "auc_conf": float(roc_auc_score(need_more,-conf)),
           "auc_entropy": float(roc_auc_score(need_more,ent))},
          open(f"{RESULTS}/e7_signal_{task}.json","w"))
print(f"\n  wrote {RESULTS}/e7_signal_{task}.json")
