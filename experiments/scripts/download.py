"""Download every model and dataset once. Safe to re-run."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from models import TEACHER, STUDENTS, BASELINE, TASKS, MODELS_DIR, DATA_DIR

os.environ["HF_HOME"] = os.path.abspath(MODELS_DIR)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from datasets import load_dataset

todo = list(TEACHER.values()) + list(STUDENTS.values()) + [BASELINE]
for i, name in enumerate(dict.fromkeys(todo), 1):
    print(f"[{i}/{len(set(todo))}] {name}", flush=True)
    AutoTokenizer.from_pretrained(name)
    if name in TEACHER.values():
        AutoModelForSequenceClassification.from_pretrained(name)
    else:
        AutoModel.from_pretrained(name)

for t in TASKS:
    print(f"[data] glue/{t}", flush=True)
    ds = load_dataset("nyu-mll/glue", t, cache_dir=DATA_DIR)
    print({k: len(v) for k, v in ds.items()})
print("DONE")
