"""Fleet definition shared by every experiment script. Edit here only."""

# Teacher: published bert-large fine-tuned per task (Matsubara, torchdistill).
TEACHER = {
    "sst2": "yoshitomo-matsubara/bert-large-uncased-sst2",
    "rte":  "yoshitomo-matsubara/bert-large-uncased-rte",
    "qnli": "yoshitomo-matsubara/bert-large-uncased-qnli",
    "mnli": "yoshitomo-matsubara/bert-large-uncased-mnli",
}

# Students: Turc et al. miniatures (pre-trained, NOT task-tuned). We distil them.
# Ordered smallest -> largest. Keys are the tier names used in the paper.
STUDENTS = {
    "s1": "google/bert_uncased_L-4_H-256_A-4",    # L=4  H=256   ~11M
    "s2": "google/bert_uncased_L-4_H-512_A-8",    # L=4  H=512   ~29M
    "s3": "google/bert_uncased_L-6_H-768_A-12",   # L=6  H=768   ~67M
    "s4": "bert-base-uncased",                    # L=12 H=768   ~110M
}

# Single-student baseline named in the abstract.
BASELINE = "distilbert-base-uncased"

# Neural router: DistilBERT-scale, as in RouteNLP. Same checkpoint as baseline
# so its cost is directly comparable to a fleet member.
NEURAL_ROUTER = "distilbert-base-uncased"

TASKS = ["sst2", "rte", "qnli", "mnli"]

# Fixed for every model and router. State these in the paper.
SEQ_LEN   = 128
BATCH     = 1
DTYPE     = "float32"
THREADS   = 6

MODELS_DIR = "experiments/models"
DATA_DIR   = "experiments/data"
RESULTS    = "experiments/results"
