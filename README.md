# AMSD-C: Adaptive Multi-Student Knowledge Distillation by Capacity

Measurement code, experiment scripts and raw results for

> **Adaptive Multi-Student Knowledge Distillation by Capacity for Energy-Efficient LLM Inference in Resource-Constrained Settings**
> Tensae Aschalew, Beakal Gizachew — Addis Ababa University
> PanAfriCon AI 2026 (to appear)
>
> MSc thesis work by Tensae Aschalew, supervised by Dr. Beakal Gizachew.

A single teacher is distilled into a fleet of students that differ only in capacity.
The smallest student answers first; its own confidence decides whether to accept or
escalate. Every model, and every router, is charged in measured joules.

## Main results (SST-2, 872 validation prompts, CPU, RAPL counters)

| quantity | value |
|---|---|
| teacher energy per prompt (bert-large, 335M) | 10,020.7 mJ at 93.46% |
| recoverable inference energy (oracle over the fleet) | 91.3% — 869.3 mJ at 97.48% |
| best query-only predictor of required capacity | AUC 0.605 |
| smallest student's own confidence | AUC 0.784 |
| router-overhead ratio ρ, DistilBERT-scale scorer | 5.13× the model it protects |
| efficiency point: s1→s4 cascade, t = 0.98 | −87.3% energy, −3.78 pts |
| matching point: s2→teacher, threshold calibrated on 654 prompts | −23.8% energy, −0.07 ± 0.09 pts |
| re-distilling s2 on 2.8× more data | accuracy ↑, ceiling ↑, deployed saving ↓ (23.8 → 20.5%) |
| complement-aware distillation (loss changed on 1.7% of data) | saving 20.5 → 28.2% |

Every number in the paper is produced by a script in `experiments/scripts/` from a file
in `experiments/results/`.

## Layout

```
energy/meter.py            RAPL energy reader (package + DRAM, wrap-around corrected)
experiments/scripts/       e1 … e28, one script per experiment; models.py defines the fleet
experiments/results/       CSV/JSON outputs every table and figure is built from
paper/tex/figures/         the three figures
requirements.txt           Python dependencies (CPU measurement environment)
```

## Reproducing

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python experiments/scripts/download.py          # fetches public checkpoints and GLUE
python experiments/scripts/e1_profile.py        # energy profile  (needs /sys/class/powercap RAPL)
python experiments/scripts/e2_distill.py        # distil the fleet
python experiments/scripts/e25_table_match.py   # teacher-matching table (Table 11)
python experiments/scripts/e23_opposition.py    # better-student / worse-cascade (Table 13)
python experiments/scripts/e27_compare_students.py sst2 \
    plain20k=experiments/results/e21_d20k_sst2.json \
    cad_g1=experiments/results/e21_cad_g1_sst2.json   # complement-aware distillation (Table 14)
```

Energy measurement requires an Intel CPU exposing RAPL counters and read access to
`/sys/class/powercap/intel-rapl`. Measurements in the paper were taken on a quiet machine
with a 60 s idle baseline, 100 discarded warm-up queries, 1000-query loops, five trials,
randomised model order and 60 s cool-downs between trials (paper, Section 4.5).

The GPU scripts (`e21_full_distill.py` and later) were run in a separate environment with
`torch==2.5.1+cu121`; energy was never measured on the GPU.

## Models and data

Teacher: `yoshitomo-matsubara/bert-large-uncased-sst2` (and `-qnli`, `-mnli`).
Students: `google/bert_uncased_L-4_H-256_A-4`, `L-4_H-512_A-8`, `L-6_H-768_A-12`,
`bert-base-uncased`. Baseline: `distilbert-base-uncased`. Data: GLUE via
`nyu-mll/glue` on the Hugging Face Hub. All public; none are redistributed here.

## Citation

```bibtex
@inproceedings{aschalew2026amsdc,
  title     = {Adaptive Multi-Student Knowledge Distillation by Capacity for
               Energy-Efficient {LLM} Inference in Resource-Constrained Settings},
  author    = {Aschalew, Tensae and Gizachew, Beakal},
  booktitle = {PanAfriCon AI 2026},
  year      = {2026}
}
```

## License

MIT. See `LICENSE`.
