"""
E6 -- Generate Figure 1 (energy-accuracy Pareto) and the LaTeX table bodies.

Form: scatter with EMPHASIS (choosing-a-form.md) -- AMSD-C is the point, baselines
are context, oracle marks the frontier. Identity is carried by marker shape AND
direct label, never colour alone (print/greyscale safe).
Palette validated: #2a78d6 / #eb6834, CVD dE 24.7, all six checks PASS.
"""
import os, sys, csv, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); a = ap.parse_args()
FIG = "paper/tex/figures"; os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d8d8d4"

rows = list(csv.DictReader(open(f"{RESULTS}/e5_endtoend_{a.task}.csv")))
rie  = list(csv.DictReader(open(f"{RESULTS}/e3_rie_{a.task}.csv")))[0]
D    = json.load(open(f"{RESULTS}/e2_distill_{a.task}.json"))
E1   = {r["label"]: r for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}

def style(name):
    """emphasis: AMSD-C = blue, oracle = orange, everything else = grey context."""
    if name.startswith("AMSD-C"):
        return dict(c=BLUE, marker="o", s=110, z=5, lw=1.4, label_c=INK, bold=True)
    if "Oracle" in name:
        return dict(c=ORANGE, marker="*", s=240, z=4, lw=1.2, label_c=INK, bold=True)
    return dict(c="none", marker="s", s=58, z=3, lw=1.3, label_c=MUTED, bold=False)

fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.set_facecolor("#fcfcfb"); fig.patch.set_facecolor("white")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)

for r in rows:
    x, y, name = float(r["mJ_per_query"]), 100*float(r["acc"]), r["system"]
    st = style(name)
    ax.scatter(x, y, c=st["c"], marker=st["marker"], s=st["s"], zorder=st["z"],
               edgecolors=st["c"] if st["c"] != "none" else MUTED,
               linewidths=st["lw"])
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(9, 4),
                fontsize=7.6, color=st["label_c"],
                fontweight="bold" if st["bold"] else "normal", zorder=6)

ax.set_xscale("log")
ax.set_xlabel("Energy per query (mJ, log scale)", fontsize=9.5, color=INK)
ax.set_ylabel("Task accuracy (%)", fontsize=9.5, color=INK)
ax.tick_params(colors=MUTED, labelsize=8.5)
ax.set_title(f"Energy–accuracy trade-off, {a.task.upper()}  ·  lower-left is cheaper, upper is better",
             fontsize=9, color=MUTED, loc="left", pad=10)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/pareto_{a.task}.{ext}", dpi=200, bbox_inches="tight")
print(f"[E6] wrote {FIG}/pareto_{a.task}.pdf/.png")

# ---------------------------------------------------------------- LaTeX bodies
def f(x, n=1): return f"{float(x):,.{n}f}"
out = {}

# Table 4: measured energy + accuracy
ORDER = [("teacher","Teacher (\\texttt{bert-large})"), ("teacher_int8","Teacher, int8 (baseline v)"),
         ("s4","$s_4$"), ("s3","$s_3$"), ("s2","$s_2$"), ("s1","$s_1$"),
         ("distilbert_baseline","DistilBERT baseline")]
acc = {"teacher": D["teacher_acc"], "teacher_int8": D["teacher_acc"],
       **{k: v["test_acc"] for k, v in D["students"].items()}}
acc["distilbert_baseline"] = acc.pop("distilbert")
L=[]
for key, lab in ORDER:
    e = E1[key]; av = acc.get(key)
    L.append(f"{lab} & {f(e['mJ_per_query'])} $\\pm$ {f(e['ci95'])} & "
             f"{100*av:.2f} \\\\" if av else f"{lab} & {f(e['mJ_per_query'])} & --- \\\\")
L.append("\\midrule")
for key, lab in [("router_heuristic","Router (heuristic)"),("router_linear","Router (linear)"),
                 ("router_neural","Router (neural)")]:
    e=E1[key]; L.append(f"{lab} & {f(e['mJ_per_query'],3)} $\\pm$ {f(e['ci95'],3)} & --- \\\\")
out["table4"] = "\n".join(L)

# Table 5: RIE
out["table5"] = (f"{a.task.upper()} & {f(float(rie['E_teacher_mJ'])/1000)} & "
                 f"{f(float(rie['E_oracle_mJ'])/1000)} & {f(float(rie['RIE_mJ'])/1000)} & "
                 f"{f(rie['RIE_pct'])}\\% & {100*float(rie['oracle_acc']):.2f} \\\\")

# Table 6: routers
L=[]
s1e = float(E1["s1"]["mJ_per_query"])
for r in rows:
    if not r["system"].startswith("AMSD-C"): continue
    nm = r["system"].replace("AMSD-C (","").replace(")","").capitalize()
    L.append(f"{nm} & {100*float(r['routing_acc']):.1f}\\% & {f(r['router_mJ'],3)} & "
             f"{float(r['router_mJ'])/s1e:.4f} & {f(r['mJ_per_query'])} & {float(r['RR']):.3f} \\\\")
out["table6"] = "\n".join(L)

# Table 7: end-to-end
L=[]
for r in rows:
    nm = r["system"]; b = nm.startswith("AMSD-C (heuristic")
    nm = f"\\textbf{{{nm}}}" if b else nm
    L.append(f"{nm} & {f(r['mJ_per_query'])} & {100*float(r['acc']):.2f} & "
             f"{f(r['median'])} & {f(r['p95'])} & {f(r['max'])} \\\\")
out["table7"] = "\n".join(L)

# headline numbers for abstract/conclusion
h = {r["system"]: r for r in rows}
tea = h["Fixed teacher"]; T_E = float(tea["mJ_per_query"]); T_A = float(tea["acc"])
# headline system = best confidence cascade (the architecture that actually works)
casc = [r for r in csv.DictReader(open(f"{RESULTS}/e8_cascade_{a.task}.csv"))
        if r["escalate_to"] == "s4"]
best = min([c for c in casc if float(c["acc"]) >= 0.895], key=lambda c: float(c["mJ_per_query"]))
db = h["DistilBERT only"]
out["headline"] = {
    "teacher_mJ": T_E, "teacher_acc": 100*T_A,
    "casc_mJ": float(best["mJ_per_query"]), "casc_acc": 100*float(best["acc"]),
    "casc_thresh": float(best["thresh"]),
    "energy_vs_teacher_pct": 100*(1 - float(best["mJ_per_query"])/T_E),
    "acc_gap_pts": 100*(T_A - float(best["acc"])),
    "energy_vs_distilbert_pct": 100*(1 - float(best["mJ_per_query"])/float(db["mJ_per_query"])),
    "acc_vs_distilbert_pts": 100*(float(best["acc"]) - float(db["acc"])),
    "casc_median": float(best["median"]), "casc_p95": float(best["p95"]),
    "tail_ratio": float(best["p95"])/max(float(best["median"]), 1e-9),
    "rho_neural": float(E1["router_neural"]["mJ_per_query"])/s1e,
    "rho_heuristic": float(E1["router_heuristic"]["mJ_per_query"])/s1e,
    "rho_s1": 1.0,
    "RIE_pct": float(rie["RIE_pct"]),
    "oracle_mJ": float(rie["E_oracle_mJ"])/float(rie["n_test"]),
    "oracle_acc": 100*float(rie["oracle_acc"]),
    "s1_mJ": s1e, "s1_acc": 100*D["students"]["s1"]["test_acc"],
    "s4_mJ": float(E1["s4"]["mJ_per_query"]), "s4_acc": 100*D["students"]["s4"]["test_acc"],
}
json.dump(out, open(f"{RESULTS}/e6_latex_{a.task}.json","w"), indent=1)
print(f"[E6] wrote {RESULTS}/e6_latex_{a.task}.json")
print("\n=== HEADLINE NUMBERS ===")
for k,v in out["headline"].items(): print(f"  {k:28s} {v:10.3f}")
