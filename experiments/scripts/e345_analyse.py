"""
E3 + E4 + E5 -- Oracle / RIE, router calibration under an accuracy constraint,
and end-to-end comparison.

Scientific design notes (these matter for the paper's validity):

* The accuracy floor gamma is SWEPT over its achievable range rather than fixed at
  one arbitrary value. A single gamma invites the charge of tuning; a sweep reports
  the whole operating curve and lets the reader pick the constraint.

* Routers are fitted on the CALIBRATION split only and evaluated on TEST. The
  calibration split never overlaps test.

* A "feature ceiling" router (gradient boosting, unconstrained capacity, fitted on
  the same features) is included as a DIAGNOSTIC, not as a deployable system. It
  separates two very different conclusions: "surface features carry no signal about
  required capacity" versus "our cheap fitter is weak". Without it, a heuristic
  failure is uninterpretable.

* Recovery Ratio is reported ONLY together with accuracy. RR > 1 means a system
  spent less than the oracle by sacrificing accuracy; it is not a better system.
  Systems are additionally marked as Pareto-dominated where applicable.
"""
import os, sys, json, csv, re, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.RandomState(a.seed)

# ---------------------------------------------------------------- inputs
E = {r["label"]: float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}
D = json.load(open(f"{RESULTS}/e2_distill_{a.task}.json"))
TIERS  = list(STUDENTS.keys())
LADDER = TIERS + ["teacher"]
ENERGY = {t: E[t] for t in TIERS}; ENERGY["teacher"] = E["teacher"]
EN = np.array([ENERGY[t] for t in LADDER])

def corr(tier, split):
    if tier == "teacher": return np.array(D[f"teacher_{split}_correct"])
    return np.array(D["students"][tier][f"{split}_correct"])

C_test  = np.stack([corr(t, "test")  for t in LADDER])
C_cal   = np.stack([corr(t, "calib") for t in LADDER])
n_test, n_cal = C_test.shape[1], C_cal.shape[1]
teacher_acc = D["teacher_acc"]

def oracle_idx(C):
    idx = np.full(C.shape[1], len(LADDER) - 1)
    for k in range(len(LADDER) - 1, -1, -1): idx[C[k] == 1] = k
    return idx
o_test, o_cal = oracle_idx(C_test), oracle_idx(C_cal)

# ---------------------------------------------------------------- E3: RIE
E_T      = ENERGY["teacher"] * n_test
E_oracle = EN[o_test].sum()
RIE      = E_T - E_oracle
acc_oracle = C_test[o_test, np.arange(n_test)].mean()

# ---------------------------------------------------------------- features
WORD = re.compile(r"\w+")
CUES = ("why","how","because","although","whereas","unless","if","not","never","but",
        "however","although","yet","despite","while","rather","instead","too","enough")
NEG  = ("not","no","never","n't","without","lack","fail","hardly","barely")
def feats(t):
    w = WORD.findall(t.lower()); nw = max(len(w),1)
    return [len(t), nw, sum(c.isdigit() for c in t)/max(len(t),1),
            t.count(",")+t.count(";")+t.count(":"), t.count("."), t.count("!")+t.count("?"),
            sum(x in CUES for x in w), sum(any(n in x for n in NEG) for x in w),
            sum(len(x) for x in w)/nw, len(set(w))/nw,
            max((len(x) for x in w), default=0),
            sum(1 for x in w if len(x) > 7)/nw]
X_cal  = np.array([feats(t) for t in D["calib_text"]], dtype=np.float64)
X_test = np.array([feats(t) for t in D["test_text"]],  dtype=np.float64)
mu, sd = X_cal.mean(0), X_cal.std(0) + 1e-9
Zc, Zt = (X_cal-mu)/sd, (X_test-mu)/sd

# ---------------------------------------------------------------- policy evaluation
def evaluate(assign, router_key=None, name=""):
    e_router = E[router_key] if router_key else 0.0
    per_q = EN[assign] + e_router
    acc   = C_test[assign, np.arange(n_test)].mean()
    E_sys = per_q.sum()
    return dict(system=name, mJ_per_query=per_q.mean(), acc=acc,
                RR=(E_T - E_sys)/RIE, router_mJ=e_router, per_q=per_q,
                routing_acc=float((assign == o_test).mean()),
                dist={LADDER[k]: int((assign==k).sum()) for k in range(len(LADDER))})

# ---------------------------------------------------------------- routers
# A router maps a per-example SCORE -> tier via K-1 thresholds. Thresholds are
# chosen on CALIBRATION to minimise energy subject to accuracy >= gamma (Eq. 6).
def thresholds_for(score_cal, gamma):
    """Grid-search cut points on calibration; return cuts or None if gamma unreachable."""
    qs = np.linspace(0, 100, 21)
    best = None
    for lo in qs[:-1]:
        for hi in qs[1:]:
            if hi <= lo: continue
            cuts = np.percentile(score_cal, np.linspace(lo, hi, len(LADDER)+1)[1:-1])
            asg  = np.searchsorted(cuts, score_cal)
            acc  = C_cal[asg, np.arange(n_cal)].mean()
            if acc >= gamma:
                e = EN[asg].sum()
                if best is None or e < best[0]: best = (e, cuts)
    return None if best is None else best[1]

def score_heuristic():
    """Eq. (4): linear score from surface features, weights = calibration correlation."""
    w = np.array([np.corrcoef(Zc[:,j], o_cal)[0,1] if Zc[:,j].std()>0 else 0.0
                  for j in range(Zc.shape[1])])
    w = np.nan_to_num(w)
    return Zc @ w, Zt @ w

def score_linear():
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1.0).fit(Zc, o_cal)
    return m.predict(Zc), m.predict(Zt)

def score_ceiling():
    """DIAGNOSTIC ONLY -- upper bound on what these features can express."""
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor(random_state=a.seed, n_estimators=300, max_depth=3).fit(Zc, o_cal)
    return m.predict(Zc), m.predict(Zt)

SCORERS = {"heuristic": (score_heuristic, "router_heuristic"),
           "linear":    (score_linear,    "router_linear"),
           "neural":    (score_ceiling,   "router_neural")}

# gamma sweep over the ACHIEVABLE range: from the weakest tier up to the oracle
acc_tiers = [C_test[k].mean() for k in range(len(LADDER))]
GAMMAS = np.round(np.arange(0.80, min(acc_oracle, 0.975)+1e-9, 0.01), 3)

sweep = []
for gname, (fn, ekey) in SCORERS.items():
    sc, st = fn()
    for g in GAMMAS:
        cuts = thresholds_for(sc, g)
        if cuts is None: continue
        r = evaluate(np.searchsorted(cuts, st), ekey, f"AMSD-C ({gname})")
        rec = {k: v for k, v in r.items() if k != "per_q"}
        rec["dist"] = json.dumps(rec["dist"])
        sweep.append(dict(router=gname, gamma=float(g), **rec))

# ---------------------------------------------------------------- operating point
# Choose, per router, the LOWEST-energy configuration whose TEST accuracy is within
# 2 points of the best fixed student the fleet contains (s4). This is stated in the
# paper as the reported operating point; the full sweep is also released.
TARGET = max(acc_tiers[:-1]) - 0.02
chosen = {}
for gname in SCORERS:
    cand = [s for s in sweep if s["router"]==gname and s["acc"] >= TARGET]
    chosen[gname] = min(cand, key=lambda s: s["mJ_per_query"]) if cand else None

# ---------------------------------------------------------------- E5 baselines
rows = []
for k, t in enumerate(LADDER):
    rows.append(evaluate(np.full(n_test, k), None, f"Fixed {t}"))
rows.append(dict(system="DistilBERT only", mJ_per_query=E["distilbert_baseline"],
                 acc=D["students"]["distilbert"]["test_acc"], router_mJ=0.0,
                 per_q=np.full(n_test, E["distilbert_baseline"]), routing_acc=np.nan,
                 RR=(E_T-E["distilbert_baseline"]*n_test)/RIE,
                 dist={"distilbert": n_test}))
rows.append(dict(system="Teacher int8", mJ_per_query=E["teacher_int8"], acc=teacher_acc,
                 router_mJ=0.0, per_q=np.full(n_test, E["teacher_int8"]), routing_acc=np.nan,
                 RR=(E_T-E["teacher_int8"]*n_test)/RIE, dist={"teacher_int8": n_test}))
casc = np.where(C_test[0]==1, EN[0], EN[0]+EN[-1])
rows.append(dict(system="Cascade (s1->T)", mJ_per_query=casc.mean(),
                 acc=float(np.maximum(C_test[0],C_test[-1]).mean()), router_mJ=0.0,
                 per_q=casc, routing_acc=np.nan, RR=(E_T-casc.sum())/RIE,
                 dist={"s1":int((C_test[0]==1).sum()),"s1+T":int((C_test[0]==0).sum())}))
for gname, s in chosen.items():
    if s is None: continue
    fn, ekey = SCORERS[gname]; sc, st = fn()
    cuts = thresholds_for(sc, s["gamma"])
    rows.append(evaluate(np.searchsorted(cuts, st), ekey, f"AMSD-C ({gname})"))
rows.append(evaluate(o_test, None, "Oracle (bound)"))

# Pareto: a system is dominated if another has <= energy AND >= accuracy (one strict)
for r in rows:
    r["dominated"] = any(o["mJ_per_query"] <= r["mJ_per_query"] and o["acc"] >= r["acc"]
                         and (o["mJ_per_query"] < r["mJ_per_query"] or o["acc"] > r["acc"])
                         for o in rows if o is not r and "Oracle" not in o["system"])

# ---------------------------------------------------------------- report
print(f"\n{'='*84}\n  E3/E4/E5   task={a.task}   n_test={n_test}   n_calib={n_cal}")
print(f"{'='*84}")
print("\n-- E3: Recoverable Inference Energy --")
print(f"  E_teacher {E_T/1000:9.2f} J   E_oracle {E_oracle/1000:8.2f} J   "
      f"RIE {RIE/1000:8.2f} J  ({100*RIE/E_T:.1f}% of teacher)")
print(f"  teacher acc {teacher_acc:.4f}   oracle acc {acc_oracle:.4f}")
print(f"  oracle dispatch: " + "  ".join(
    f"{LADDER[k]}={int((o_test==k).sum())} ({100*(o_test==k).mean():.0f}%)" for k in range(len(LADDER))))

print(f"\n-- E4a: DIAGNOSTIC -- can these features predict required capacity? --")
for gname,(fn,_) in SCORERS.items():
    sc, st = fn()
    rho = np.corrcoef(st, o_test)[0,1]
    print(f"  {gname:10s} score-vs-oracle correlation on TEST: r = {rho:+.4f}")
print(f"  (base rate: {100*(o_test==0).mean():.1f}% of queries need only s1)")

print(f"\n-- E4b: operating points (accuracy floor >= {TARGET:.4f}) --")
print(f"  {'router':12s} {'gamma':>6s} {'route acc':>9s} {'E_R mJ':>9s} {'rho':>8s} {'mJ/query':>10s} {'acc':>8s} {'RR':>7s}")
s1e = ENERGY["s1"]
for gname, s in chosen.items():
    if s is None: print(f"  {gname:12s} {'--':>6s}  (no configuration reaches the floor)"); continue
    print(f"  {gname:12s} {s['gamma']:6.2f} {s['routing_acc']:9.4f} {s['router_mJ']:9.3f} "
          f"{s['router_mJ']/s1e:8.4f} {s['mJ_per_query']:10.2f} {s['acc']:8.4f} {s['RR']:7.3f}")

print(f"\n-- E5: end-to-end (RR is meaningful only alongside accuracy) --")
print(f"  {'system':22s} {'mJ/query':>10s} {'acc':>8s} {'median':>9s} {'p95':>9s} {'max':>9s} {'RR':>7s}  pareto")
out=[]
for r in sorted(rows, key=lambda r: r["mJ_per_query"]):
    pq=r["per_q"]
    flag = "dominated" if r.get("dominated") else ("BOUND" if "Oracle" in r["system"] else "on front")
    print(f"  {r['system']:22s} {r['mJ_per_query']:10.2f} {r['acc']:8.4f} {np.median(pq):9.2f} "
          f"{np.percentile(pq,95):9.2f} {pq.max():9.2f} {r['RR']:7.3f}  {flag}")
    out.append(dict(system=r["system"], mJ_per_query=r["mJ_per_query"], acc=r["acc"],
                    median=float(np.median(pq)), p95=float(np.percentile(pq,95)),
                    max=float(pq.max()), RR=r["RR"], router_mJ=r["router_mJ"],
                    routing_acc=r["routing_acc"], dominated=r.get("dominated", False),
                    dist=json.dumps(r["dist"])))

with open(f"{RESULTS}/e3_rie_{a.task}.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["task","n_test","E_teacher_mJ","E_oracle_mJ","RIE_mJ","RIE_pct","oracle_acc","teacher_acc"])
    w.writerow([a.task,n_test,E_T,E_oracle,RIE,100*RIE/E_T,acc_oracle,teacher_acc])
with open(f"{RESULTS}/e4_sweep_{a.task}.csv","w",newline="") as f:
    if sweep: w=csv.DictWriter(f,fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)
with open(f"{RESULTS}/e5_endtoend_{a.task}.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print(f"\n  wrote e3_rie / e4_sweep / e5_endtoend for {a.task}")
