"""E17 -- did decorrelation training (E16) work?

Compares the mutual-learning fleet (beta=0.5 negative peer divergence) against the
plain-distillation fleet (E2) on the SAME 872 SST-2 queries:
  (a) per-student accuracy
  (b) pairwise error correlation vs independence  = P(both wrong)/(P(a wrong)P(b wrong))
  (c) collective ceiling (>=1 student correct) and 4-student majority vote
  (d) whether ANY agreement/unanimity/confidence config now reaches teacher accuracy
      at lower energy -- the goal the plain fleet failed to reach.
Energy charges every model executed.
"""
import os, sys, json, csv, itertools
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS, STUDENTS
import numpy as np

task="sst2"; TIERS=list(STUDENTS.keys())
E={r["label"]:float(r["mJ_per_query"]) for r in csv.DictReader(open(f"{RESULTS}/e1_energy.csv"))}

def load(path):
    D=json.load(open(path))
    C={t:np.array(D["students"][t]["test_correct"]) for t in TIERS}
    C["teacher"]=np.array(D["teacher_test_correct"])
    return D,C

def report(name,C):
    n=len(C["s1"]); T_A=C["teacher"].mean()
    print(f"\n{'='*66}\n{name}   (n={n}, teacher={T_A:.4f})\n{'='*66}")
    for t in TIERS: print(f"  {t:4s} acc={C[t].mean():.4f}   E={E[t]:8.1f} mJ")
    print(f"\n  pairwise error correlation (ratio vs independence):")
    print(f"    [paper definition: all pairs among students AND teacher]")
    ratios=[]
    for a,b in itertools.combinations(TIERS+["teacher"],2):
        wa,wb=1-C[a],1-C[b]
        pj=(wa&wb).mean(); pi=wa.mean()*wb.mean()
        r=pj/pi if pi>0 else float("nan"); ratios.append(r)
        print(f"    {a}-{b}: P(both wrong)={pj:.4f}  independent={pi:.4f}  ratio={r:.2f}x")
    print(f"    -> range {min(ratios):.2f}x - {max(ratios):.2f}x   (1.0 = independent)")
    M=np.stack([C[t] for t in TIERS]); sm=M.sum(0)
    ceil=(sm>0).mean()
    # paper's vote rule: majority of 4; a 2-2 tie is broken by the largest student (s4)
    vote=((sm>=3)|((sm==2)&(C["s4"]==1))).mean()
    allw=(sm==0).mean()
    print(f"\n  all four wrong      : {allw*100:.2f}% of queries")
    print(f"  collective ceiling  : {ceil:.4f}  (>=1 student correct)")
    print(f"  4-student vote       : {vote:.4f}")
    print(f"  lost to correlation : {(ceil-vote)*100:.2f} points")
    return dict(ratios=ratios, ceiling=float(ceil), vote=float(vote),
                accs={t:float(C[t].mean()) for t in TIERS}, teacher=float(T_A))

def search(C,label):
    """Every acceptance rule computable from correctness alone.

    Binary task: a model's prediction is determined by (label, correctness), so all
    agreement and voting rules are recoverable from the correctness matrix without
    needing stored logits. Confidence-gated rules are NOT included here -- E16 does
    not store confidences -- so this is a subset of the paper's 151-config search.
    """
    T_A=C["teacher"].mean(); T_E=E["teacher"]; cfgs=[]
    def add(nm,e,ok): cfgs.append((float(np.asarray(ok,float).mean()),
                                   float(np.asarray(e,float).mean()),nm))
    for k in (2,3,4):
        for combo in itertools.combinations(TIERS,k):
            M=np.stack([C[t] for t in combo]); base=sum(E[t] for t in combo); sm=M.sum(0)
            for tgt in ["s3","s4","teacher"]:
                if E[tgt]<=max(E[t] for t in combo): continue
                # (a) unanimity gate: accept only when all k agree
                un=(sm==k)|(sm==0)
                add(f"{'+'.join(combo)} unanimous ->{tgt}",
                    base+np.where(un,0.0,E[tgt]), np.where(un,M[0],C[tgt]))
                # (b) j-of-k agreement gate, j from majority up to k-1
                for j in range(k//2+1,k):
                    ag=(sm>=j)|(sm<=k-j)          # >=j agree on one side
                    maj=(sm>=j).astype(int)       # the agreed answer is correct iff >=j correct
                    add(f"{'+'.join(combo)} {j}-of-{k} ->{tgt}",
                        base+np.where(ag,0.0,E[tgt]), np.where(ag,maj,C[tgt]))
            # (c) plain vote, no escalation (largest member breaks ties)
            big=combo[-1]
            v=(sm>k/2)|((sm==k/2)&(C[big]==1)) if k%2==0 else (sm>k/2)
            add(f"{'+'.join(combo)} vote (no escalation)", np.full(len(sm),base), v)
    hits=[c for c in cfgs if c[0]>=T_A and c[1]<T_E]
    print(f"\n  configurations searched: {len(cfgs)}")
    print(f"  reaching teacher accuracy at lower energy: {len(hits)}")
    if hits:
        for a,e,nm in sorted(hits,key=lambda x:x[1])[:6]:
            print(f"    *** {nm:34s} acc={a:.4f} (teacher {T_A:.4f})  E={e:8.1f} mJ "
                  f"({100*(1-e/T_E):+.1f}% vs teacher)")
    else:
        b=max(cfgs,key=lambda x:x[0])
        print(f"    closest on accuracy: {b[2]:34s} acc={b[0]:.4f} "
              f"({100*(b[0]-T_A):+.2f} pts) at {b[1]:.1f} mJ")
        cheap=[c for c in cfgs if c[1]<T_E]
        bc=max(cheap,key=lambda x:x[0])
        print(f"    best under teacher energy: {bc[2]:28s} acc={bc[0]:.4f} "
              f"({100*(bc[0]-T_A):+.2f} pts) at {bc[1]:.1f} mJ ({100*(1-bc[1]/T_E):+.1f}%)")
    return hits

out={}
base_p=f"{RESULTS}/e2_distill_{task}.json"
Db,Cb=load(base_p); out["baseline"]=report("BASELINE  (plain distillation, E2)",Cb)
search(Cb,"baseline")

mut_p=f"{RESULTS}/e16_mutual_{task}.json"
if os.path.exists(mut_p):
    Dm,Cm=load(mut_p); out["mutual"]=report("MUTUAL LEARNING  (beta=0.5, E16)",Cm)
    hits=search(Cm,"mutual")
    print(f"\n{'='*66}\nVERDICT\n{'='*66}")
    rb,rm=out["baseline"]["ratios"],out["mutual"]["ratios"]
    print(f"  mean correlation ratio: {np.mean(rb):.2f}x -> {np.mean(rm):.2f}x "
          f"({'DOWN' if np.mean(rm)<np.mean(rb) else 'UP'})")
    print(f"  collective ceiling    : {out['baseline']['ceiling']:.4f} -> {out['mutual']['ceiling']:.4f}")
    print(f"  4-student vote        : {out['baseline']['vote']:.4f} -> {out['mutual']['vote']:.4f}")
    print(f"  mean student accuracy : {np.mean(list(out['baseline']['accs'].values())):.4f} -> "
          f"{np.mean(list(out['mutual']['accs'].values())):.4f}")
    print(f"  reached teacher cheaper: {'YES' if hits else 'NO'}")
    json.dump(out,open(f"{RESULTS}/e17_mutual_compare_{task}.json","w"),indent=1)
    print(f"\n[E17] wrote {RESULTS}/e17_mutual_compare_{task}.json")
else:
    print(f"\n[E17] {mut_p} not present yet -- baseline half only.")
