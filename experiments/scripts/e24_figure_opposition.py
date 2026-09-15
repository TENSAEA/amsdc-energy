"""E24 -- Figure: why the better-distilled student makes a worse cascade.

x: acceptance rate (top-k% by the student's own confidence)
y: student correct minus teacher correct, on that accepted set
A cascade can only match the teacher where this curve is >= 0.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from models import RESULTS
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size":8,"font.family":"serif","axes.linewidth":0.6,
                     "xtick.major.width":0.6,"ytick.major.width":0.6})
task="sst2"
A=json.load(open(f"{RESULTS}/e2_distill_{task}.json")); B=json.load(open(f"{RESULTS}/e21_d20k_{task}.json"))
CFa=np.array(json.load(open(f"{RESULTS}/e19_conf_{task}.json"))["s2"])
Ca=np.array(A["students"]["s2"]["test_correct"]); CT=np.array(A["teacher_test_correct"])
Cb=np.array(B["students"]["s2"]["test_correct"]); CFb=np.array(B["students"]["s2"]["conf"])
n=len(CT); ks=np.arange(5,101,1)

def curve(C,CF):
    o=np.argsort(-CF); return np.array([C[o[:int(n*k/100)]].sum()-CT[o[:int(n*k/100)]].sum() for k in ks])
ya,yb=curve(Ca,CFa),curve(Cb,CFb)
def xzero(y):
    i=np.where(y<0)[0]
    return ks[i[0]] if len(i) else 100

fig,ax=plt.subplots(figsize=(3.3,2.5),dpi=300)
ax.axhline(0,color="0.35",lw=0.7,zorder=1)
ax.plot(ks,ya,color="#1b6ca8",lw=1.6,ls="-", label=r"$s_2$ on 7,200",zorder=3)
ax.plot(ks,yb,color="#c2570a",lw=1.6,ls="--",label=r"$s_2$ on 20,000",zorder=3)
for y,c,m,lab,tx,ty in ((ya,"#1b6ca8","o","39%",30,-46),(yb,"#c2570a","s","22%",7,-26)):
    x=xzero(y); yy=y[list(ks).index(x)]
    ax.plot([x],[yy],m,color=c,ms=4.5,mec="white",mew=0.7,zorder=5)
    ax.annotate(f"leaves at {lab}",xy=(x,yy),xytext=(tx,ty),fontsize=6.5,color=c,
                ha="left",va="center",
                arrowprops=dict(arrowstyle="-",color=c,lw=0.5,shrinkA=2,shrinkB=3))
ax.axhspan(0,16,color="#2e7d32",alpha=0.055,zorder=0)
ax.annotate("matching region: student $\\geq$ teacher",xy=(7,10),fontsize=6.5,color="#2e7d32",ha="left")
ax.set_xlabel("acceptance rate (top-$k$% by student confidence)")
ax.set_ylabel("student $-$ teacher (queries correct)")
ax.set_xlim(5,100); ax.set_ylim(-90,17)
ax.legend(frameon=False,loc="lower left",fontsize=7)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.grid(axis="y",color="0.9",lw=0.5,zorder=0)
fig.tight_layout(pad=0.3)
out="paper/tex/figures/opposition_sst2"
fig.savefig(out+".pdf"); fig.savefig(out+".png")
print(f"zero-crossing: 7,200 -> {xzero(ya)}%   20,000 -> {xzero(yb)}%")
print(f"wrote {out}.pdf/.png")
