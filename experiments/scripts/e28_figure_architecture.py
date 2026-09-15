"""E28 -- Architecture figure for the paper. Every number is measured (e1/e2/e8/e25)."""
import os, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
plt.rcParams.update({"font.size":7.2,"font.family":"serif"})

BLUE=["#9ecae1","#6baed6","#3182bd","#08519c"]; TEA="#4a4a4a"; ORG="#c2570a"; GRN="#2e7d32"; INK="0.15"
fig=plt.figure(figsize=(7.0,4.6),dpi=300); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")

def box(x,y,w,h,fc,txt,fs=7.2,ec=INK,lw=0.7,tc=INK,bold=False,r=1.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,zorder=2))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,zorder=3,
            fontweight="bold" if bold else "normal",linespacing=1.25)
def arrow(x0,y0,x1,y1,c=INK,lw=0.9,ls="-",label=None,lx=0,ly=0,fs=6.4,lc=None):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=8,color=c,lw=lw,ls=ls,zorder=1,shrinkA=1,shrinkB=1))
    if label: ax.text((x0+x1)/2+lx,(y0+y1)/2+ly,label,fontsize=fs,color=lc or c,ha="center",va="center",zorder=3)

# ---------- panel (a): fleet construction ----------
ax.text(2,96.5,"(a)  Fleet: one teacher, one dataset, one loss; only capacity varies",fontsize=7.0,fontweight="bold",color=INK)
box(14,82,30,9.5,"#e6e6e6","Teacher  $T$\nbert-large, 335M\n10,020.7 mJ / query    93.46%",fs=6.8,bold=False)
S=[("$s_1$","11M","298.9 mJ","80.85%",11),("$s_2$","29M","699.4 mJ","84.06%",11),("$s_3$","67M","1,500.7 mJ","88.53%",11),("$s_4$","110M","2,915.6 mJ","91.40%",11)]
xs=[3,15,27,39]
for (nm,p,e,a,h),x,c in zip(S,xs,BLUE):
    box(x,57,11,h,c,f"{nm}\n{p}\n{e}\n{a}",fs=6.2,tc="white" if c in BLUE[2:] else INK)
    arrow(29,82,x+5.5,68,c="0.45",lw=0.7)
ax.text(1.5,75,"distil each:\n$\\alpha\\,T^2\\,\\mathrm{KL}(T\\|s)$\n$+(1-\\alpha)\\,\\mathrm{CE}$",fontsize=5.9,ha="left",va="center",color="0.35",linespacing=1.3)
ax.text(27,53.3,"energy spread 33.5$\\times$   (RAPL, CPU, seq 128, batch 1, fp32)",fontsize=6.0,ha="center",color="0.35")

# ---------- panel (b): dispatch ----------
ax.text(55,96.5,"(b)  Dispatch: the smallest student is its own router",fontsize=7.0,fontweight="bold",color=INK)
box(56,84,12,7,"white","query $q$",fs=7)
arrow(68,87.5,71.5,87.5)
box(71.5,82.5,17,10,BLUE[1],"$s_1$ forward pass\n$\\to$ answer $\\hat y$\n$\\to$ confidence $c$",fs=6.4)
arrow(80,82.5,80,76.5)
# diamond gate
ax.add_patch(Polygon([[80,76],[87,70.5],[80,65],[73,70.5]],closed=True,fc="#fff3e0",ec=ORG,lw=0.9,zorder=2))
ax.text(80,70.5,"$c \\geq t$ ?",fontsize=7,ha="center",va="center",color=ORG,zorder=3)
arrow(73,70.5,64,70.5,c=GRN,label="yes",ly=1.8,fs=6.4)
box(52,66,12,9,"#e8f5e9","accept $\\hat y$\n$E=298.9$ mJ",fs=6.4,ec=GRN,tc=GRN)
ax.plot([87,92],[70.5,70.5],color=ORG,lw=0.9,zorder=1); ax.text(89.5,72.3,"no",fontsize=6.4,color=ORG,ha="center")
arrow(92,70.5,92,58.2,c=ORG,label="escalate",lx=3.6,ly=0,fs=6.0)
box(85,47,14,11,BLUE[3],"target\n$s_4$ (2,915.6 mJ)\nor $T$ (10,020.7 mJ)",fs=6.0,tc="white")
ax.text(76,42.5,"$E(q)=E(s_1)+\\mathbb{1}[c<t]\\,E(\\mathrm{target})$",fontsize=6.4,ha="center",color=INK)
# rho note
box(54,46,29,14,"#fafafa","No separate router.  $\\rho = E_R / E(s_1)$:\n"
    "DistilBERT scorer   $\\rho=5.13$   $\\times$  (never pays)\n"
    "bert-tiny router      $\\rho=0.39$   AUC 0.605\n"
    "$s_1$'s own $c$           $\\rho=0$        AUC 0.784  $\\checkmark$",fs=6.0,ec="0.6",lw=0.6,r=0.8)

# ---------- bottom strip: measured operating points ----------
ax.add_patch(FancyBboxPatch((2,4),96,33,boxstyle="round,pad=0,rounding_size=1.2",fc="#f5f5f5",ec="0.75",lw=0.6,zorder=0))
ax.text(4,33.5,"(c)  Two measured operating points on the identical 872 SST-2 prompts",fontsize=7.6,fontweight="bold",color=INK)
box(5,15,42,15,"white","Efficiency:  $s_1 \\to s_4$,  $t=0.98$\n\n$\\mathbf{-87.3\\%}$ energy  (1,275.2 mJ)\n89.68% accuracy  ($-3.78$ pts vs $T$)",fs=6.8,ec=BLUE[3])
box(53,15,42,15,"white","Matching:  $s_2 \\to T$,  $t$ calibrated on $m{=}654$ held-out\n\n$\\mathbf{-23.8\\%}$ energy\n$\\mathbf{-0.07 \\pm 0.09}$ pts, indistinguishable from $T$",fs=6.8,ec=GRN)
ax.text(50,10,"Oracle over the fleet (perfect gate):  869.3 mJ = 8.7% of $T$  at  97.48%   $\\Rightarrow$   91.3% of $T$'s energy is recoverable",
        fontsize=6.6,ha="center",color=INK)
ax.text(50,6.3,"Ensemble of all four: 5,414.6 mJ = 54% of $T$ before any escalation, errors 3.2–7.6$\\times$ correlated  $\\Rightarrow$  closed",
        fontsize=6.2,ha="center",color="0.4")
out="paper/tex/figures/architecture"; fig.savefig(out+".pdf"); fig.savefig(out+".png"); print("wrote",out)
