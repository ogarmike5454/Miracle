"""Figures for the revised paper. Usage: python -I make_figures.py <results_dir> <out_dir>"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import load, vectorizer, models, TEXT_VARIANTS
from sklearn.base import clone

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
BUG, NON = "#2a78d6", "#eb6834"
INK, MUTED = "#000000", "#333333"
plt.rcParams.update({"font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.9, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK, "text.color": INK})

# ---- Fig. 1: LR target score distributions under transfer ----
df = load(RES.parent / "data" / "all_repos_combined.csv")
panels = [("eclipse", "kubernetes", "combined", "(a) E→K, full"),
          ("kubernetes", "eclipse", "combined", "(b) K→E, full"),
          ("kubernetes", "eclipse", "combined_no_template", "(c) K→E, stripped")]
fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.6), sharey=False)
bins = np.linspace(0, 1, 26)
for ax, (src, tgt, var, title) in zip(axes, panels):
    S, T = df[df.repo == src], df[df.repo == tgt]
    vec = vectorizer(3); Xs = vec.fit_transform(S[TEXT_VARIANTS[var]]); Xt = vec.transform(T[TEXT_VARIANTS[var]])
    lr = clone(models()["LR"]).fit(Xs, S.is_bug); p = lr.predict_proba(Xt)[:, 1]
    for cls, col, lab in [(0, NON, "Non-bug"), (1, BUG, "Bug")]:
        w = np.ones((T.is_bug == cls).sum()) / len(T)
        ax.hist(p[T.is_bug.values == cls], bins=bins, weights=w * 100, color=col, alpha=0.6,
                edgecolor="white", linewidth=0.4, label=lab, hatch="////" if cls == 0 else None)
    ax.axvline(0.5, color=INK, lw=0.8, ls="--")
    ax.set_title(title, pad=2, fontsize=6.5); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e6e6", lw=0.4)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Target reports (%)")
axes[1].set_xlabel("Predicted bug probability (LR)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="upper center", ncol=2, handlelength=1.2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.9))
fig.savefig(OUT / "fig1_transfer_scores.png", dpi=400, bbox_inches="tight", pad_inches=0.02)

# ---- Fig. 2: macro-F1 by evaluation setting, full text vs templates removed ----
cv = pd.read_csv(RES / "cv_pooled_folds.csv"); wr = pd.read_csv(RES / "cv_within_repo_folds.csv")
tr = pd.read_csv(RES / "transfer_metrics.csv")
rows = []
for var in ["combined", "combined_no_template"]:
    c = cv[(cv.variant == var) & (cv.model == "LR")]
    rows += [("Pooled CV, Kubernetes part", var, c.macro_f1_kubernetes.mean(), c.macro_f1_kubernetes.std()),
             ("Pooled CV, Eclipse part", var, c.macro_f1_eclipse.mean(), c.macro_f1_eclipse.std())]
    for repo, name in [("kubernetes", "Within Kubernetes"), ("eclipse", "Within Eclipse")]:
        w = wr[(wr.repo == repo) & (wr.variant == var) & (wr.model == "LR")]
        rows.append((name, var, w.macro_f1.mean(), w.macro_f1.std()))
    for src, name in [("kubernetes", "Transfer K→E"), ("eclipse", "Transfer E→K")]:
        t = tr[(tr.source == src) & (tr.variant == var) & (tr.model == "LR") & (tr.threshold == "default")]
        rows.append((name, var, float(t.macro_f1.iloc[0]), 0.0))
d = pd.DataFrame(rows, columns=["setting", "variant", "mean", "sd"])
order = ["Pooled CV, Kubernetes part", "Within Kubernetes", "Pooled CV, Eclipse part", "Within Eclipse",
         "Transfer K→E", "Transfer E→K"]
fig, ax = plt.subplots(figsize=(3.5, 1.85))
y = np.arange(len(order))[::-1]
for var, col, mk, lab, off in [("combined", BUG, "o", "Full text", 0.12), ("combined_no_template", NON, "s", "Templates removed", -0.12)]:
    s = d[d.variant == var].set_index("setting").loc[order]
    ax.errorbar(s["mean"], y + off, xerr=s["sd"], fmt=mk, color=col, ms=4, lw=0.8, capsize=1.5, label=lab)
ax.axvline(1/3, color=MUTED, lw=0.6, ls=":"); ax.text(0.335, y[0] + 0.45, "single-class\nlevel", fontsize=5.5, color=MUTED, va="top")
ax.set_yticks(y); ax.set_yticklabels(order); ax.set_xlim(0.3, 1.0); ax.set_xlabel("Macro-F1 (LR)")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e6e6e6", lw=0.4); ax.set_axisbelow(True)
fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.62, 1.0), handletextpad=0.3)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
fig.savefig(OUT / "fig2_settings_macro_f1.png", dpi=400)
print(d.round(4).to_string())
