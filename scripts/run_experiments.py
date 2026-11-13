"""Revised experiments for the bug-report classification paper.

Usage: python -I run_experiments.py <data.csv> <out_dir>
Every number reported in the paper is written to <out_dir> by this script.
"""
import sys, json, re, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, matthews_corrcoef)
from sklearn.feature_selection import chi2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))
from common import load, vectorizer, models, scores, TEXT_VARIANTS, SEED

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
R = {}  # all scalar results
df = load(DATA)
y_all = df["is_bug"].to_numpy()


def metrics(y, pred, s=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = dict(accuracy=accuracy_score(y, pred),
             bug_precision=precision_score(y, pred, zero_division=0),
             bug_recall=recall_score(y, pred, zero_division=0),
             bug_f1=f1_score(y, pred, zero_division=0),
             nonbug_precision=precision_score(y, pred, pos_label=0, zero_division=0),
             nonbug_recall=recall_score(y, pred, pos_label=0, zero_division=0),
             nonbug_f1=f1_score(y, pred, pos_label=0, zero_division=0),
             macro_f1=f1_score(y, pred, average="macro", zero_division=0),
             mcc=matthews_corrcoef(y, pred),
             tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
             pred_bug_share=float(np.mean(pred)), n=int(len(y)))
    if s is not None and len(np.unique(y)) == 2:
        m["auc"] = roc_auc_score(y, s)
    return m


# ---------------------------------------------------------------------------
# 0. Dataset provenance summary
# ---------------------------------------------------------------------------
prov = []
for (repo, b), g in df.groupby(["repo", "is_bug"]):
    prov.append(dict(repo=repo, is_bug=b, n=len(g),
                     created_min=g.created_at.min()[:10], created_max=g.created_at.max()[:10],
                     closed_share=float((g.state == "closed").mean()),
                     no_label=int(g.labels.isna().sum()),
                     url_repo=g.url.str.extract(r"github.com/([^/]+/[^/]+)")[0].unique().tolist()))
pd.DataFrame(prov).to_csv(OUT / "provenance.csv", index=False)
R["latest_timestamp"] = max(df.created_at.max(), df.closed_at.dropna().max())
R["dup_repo_id"] = int(df.duplicated(["repo", "issue_id"]).sum())
R["dup_exact_text"] = int(df.duplicated(["text"]).sum())
R["dup_title"] = int(df.duplicated(["title"]).sum())
lab = df["labels"].fillna("")
R["k_bug_kindbug"] = int(((df.repo == "kubernetes") & (df.is_bug == 1) & lab.str.contains("kind/bug")).sum())
R["k_bug_regression_only"] = int(((df.repo == "kubernetes") & (df.is_bug == 1) & ~lab.str.contains("kind/bug")).sum())
R["e_bug_bug"] = int(((df.repo == "eclipse") & (df.is_bug == 1) & lab.str.contains(r"\bbug\b")).sum())
R["e_bug_regression_only"] = int(((df.repo == "eclipse") & (df.is_bug == 1) & ~lab.str.contains(r"\bbug\b")).sum())
R["e_nonbug_unlabelled"] = int(((df.repo == "eclipse") & (df.is_bug == 0) & (lab == "")).sum())
for pat, key in [(r"^\s*#{1,6}\s*What happened\?", "what_happened"),
                 (r"^\s*/kind\s+bug", "slash_kind_bug"),
                 (r"^\s*/kind\b", "slash_kind_any"),
                 (r"^\s*#{1,6}\s*What would you like to be added", "feature_heading"),
                 (r"^\s*#{1,6}\s*Which (tests|jobs) are (failing|flaking)", "failing_heading")]:
    m = df.body.str.contains(pat, flags=re.M | re.I, regex=True)
    for repo in ["kubernetes", "eclipse"]:
        for b in [0, 1]:
            R[f"{key}_{repo}_{b}"] = int((m & (df.repo == repo) & (df.is_bug == b)).sum())
R["median_words_full"] = df.groupby(["repo", "is_bug"])["text"].apply(lambda s: s.str.split().str.len().median()).to_dict()
R["median_words_clean"] = df.groupby(["repo", "is_bug"])["text_clean"].apply(lambda s: s.str.split().str.len().median()).to_dict()
R["median_words_full"] = {f"{k[0]}_{k[1]}": v for k, v in R["median_words_full"].items()}
R["median_words_clean"] = {f"{k[0]}_{k[1]}": v for k, v in R["median_words_clean"].items()}

# ---------------------------------------------------------------------------
# 1. Shared holdout (identical to the original split: stratified by class, seed 42)
# ---------------------------------------------------------------------------
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20,
                                  stratify=y_all, random_state=SEED)
split = pd.DataFrame({"row": np.r_[idx_tr, idx_te], "issue_id": df.issue_id.values[np.r_[idx_tr, idx_te]],
                      "repo": df.repo.values[np.r_[idx_tr, idx_te]],
                      "partition": ["train"] * len(idx_tr) + ["test"] * len(idx_te)})
split.to_csv(OUT / "holdout_split_ids.csv", index=False)
te = df.iloc[idx_te]
R["holdout_test_comp"] = {f"{a}_{b}": int(v) for (a, b), v in te.groupby(["repo", "is_bug"]).size().items()}

hold_rows, preds = [], pd.DataFrame({"issue_id": te.issue_id.values, "repo": te.repo.values, "y": te.is_bug.values})
fitted = {}
for variant in ["combined", "combined_no_template"]:
    col = TEXT_VARIANTS[variant]
    vec = vectorizer(5)
    Xtr = vec.fit_transform(df[col].values[idx_tr]); Xte = vec.transform(df[col].values[idx_te])
    R[f"n_features_{variant}"] = Xtr.shape[1]
    for name, mdl in models().items():
        mdl.fit(Xtr, y_all[idx_tr]); p = mdl.predict(Xte); s = scores(mdl, Xte)
        preds[f"{variant}|{name}|pred"] = p; preds[f"{variant}|{name}|score"] = s
        m = metrics(y_all[idx_te], p, s); m.update(variant=variant, model=name, subset="all")
        hold_rows.append(m)
        for repo in ["kubernetes", "eclipse"]:
            mask = te.repo.values == repo
            m = metrics(y_all[idx_te][mask], p[mask], s[mask]); m.update(variant=variant, model=name, subset=repo)
            hold_rows.append(m)
        fitted[(variant, name)] = (vec, mdl, Xtr, Xte)
hold = pd.DataFrame(hold_rows); hold.to_csv(OUT / "holdout_metrics.csv", index=False)
preds.to_csv(OUT / "holdout_predictions.csv", index=False)

# Bootstrap CI for holdout macro-F1 and paired difference
rng = np.random.default_rng(SEED)
yt = y_all[idx_te]
B = 2000
boot = {k: [] for k in ["LR", "MNB", "Linear SVM", "LR-MNB", "LR-SVM"]}
pl, pm, ps = (preds[f"combined|{k}|pred"].values for k in ["LR", "MNB", "Linear SVM"])
for _ in range(B):
    i = rng.integers(0, len(yt), len(yt))
    a = f1_score(yt[i], pl[i], average="macro"); b = f1_score(yt[i], pm[i], average="macro"); c = f1_score(yt[i], ps[i], average="macro")
    boot["LR"].append(a); boot["MNB"].append(b); boot["Linear SVM"].append(c)
    boot["LR-MNB"].append(a - b); boot["LR-SVM"].append(a - c)
R["boot_ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}


def mcnemar(y, a, b):
    ea, eb = a != y, b != y
    n01 = int((ea & ~eb).sum()); n10 = int((~ea & eb).sum())
    p_exact = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0
    return dict(only_first_wrong=n01, only_second_wrong=n10, chi2_cc=chi, p_exact=p_exact)


R["mcnemar_LR_vs_MNB"] = mcnemar(yt, pl, pm)
R["mcnemar_LR_vs_SVM"] = mcnemar(yt, pl, ps)

# Near-duplicate check between test and training reports (combined text, LR vectoriser)
vec, _, Xtr, Xte = fitted[("combined", "LR")]
sim = cosine_similarity(Xte, Xtr)
mx = sim.max(axis=1)
R["near_dup_test_ge_0.9"] = int((mx >= 0.9).sum())
R["near_dup_test_ge_0.8"] = int((mx >= 0.8).sum())
keep = mx < 0.8
m = metrics(yt[keep], pl[keep]); R["LR_macro_f1_excluding_near_dups"] = m["macro_f1"]; R["n_excluding_near_dups"] = int(keep.sum())
# same-repo pairs
R["near_dup_ge_0.9_by_repo"] = te[mx >= 0.9].repo.value_counts().to_dict()

# ---------------------------------------------------------------------------
# 2. Chi-square associations and LR coefficients
# ---------------------------------------------------------------------------
coef_tables = {}
for variant in ["combined", "combined_no_template"]:
    vec, lr, Xtr, Xte = fitted[(variant, "LR")]
    names = np.array(vec.get_feature_names_out())
    c = lr.coef_.ravel()
    o = np.argsort(c)
    coef_tables[variant] = dict(pos=[(names[j], float(c[j])) for j in o[::-1][:15]],
                                neg=[(names[j], float(c[j])) for j in o[:15]])
    chi, p = chi2(Xtr, y_all[idx_tr])
    mb = np.asarray(Xtr[y_all[idx_tr] == 1].mean(0)).ravel(); mn = np.asarray(Xtr[y_all[idx_tr] == 0].mean(0)).ravel()
    ct = pd.DataFrame(dict(feature=names, chi2=chi, p=p, cls=np.where(mb > mn, "Bug", "Non-bug"))).sort_values("chi2", ascending=False)
    ct.to_csv(OUT / f"chi2_{variant}.csv", index=False)
    pd.DataFrame(dict(feature=names, coef=c)).sort_values("coef").to_csv(OUT / f"lr_coef_{variant}.csv", index=False)
R["coef_tables"] = coef_tables
R["lr_intercept"] = float(fitted[("combined", "LR")][1].intercept_[0])


def explain(vec, lr, X, i, k=6):
    names = vec.get_feature_names_out(); row = X[i]
    contrib = row.multiply(lr.coef_.ravel()).tocsr()
    idx = contrib.indices; vals = contrib.data
    o = np.argsort(vals)
    return dict(top_bug=[(names[idx[j]], float(vals[j])) for j in o[::-1][:k] if vals[j] > 0],
                top_nonbug=[(names[idx[j]], float(vals[j])) for j in o[:k] if vals[j] < 0])


vec, lr, Xtr, Xte = fitted[("combined", "LR")]
prob = preds["combined|LR|score"].values
fp_i = int(np.argmax(np.where((pl == 1) & (yt == 0), prob, -1)))
fn_i = int(np.argmin(np.where((pl == 0) & (yt == 1), prob, 2)))
R["example_fp"] = dict(issue_id=int(te.issue_id.values[fp_i]), repo=te.repo.values[fp_i], prob=float(prob[fp_i]),
                       title=te.title.values[fp_i], labels=str(te.labels.values[fp_i]), **explain(vec, lr, Xte, fp_i))
R["example_fn"] = dict(issue_id=int(te.issue_id.values[fn_i]), repo=te.repo.values[fn_i], prob=float(prob[fn_i]),
                       title=te.title.values[fn_i], labels=str(te.labels.values[fn_i]), **explain(vec, lr, Xte, fn_i))

# ---------------------------------------------------------------------------
# 3. Repeated stratified CV on pooled data (5 x 5), vectoriser fitted per fold
# ---------------------------------------------------------------------------
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
folds = list(rskf.split(df, df["strata"]))
cv_rows = []
for variant, col in TEXT_VARIANTS.items():
    X_text = df[col].values
    for f, (tr, tst) in enumerate(folds):
        vec = vectorizer(5); Xtr = vec.fit_transform(X_text[tr]); Xte = vec.transform(X_text[tst])
        for name, mdl in models().items():
            mdl = clone(mdl).fit(Xtr, y_all[tr]); p = mdl.predict(Xte); s = scores(mdl, Xte)
            row = dict(variant=variant, model=name, fold=f,
                       macro_f1=f1_score(y_all[tst], p, average="macro"),
                       mcc=matthews_corrcoef(y_all[tst], p), auc=roc_auc_score(y_all[tst], s))
            for repo in ["kubernetes", "eclipse"]:
                mk = df.repo.values[tst] == repo
                row[f"macro_f1_{repo}"] = f1_score(y_all[tst][mk], p[mk], average="macro")
            cv_rows.append(row)
cv = pd.DataFrame(cv_rows); cv.to_csv(OUT / "cv_pooled_folds.csv", index=False)
cv_sum = cv.groupby(["variant", "model"]).agg(["mean", "std"]).drop(columns="fold")
cv_sum.to_csv(OUT / "cv_pooled_summary.csv")

# ---------------------------------------------------------------------------
# 4. Within-repository repeated CV (reference for transfer targets)
# ---------------------------------------------------------------------------
wr_rows = []
for repo in ["kubernetes", "eclipse"]:
    sub = df[df.repo == repo].reset_index(drop=True); ys = sub.is_bug.to_numpy()
    for variant in ["combined", "combined_no_template"]:
        Xt = sub[TEXT_VARIANTS[variant]].values
        for f, (tr, tst) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(sub, ys)):
            vec = vectorizer(3); Xtr = vec.fit_transform(Xt[tr]); Xte = vec.transform(Xt[tst])
            for name, mdl in models().items():
                mdl = clone(mdl).fit(Xtr, ys[tr]); p = mdl.predict(Xte); s = scores(mdl, Xte)
                wr_rows.append(dict(repo=repo, variant=variant, model=name, fold=f,
                                    macro_f1=f1_score(ys[tst], p, average="macro"),
                                    auc=roc_auc_score(ys[tst], s)))
wr = pd.DataFrame(wr_rows); wr.to_csv(OUT / "cv_within_repo_folds.csv", index=False)
wr.groupby(["repo", "variant", "model"])[["macro_f1", "auc"]].agg(["mean", "std"]).to_csv(OUT / "cv_within_repo_summary.csv")

# ---------------------------------------------------------------------------
# 5. Directional transfer, with confusion matrices and source-side thresholds
# ---------------------------------------------------------------------------
def source_threshold(model, Xs, ys):
    """Threshold maximising macro-F1 on 5-fold out-of-fold source scores."""
    oof = np.zeros(len(ys))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xs, ys):
        m = clone(model).fit(Xs[tr], ys[tr]); oof[va] = scores(m, Xs[va])
    grid = np.unique(np.quantile(oof, np.linspace(0.01, 0.99, 197)))
    f = [f1_score(ys, (oof >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(f))])


tr_rows, tr_scores = [], {}
for src, tgt in [("kubernetes", "eclipse"), ("eclipse", "kubernetes")]:
    S = df[df.repo == src]; T = df[df.repo == tgt]
    for variant in ["combined", "combined_no_template", "title_only", "description_only"]:
        col = TEXT_VARIANTS[variant]
        vec = vectorizer(3); Xs = vec.fit_transform(S[col].values); Xt = vec.transform(T[col].values)
        for name, mdl in models().items():
            mdl = clone(mdl).fit(Xs, S.is_bug.values); p = mdl.predict(Xt); s = scores(mdl, Xt)
            m = metrics(T.is_bug.values, p, s)
            m.update(source=src, target=tgt, variant=variant, model=name, threshold="default")
            m["mean_score_bug"] = float(s[T.is_bug.values == 1].mean()); m["mean_score_nonbug"] = float(s[T.is_bug.values == 0].mean())
            tr_rows.append(m)
            if name in ("LR", "MNB", "Linear SVM"):
                thr = source_threshold(models()[name], Xs, S.is_bug.values)
                p2 = (s >= thr).astype(int); m2 = metrics(T.is_bug.values, p2, s)
                m2.update(source=src, target=tgt, variant=variant, model=name, threshold="source_cv", thr_value=thr)
                tr_rows.append(m2)
            if variant == "combined" and name in ("LR", "MNB"):
                tr_scores[f"{src}->{tgt}|{name}"] = (T.is_bug.values, s)
                if name == "LR" and src == "eclipse":
                    # failed transfer example: Kubernetes bug with lowest LR probability
                    yb = T.is_bug.values; i = int(np.argmin(np.where(yb == 1, s, 2)))
                    R["example_transfer_fail"] = dict(issue_id=int(T.issue_id.values[i]), prob=float(s[i]),
                                                      title=T.title.values[i], **explain(vec, mdl, Xt, i))
                    names = np.array(vec.get_feature_names_out()); c = mdl.coef_.ravel(); o = np.argsort(c)
                    R["eclipse_lr_top_pos"] = [(names[j], float(c[j])) for j in o[::-1][:10]]
                    R["eclipse_lr_top_neg"] = [(names[j], float(c[j])) for j in o[:10]]
                    R["eclipse_lr_intercept"] = float(mdl.intercept_[0])
                    R["E2K_vocab_overlap"] = float(np.mean(np.asarray(Xt.sum(1)).ravel() > 0))
tr = pd.DataFrame(tr_rows); tr.to_csv(OUT / "transfer_metrics.csv", index=False)
np.save(OUT / "transfer_scores.npy", {k: (v[0], v[1]) for k, v in tr_scores.items()}, allow_pickle=True)

# Matched-size source subsampling: K->E with 711 Kubernetes reports (30 draws)
E = df[df.repo == "eclipse"]; K = df[df.repo == "kubernetes"]
ms_rows = []
for rep in range(30):
    # match both size and class ratio of the Eclipse source (269 bug / 442 non-bug)
    kb = K[K.is_bug == 1].sample(269, random_state=SEED + rep); kn = K[K.is_bug == 0].sample(442, random_state=SEED + rep)
    Ks = pd.concat([kb, kn])
    for variant in ["combined", "combined_no_template"]:
        col = TEXT_VARIANTS[variant]
        vec = vectorizer(3); Xs = vec.fit_transform(Ks[col].values); Xt = vec.transform(E[col].values)
        for name in ["LR", "MNB"]:
            mdl = clone(models()[name]).fit(Xs, Ks.is_bug.values); p = mdl.predict(Xt); s = scores(mdl, Xt)
            ms_rows.append(dict(rep=rep, variant=variant, model=name, macro_f1=f1_score(E.is_bug.values, p, average="macro"),
                                auc=roc_auc_score(E.is_bug.values, s), pred_bug_share=float(p.mean())))
ms = pd.DataFrame(ms_rows); ms.to_csv(OUT / "matched_size_K2E.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Indicator rules applied to both classes, before and after template removal
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from indicator_rules import TAXONOMY_RULES, CAUSAL_PATTERNS
comp = {k: [re.compile(p, re.I) for p in v] for k, v in TAXONOMY_RULES.items()}
caus = [re.compile(p, re.I) for p in CAUSAL_PATTERNS]
ind_rows = []
for variant in ["combined", "combined_no_template"]:
    col = TEXT_VARIANTS[variant]
    flags = pd.DataFrame({k: df[col].map(lambda t, ps=ps: any(p.search(t) for p in ps)) for k, ps in comp.items()})
    flags["Any category"] = flags.any(axis=1)
    flags["Causal phrase"] = df[col].map(lambda t: any(p.search(t) for p in caus))
    flags["repo"] = df.repo; flags["is_bug"] = df.is_bug
    flags.to_csv(OUT / f"indicator_flags_{variant}.csv", index=False)
    for cat in list(comp) + ["Any category", "Causal phrase"]:
        for scope in ["all", "kubernetes", "eclipse"]:
            g = flags if scope == "all" else flags[flags.repo == scope]
            a = g[g.is_bug == 1][cat]; b = g[g.is_bug == 0][cat]
            tab = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
            p = stats.chi2_contingency(tab)[1] if tab.min() > 0 else np.nan
            ind_rows.append(dict(variant=variant, scope=scope, category=cat, bug_n=int(a.sum()), bug_pct=100 * a.mean(),
                                 nonbug_n=int(b.sum()), nonbug_pct=100 * b.mean(), diff_pp=100 * (a.mean() - b.mean()), p=p))
ind = pd.DataFrame(ind_rows); ind.to_csv(OUT / "indicator_prevalence.csv", index=False)

# Which terms trigger Dependency/build in bug reports (first matching pattern)
dep = [re.compile(p, re.I) for p in TAXONOMY_RULES["Dependency/build"]]
trig = {}
for t in df[df.is_bug == 1].text:
    for p in dep:
        if p.search(t):
            trig[p.pattern] = trig.get(p.pattern, 0) + 1
R["dependency_term_hits_bug"] = trig

# ---------------------------------------------------------------------------
R["env"] = dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                pandas=pd.__version__, scipy=__import__("scipy").__version__)
json.dump(R, open(OUT / "results.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("done")
