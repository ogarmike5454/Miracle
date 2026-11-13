# Replication package

Interpretable Bug Report Classification Across Repositories: Template Effects and Transfer Limitations

## Data

`data/all_repos_combined.csv` holds 5,711 GitHub issues:

| Source | GitHub repository | Issues created | Bugs / non-bugs |
|---|---|---|---|
| Kubernetes | https://github.com/kubernetes/kubernetes | Oct 2022 to Mar 2026 | 2,500 / 2,500 (all closed) |
| Eclipse Platform | https://github.com/eclipse-platform/eclipse.platform | Mar 2022 to Mar 2026 | 269 / 442 (509 closed, 202 open) |

The latest timestamp in the data is 2026-03-26. `is_bug` = 1 when the issue carries `kind/bug` or
`kind/regression` (Kubernetes) or `bug` or `regression` (Eclipse); every other issue is 0, including
291 Eclipse issues with no label. Titles, bodies and labels are as they stood at retrieval.

## Code

All scripts run with `python -I` from the package root.

```
python -I scripts/run_experiments.py data/all_repos_combined.csv results   # about 9 minutes on one CPU
python -I scripts/make_figures.py results figs
```

* `scripts/common.py` – text variants, the template filter (`strip_template`), TF-IDF settings and classifier settings.
* `scripts/indicator_rules.py` – all 73 category regular expressions and 8 causal patterns.
* `scripts/run_experiments.py` – every number reported in the paper.
* `scripts/make_figures.py` – Fig. 1 and Fig. 2.

Environment used: Python 3.13.18, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1. Random seed 42.

## Results files

| File | Content |
|---|---|
| `results/holdout_split_ids.csv` | Train/test assignment of every issue in the shared holdout |
| `results/holdout_predictions.csv` | Saved test predictions and scores for every model and text variant |
| `results/holdout_metrics.csv` | Holdout metrics, pooled and per repository, with confusion counts |
| `results/cv_pooled_folds.csv` | Fold-level scores for the 5×5 cross-validation (all text variants) |
| `results/cv_within_repo_folds.csv` | Fold-level scores within each repository |
| `results/transfer_metrics.csv` | Directional transfer, default and source-side thresholds, confusion counts |
| `results/matched_size_K2E.csv` | 30 matched-size Kubernetes source samples |
| `results/indicator_prevalence.csv` | Indicator prevalence in bugs and non-bugs, full and stripped text |
| `results/indicator_flags_*.csv` | Per-report indicator flags |
| `results/chi2_*.csv`, `results/lr_coef_*.csv` | Chi-square associations and LR coefficients |
| `results/results.json` | Provenance counts, McNemar tests, bootstrap CIs, near-duplicate check, explanation examples |

## Optional indicator annotation

`annotation_sample_160.csv` is a stratified sample (40 reports per repository and class) prepared
for two independent annotators. Suggested guideline: read the title and description at the URL;
for each rule category, mark it wrong if the text does not describe that kind of problem; list
categories the rules missed; and classify the matched cue as a symptom, an environment description,
a temporal relationship or a causal hypothesis. Report Cohen's kappa before adjudication. This
checks the textual categorisation only, not technical root causes.
