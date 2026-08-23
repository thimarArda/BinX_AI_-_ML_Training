# Cardiac Patient Monitoring System — ML Analysis (Large-Scale Version)

Individual AI & Machine Learning Track project, rebuilt on a large (~320K row) public dataset
instead of a small (~1,000 row) one, to give a more realistic picture of model performance and
avoid the artificially clean/optimistic results that a small, balanced clinical dataset can produce.

## Why this dataset

The earlier version of this project used a 1,000-row clinical dataset with a roughly 50/50 class
split — clean, but small enough that results can look unrealistically strong, and complex enough
models can appear to add value that a larger, messier dataset would reveal doesn't hold up. This
version deliberately trades that clean setup for something closer to real-world conditions:
~300K rows, ~9% positive class, mostly self-reported lifestyle/survey features rather than clinical
measurements. See Section 1 of the notebook for the full reasoning.

## Dataset

**Personal Key Indicators of Heart Disease** — CDC Behavioral Risk Factor Surveillance System
(BRFSS) 2020 annual survey, distributed publicly on Kaggle by Kamil Pytlak
(kamilpytlak/personal-key-indicators-of-heart-disease). 319,795 respondents, 17 features, binary
target (`HeartDisease`).

The CSV (`heart_2020_raw.csv`) is included in `data/`.

## Project structure

```
project1/
├── data ── heart_2020_raw.csv
│   
├── notebooks ── cardiac_ml_analysis_large.ipynb    <- main deliverable, runs top to bottom
│   
├── outputs/ folder with pictures of the outcomes                               
├── requirements.txt
└── README.md
```

## How to run

1. Create a virtual environment (optional but recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Launch Jupyter and run the notebook top to bottom:
   ```
   jupyter notebook notebooks/cardiac_ml_analysis_large.ipynb
   ```
   Full execution takes a few minutes (the dataset is ~300K rows). Compute settings
   (`n_estimators=150`, 3-fold CV) were tuned for a single-core machine — increase them for
   marginally more stable numbers on a multi-core machine; conclusions won't change qualitatively.

## What changed vs. the small-dataset version, and why

- **Class imbalance is front and center.** Only ~8.6% of respondents report heart disease.
  Accuracy is explicitly de-emphasized in favor of precision/recall/F1/ROC-AUC, and
  `class_weight="balanced"` is used on both models throughout.
- **Model complexity was deliberately capped, not maximized.** The Random Forest is intentionally
  shallow (`max_depth=8`) rather than tuned for peak accuracy — the goal was testing whether a
  complex model earns its keep on this data, not chasing a leaderboard number.
- **Ordinal columns are encoded as ordinal, not one-hot.** `AgeCategory` and `GenHealth` are
  converted to numeric scales that preserve their natural ordering, rather than blown out into many
  one-hot columns.
- **Unsupervised analysis runs on a stratified 20,000-row sample**, explicitly for tractability —
  PCA/KMeans on the full one-hot-encoded ~300K-row matrix is both slow and unreadable as a 2D plot
  regardless. The sample preserves the true ~9% positive rate.
- **Duplicate-row removal is treated as a judgment call**, not an obvious step — with mostly
  low-cardinality categorical columns and no respondent ID, some "duplicate" rows may be
  coincidentally identical distinct people. This is stated explicitly in Section 2 rather than
  silently assumed.

## Actual results (this run — see notebook for full detail)

| Metric | Logistic Regression | Random Forest (shallow) | Random Forest + engineered features |
|---|---|---|---|
| Accuracy | 0.743 | 0.719 | 0.731 |
| Balanced Accuracy | 0.755 | 0.756 | 0.754 |
| Precision | 0.227 | 0.216 | 0.221 |
| Recall | 0.770 | 0.802 | 0.782 |
| F1-score | 0.351 | 0.340 | 0.345 |
| ROC-AUC | 0.832 | 0.828 | 0.827 |

**Read this table carefully, not just the top row.** Accuracy actually *drops* slightly for the
more complex models relative to Logistic Regression, while ROC-AUC is essentially tied across all
three. This is a genuinely useful, honest finding: on this dataset, a simple linear model does
about as well as a more complex one, and the engineered features did not clearly outperform the
raw ones either — a real result, not one that was massaged to look more impressive than it is.

Unsupervised clustering (PCA + KMeans, 20,000-row stratified sample): silhouette score 0.400,
Adjusted Rand Index vs. true labels 0.179 — low, and expectedly so, since disease status is a rare
outcome (~9%) rather than the dominant axis of variation in this feature space. A low ARI here is
not a modeling failure; see Section 8 of the notebook for the full interpretation.

## Limitations

- Self-reported survey data (BRFSS) carries known biases (recall bias, social desirability bias)
  that a clinically-measured dataset would not have.
- No respondent ID means duplicate-row removal is a judgment call, not a certainty.
- This is an educational project under a training curriculum — it makes no diagnostic or clinical
  claims and should not be used for real health decision-making.
