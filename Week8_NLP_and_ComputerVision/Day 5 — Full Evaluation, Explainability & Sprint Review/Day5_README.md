# Banking77 Intent Classification

## Overview

In this nootebook, we are evaluating the project built in [Day 4 — Model Integration & Error Analysis](https://github.com/thimarArda/BinX_AI_-_ML_Training/tree/main/Week8_NLP_and_ComputerVision/Day%204%20%E2%80%94%20Model%20Integration%20%26%20Error%20Analysis)  and adding a full evaluation on top of it.

Specifically, we add:
- **Aappropriate metrics** (precision, recall, F1) compared against a baseline, instead of relying on accuracy alone.
- **An imbalance check**, to confirm whether techniques like SMOTE are actually needed for this dataset.
- **SHAP explainability**, to understand which words drive the model's predictions, both globally and for individual examples.


## Dataset

**Banking77** — customer service queries labeled with 77 fine-grained banking intents.
- Train: 10,003 examples
- Test: 3,080 examples

Loaded directly from the original source CSVs (not via `datasets.load_dataset()`, which fails due to a deprecated script-based loader):

```
https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv
https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv
```

## Pipeline

1. **Preprocess** — lowercase, remove punctuation, tokenize, remove stopwords (keeping negations), lemmatize.
2. **Vectorize** — `TfidfVectorizer`, fit on train only.
3. **Train** — `LogisticRegression` (classical ML).
4. **Integrate** — a single `predict(raw_input)` function chains all steps, guaranteeing training-time and prediction-time preprocessing match exactly (no training/serving skew).
5. **Evaluate** — confusion matrix, misclassified examples, full classification report, baseline comparison.
6. **Explain** — SHAP global feature importance and per-prediction explanations.

## Results

- **Test accuracy:** ~85.6%
- **Baseline (majority class) accuracy:** reported in notebook, for comparison
- **Misclassified:** 444 / 3080 (~14%)
- **Largest error cluster:** `virtual_card_not_working` confused with `get_disposable_virtual_card` (14 cases) — a model weakness, not a data issue. The shared phrase "disposable virtual card" dominates TF-IDF weighting, drowning out the actual signal words like "not working" or "rejected."
- **Imbalance handling (SMOTE):** not applied. Banking77's classes are reasonably balanced, so oversampling isn't needed — confirmed by checking class distribution rather than assumed.
- **SHAP:** identifies which words drive each class's predictions, both globally (summary plot) and for individual examples (waterfall plot), supporting explainability to non-technical stakeholders.

## Suggested Next Steps

- Add bigrams to TF-IDF (`ngram_range=(1,2)`) to better capture phrases like "not working."
- Add more training examples for underperforming classes.
- Consider embedding-based features if vocabulary-overlap confusions persist after the above.

## Requirements

```
pandas
numpy
nltk
scikit-learn
seaborn
matplotlib
shap
```

NLTK resources (downloaded at runtime):
```python
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## How to Run

Run all cells top to bottom. Use the integrated pipeline directly:

```python
predict("I do not recognize this payment.")
```

## Notes

This project uses classical ML (TF-IDF + Logistic Regression), not deep learning. The same integration and evaluation pattern would apply to a deep learning model as well.
