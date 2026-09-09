# Banking77 Intent Classification

This notebook integrates a text classification pipeline (preprocessing → TF-IDF → Logistic Regression) into a single `predict()` function, and analyzes where the model fails.

## What's inside
- **Data:** Banking77 dataset (10,003 train / 3,080 test), loaded from CSV.
- **Preprocessing:** lowercase, remove punctuation, tokenize, remove stopwords (keeping negations), lemmatize.
- **Model:** TF-IDF + Logistic Regression — **~85.6% accuracy**.
- **Integration:** `predict(raw_input)` runs the exact same preprocessing used in training, avoiding train/serve skew.
- **Error analysis:** confusion matrix + misclassified examples, categorized as data issues vs. model weaknesses.

## Key finding
Biggest error cluster: `virtual_card_not_working` → `get_disposable_virtual_card` (14 cases). Model weakness — shared phrase "disposable virtual card" outweighs the actual signal words like "not working" or "rejected."

## Run it
Run all cells top to bottom, then:
```python
predict("I do not recognize this payment.")
```