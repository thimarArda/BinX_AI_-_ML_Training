# Banking77 Intent Classifier — Serialized Model

A text classification model that predicts customer banking intent from a raw
message (e.g. *"I forgot my PIN"* → `pin_blocked`). Trained on the
[Banking77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)
(PolyAI), which contains real customer-service style questions labeled with
one of 77 fine-grained banking intents.

This repo contains the **serialized, production-ready artifacts** — the
trained model, fitted vectorizer, and label encoder — plus scripts to
retrain from scratch and to verify the saved files load correctly and
reproduce training-time predictions.

## What's inside

```
.
├── artifacts/
│   ├── model.joblib            # Trained LogisticRegression classifier
│   ├── vectorizer.joblib       # Fitted TF-IDF vectorizer
│   ├── label_encoder.joblib    # Maps the 77 intent labels <-> class indices
│   └── known_check.txt         # A known input/output pair used for verification
├── train.py                    # Retrains the model from scratch and re-serializes it
├── verify_reload.py            # Loads the saved artifacts and confirms predictions match training
└── requirements.txt            # Pinned dependencies
```

## Model details

- **Task:** 77-class text classification (banking customer intent detection)
- **Pipeline:** text preprocessing (lowercase, punctuation removal,
  tokenization, stopword removal with negations preserved, lemmatization) →
  TF-IDF vectorization → Logistic Regression
- **Test accuracy:** ~85.6%
- **Dataset:** [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)
  (13,083 customer service queries, 77 intents), loaded directly from the
  official CSVs

## Setup

**Requirements:** Python 3.11+ recommended (tested on 3.14). Python 3.8 will
**not** work — scikit-learn 1.8.0 requires Python 3.11 or newer.

1. **Clone the repo and move into it:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate it:**
   - Windows (PowerShell):
     ```powershell
     venv\Scripts\Activate.ps1
     ```
   - Windows (cmd.exe):
     ```
     venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the required NLTK data (one-time):**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

## Usage

### Run the verification test

Confirms the saved model, vectorizer, and label encoder load correctly and
reproduce the exact prediction recorded during training (i.e. no
training/serving skew):

```bash
python verify_reload.py
```

**Expected output:**
```
Input:    'I do not recognize this payment.'
Expected: card_payment_not_recognised
Actual:   card_payment_not_recognised
MATCH — serialization verified, no skew.
'Why was my card payment declined?' -> declined_card_payment
'How do I top up my account?' -> topping_up_by_card
```

### Get a prediction interactively

```bash
python
```
```python
>>> import sys
>>> sys.path.insert(0, '.')
>>> from verify_reload import predict
>>> predict("I want a new account")
'terminate_account'
```

> Note: the model occasionally misclassifies inputs that are ambiguous or
> phrased differently from its training examples — as seen above, this is
> a real limitation of the current model, not a bug. See
> [Error Analysis](#known-limitations) below.

### Retrain from scratch

If you want to reproduce the training run yourself (re-downloads the
dataset, retrains, and overwrites the files in `artifacts/`):

```bash
python train.py
```

## Known limitations

- The classifier confuses semantically related intents (e.g.
  `virtual_card_not_working` vs. `get_disposable_virtual_card`), especially
  when a message is short or could plausibly belong to more than one
  category.
- 77 fine-grained classes means some intents have limited training examples,
  which can hurt performance on rarer categories.
- This model is TF-IDF + Logistic Regression, chosen for its light weight
  and fast inference — a fine-tuned transformer (e.g. DistilBERT) trained
  on the same dataset achieves higher accuracy (~92%) at the cost of a much
  larger model size and slower CPU inference.

## License / Attribution

Trained on the Banking77 dataset by PolyAI
([paper](https://arxiv.org/abs/2003.04807),
[dataset](https://github.com/PolyAI-LDN/task-specific-datasets)).
