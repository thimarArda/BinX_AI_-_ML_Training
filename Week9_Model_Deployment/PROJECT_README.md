# Banking77 Customer Intent Classifier

A machine learning system that classifies customer banking messages into
one of 77 fine-grained intents (e.g. *"I lost my card"* → `lost_or_stolen_card`),
served through a live, public web app.

**🚀 Live demo:** https://binxai-mltraining-uw8gvrchftbfti8nldhlmu.streamlit.app/

---

## Problem Statement

Customer support teams at banks and fintech companies receive large
volumes of free-text queries that need to be routed to the right process
or team quickly. Manually triaging every message is slow and expensive.
This project builds a text classification model that automatically
identifies the customer's intent from a single message, and deploys it as
a usable, public web app — demonstrating a complete path from raw data to
a live product.

## Dataset

[**Banking77**](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)
(PolyAI) — 13,083 real customer service queries from the banking domain,
labeled with one of **77 fine-grained intents** (e.g. `card_payment_fee_charged`,
`declined_card_payment`, `top_up_reverted`). Loaded directly from the
dataset's official CSV files.

This is a genuinely hard classification problem: 77 classes, many of them
semantically close (e.g. `card_arrival` vs. `card_delivery_estimate`),
extracted from real, informally phrased customer messages.

## Methodology

**1. Exploratory Data Analysis**
Reviewed class distribution across the 77 intents, message length, and
vocabulary characteristics of the raw customer queries.

**2. Preprocessing**
- Lowercasing and punctuation removal
- Tokenization
- Stopword removal — with negation words (`no`, `not`, `nor`, `never`)
  explicitly preserved, since removing them changes the meaning of a
  banking complaint (e.g. "payment **not** received" vs "payment received")
- Lemmatization

**3. Feature Extraction**
TF-IDF vectorization of the cleaned text.

**4. Modeling**
Logistic Regression classifier trained on the TF-IDF features, chosen for
its speed, low resource footprint, and ease of deployment relative to a
transformer-based alternative that was also explored (see
[Limitations & Future Work](#limitations--future-work)).

**5. Error Analysis**
Reviewed misclassifications to understand systematic weaknesses (see
Results below).

**6. Serialization & Reproducibility**
The trained model, TF-IDF vectorizer, and label encoder were serialized
with `joblib`, verified in a completely fresh process to confirm zero
training/serving skew, and paired with a pinned `requirements.txt` so the
exact environment can be reproduced.

**7. Serving**
Wrapped in a FastAPI REST API (`POST /predict`) with Pydantic request
validation, then in a Streamlit web app for interactive public use.

**8. Deployment**
Deployed to a public URL via Streamlit Community Cloud, connected
directly to this GitHub repository.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **85.6%** |
| Number of classes | 77 |
| Model | TF-IDF + Logistic Regression |

**Error analysis findings:** the model most often confuses semantically
related intents that share vocabulary — for example
`virtual_card_not_working` vs. `get_disposable_virtual_card` — which is
expected given how closely worded these categories are in real customer
language.

## Model Confidence Guardrail

Testing the deployed API surfaced a real limitation: any input sharing no
vocabulary with the training data (gibberish, numbers, keyboard mashing)
produced an all-zero TF-IDF vector, and the classifier still returned a
plausible-looking but meaningless default prediction. A guardrail was
added to detect this case and return `confident: false` / `"unknown"`
instead of a fabricated answer — this is documented in more detail in the
[Day 2 API documentation](https://github.com/thimarArda/BinX_AI_-_ML_Training/blob/main/Week9_Model_Deployment/day2_3_fastapi_streamlit/README.md) and
[Day 4 deployment documentation](https://github.com/thimarArda/BinX_AI_-_ML_Training/tree/main/Week9_Model_Deployment/Day%204%20%E2%80%94%20Public%20Deployment#readme).

## Live Application

**URL:** https://binxai-mltraining-uw8gvrchftbfti8nldhlmu.streamlit.app/

The app accepts a free-text customer message and returns the predicted
intent along with a top-5 confidence chart. It applies the exact same
preprocessing, vectorizer, and model used during training and evaluation
— no separate reimplementation, avoiding training/serving skew.

A Hugging Face Space was also fully configured as a second deployment
target (correct SDK, `app_file`, and artifacts committed), but its build
is currently blocked by a confirmed platform-side Hugging Face bug
affecting many free-tier accounts this month (see
[Day 4 documentation](https://github.com/thimarArda/BinX_AI_-_ML_Training/tree/main/Week9_Model_Deployment/Day%204%20%E2%80%94%20Public%20Deployment#readme)
for details and forum links). Streamlit Community Cloud was used as the
working deployment instead.

## Repository Structure

```
.
├── README.md                              # This file
├── Day1_Serialization/                    # Model, vectorizer, label encoder serialization
│   ├── train.py
│   ├── verify_reload.py
│   ├── requirements.txt
│   └── artifacts/
├── day2_3_fastapi_streamlit/              # FastAPI serving + Streamlit UI
│   ├── main.py                            # FastAPI app (POST /predict)
│   ├── streamlit_app.py                   # Streamlit app (deployed version)
│   ├── test_api.py
│   ├── requirements.txt
│   ├── README.md
│   └── artifacts/
└── Day 4 — Public Deployment/             # Deployment docs, bug fixes, live URL
    ├── README.md
    ├── streamlit_app.py                   # Final deployed copy
    ├── requirements.txt
    ├── artifacts/
    └── huggingface_space/                 # HF Space config (blocked deployment)
```

## Setup — Running Locally

**Requirements:** Python 3.11+ (tested on 3.14). Python 3.8 will not work —
scikit-learn 1.8.0 requires Python 3.11 or newer.

```bash
# 1. Clone the repo
git clone <this-repo-url>
cd <repo-folder>/day2_3_fastapi_streamlit

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download required NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 5a. Run the Streamlit app
streamlit run streamlit_app.py

# 5b. Or run the FastAPI service
uvicorn main:app --reload
# then open http://127.0.0.1:8000/docs
```

## Limitations & Future Work

- **Accuracy ceiling of the current model.** A fine-tuned transformer
  (DistilBERT) was also trained on the same dataset and reached ~92%
  accuracy, notably higher than this deployed model's 85.6%. The
  TF-IDF + Logistic Regression pipeline was chosen for deployment due to
  its small size (~1.4MB total) and fast CPU inference, which matters on
  free hosting tiers. Deploying the transformer version is a natural next
  step, likely requiring a paid or GPU-backed hosting tier given its size
  and inference cost.
- **No true out-of-distribution confidence scoring.** The current
  guardrail only catches inputs with *zero* vocabulary overlap with
  training data. A short, ambiguous, but still recognizable message can
  still be confidently misclassified, since no probability threshold is
  applied to genuine (non-zero) predictions.
- **Class confusion among semantically similar intents**, as noted in
  error analysis — could be improved with more training examples for
  commonly confused pairs, or a hierarchical classification approach
  (broad category first, then fine-grained intent).
- **No user feedback loop.** The deployed app doesn't currently collect
  incorrect predictions for future retraining — adding this would enable
  iterative model improvement based on real usage.

## Reproducibility

- Random seed: TF-IDF vectorization and the `lbfgs` solver used by
  Logistic Regression are deterministic by default in scikit-learn for
  this configuration, so results are reproducible without an explicit
  seed; `train.py` retrains and re-serializes from scratch and reproduces
  the same test accuracy on each run.
- Environment: `requirements.txt` pins every library actually imported by
  the code (not a full `pip freeze` dump), avoiding brittle version
  conflicts while keeping the environment reproducible.
- Verification: `verify_reload.py` loads the serialized artifacts in a
  completely fresh process and confirms predictions match a known
  training-time result, guarding against training/serving skew.

## Tools Used

Python • scikit-learn • NLTK • pandas • joblib • FastAPI • Pydantic •
Uvicorn • Streamlit • Streamlit Community Cloud • Hugging Face Spaces
(configured, blocked by platform issue) • Git/GitHub
