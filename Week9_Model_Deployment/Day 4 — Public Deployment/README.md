# Day 4 — Public Deployment

Deploying the Banking77 intent classifier's Streamlit app to a free public
hosting platform, making it accessible at a live URL anyone can open.

##  Live Demo

**https://binxai-mltraining-uw8gvrchftbfti8nldhlmu.streamlit.app/**

Try it with a real banking question (e.g. *"I lost my card and need a
replacement"*) or a gibberish input (e.g. *"asdkjfh"*) to see the
out-of-vocabulary guardrail in action.

## What this folder contains

```
.
├── README.md              # This file
├── streamlit_app.py        # Final deployed version of the app
├── requirements.txt        # Pinned dependencies used for deployment
├── artifacts/
│   ├── model.joblib
│   ├── vectorizer.joblib
│   └── label_encoder.joblib
└── huggingface_space/
    └── README.md            # Space config (see "Hugging Face Spaces" note below)
```

This app builds directly on the model serialized on Day 1 and served via
FastAPI on Day 2 — no retraining happened here. The `app.py`/
`streamlit_app.py` code is unchanged from Day 2/3 except for two
deployment-specific fixes described below.

## Deployment platform: Streamlit Community Cloud

Streamlit Community Cloud was used, connected directly to this project's
GitHub repository. It builds and hosts the app automatically from a
specified branch and file path — no manual server setup required.

**Steps taken:**
1. Pushed the Streamlit app, `requirements.txt`, and serialized model
   artifacts to GitHub.
2. Connected the repo to Streamlit Community Cloud (share.streamlit.io),
   specifying the branch and the path to `streamlit_app.py`.
3. Fixed three deployment-specific bugs (below) that only appeared once the
   app ran on a hosting platform, not locally.
4. Confirmed the live app produces the same predictions as the local
   version.

## Bugs found and fixed during deployment

Deployment surfaced three real issues that never appeared during local
testing — a reminder that "works on my machine" isn't the same as "works in
production."

### 1. NLTK data not available on the server

**Problem:** the app's preprocessing uses NLTK's stopwords, tokenizer, and
lemmatizer, which depend on separately downloaded data files. Locally, this
data was already present because it had been downloaded once during setup.
On a fresh cloud container, none of that data exists, so the app would
crash the first time a prediction was requested.

**Fix:** added a cached startup function that downloads the required NLTK
data on first run in any new environment:
```python
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

download_nltk_data()
```

### 2. Special characters in the project folder path broke the installer

**Problem:** the original folder name (`Day 2 and 3 — Serving the Model
with FastAPI + Streamlit`) contained an em dash (`—`). When Streamlit
Cloud built its install command using this path, the dash was parsed as a
separate token and passed to `pip`/`uv` as an invalid package name,
causing the entire dependency install to fail with `Failed to parse: —`,
even though `requirements.txt` itself was completely correct.

**Fix:** renamed the folder to use only alphanumeric characters and
underscores (`day2_3_fastapi_streamlit`), and updated the Streamlit Cloud
app configuration to point to the new path. Lesson: avoid spaces and
special characters (dashes, `+`, etc.) in any folder that will be part of
a deployment path.

### 3. Relative file paths broke on the hosting platform

**Problem:** the app loaded model artifacts with a relative path
(`"artifacts/model.joblib"`). Locally this worked because the script was
always run from inside its own folder. On Streamlit Cloud, the app runs
with the working directory set to the **repository root**, not the
script's folder — so the relative path resolved to a location where the
files didn't exist, causing a `FileNotFoundError`.

**Fix:** built the artifact paths relative to the script's own file
location instead of the current working directory:
```python
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "artifacts", "model.joblib"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "artifacts", "vectorizer.joblib"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "artifacts", "label_encoder.joblib"))
    return model, vectorizer, label_encoder
```

## Hugging Face Spaces (attempted, currently blocked)

A Hugging Face Space (`thimar/banking77-intent-classifier`) was also fully
configured and pushed — correct SDK (`streamlit`), correct `app_file`,
NLTK download fix included, and all serialized artifacts committed. The
Space build cannot currently be verified because the account is affected
by a confirmed, platform-side Hugging Face bug: newly created accounts with
a single Space on the free CPU Basic tier are incorrectly shown a
"CPU Basic quota limit" error and paused, even with zero other Spaces
running. This is a widely reported issue on Hugging Face's own community
forum, not an issue with this project's configuration. The Space's
`huggingface_space/README.md` in this folder shows the working
configuration for reference. Streamlit Community Cloud was used as the
working deployment platform instead.

## Verifying local vs. deployed predictions

The live app was tested against the same inputs used during local testing
on Day 2/3, confirming identical output:

| Input | Local prediction | Deployed prediction |
|---|---|---|
| "I do not recognize this payment." | `card_payment_not_recognised` | `card_payment_not_recognised` |
| "I want a new account" | `terminate_account` | `terminate_account` |
| "hhhhhhhhhhhhhhhhh." (gibberish) | `unknown` / not confident | `unknown` / not confident |

No mismatch was found between local and deployed behavior.
