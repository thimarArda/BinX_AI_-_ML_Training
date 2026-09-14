# Banking77 Intent Classifier — FastAPI Serving

A REST API that serves a trained Banking77 intent classification model.
Send it a customer banking message (e.g. *"I lost my card"*) and it returns
the predicted intent (e.g. `lost_or_stolen_card`) as JSON.

This project builds on a previously serialized model (TF-IDF + Logistic
Regression, ~85.6% accuracy, trained on the
[Banking77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data))
by wrapping it in a live FastAPI service — turning a saved model file into
something any application (a website, mobile app, or another backend) can
actually call.

## What's inside

```
.
├── main.py              # FastAPI app: loads the model and exposes /predict
├── test_api.py           # Script to manually test the running API
├── requirements.txt      # Pinned dependencies
└── artifacts/
    ├── model.joblib            # Trained LogisticRegression classifier
    ├── vectorizer.joblib       # Fitted TF-IDF vectorizer
    ├── label_encoder.joblib    # Maps the 77 intent labels <-> class indices
    └── known_check.txt         # Known input/output pair used for verification
```

The artifacts are the same serialized files produced and verified in the
previous serialization step — they were not retrained here.

## Setup

**Requirements:** Python 3.11+ (tested on 3.14). Python 3.8 will not work —
scikit-learn 1.8.0 requires Python 3.11 or newer.

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate it:**
   - Windows (PowerShell): `venv\Scripts\Activate.ps1`
   - Windows (cmd.exe): `venv\Scripts\activate`
   - macOS / Linux: `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the required NLTK data (one-time):**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

## Running the API

Start the server:

```bash
uvicorn main:app --reload
```

Then open the interactive documentation in your browser:

```
http://127.0.0.1:8000/docs
```

From there, expand the `/predict` endpoint, click **Try it out**, enter a
request body, and click **Execute** — no separate client needed.

### Example request

```json
{
  "text": "I do not recognize this payment."
}
```

### Example response

```json
{
  "input_text": "I do not recognize this payment.",
  "predicted_intent": "card_payment_not_recognised",
  "confident": true
}
```

### Testing via curl

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I want a new account"}'
```

### Testing via script

With the server running in one terminal, run this in another:

```bash
pip install requests
python test_api.py
```

## Input validation

The `/predict` endpoint uses Pydantic to validate every request. Invalid
input is rejected with a clear `422 Unprocessable Entity` response instead
of crashing the model:

| Invalid input | Example | Response |
|---|---|---|
| Wrong type | `{"text": 12345}` | `422`, "Input should be a valid string" |
| Missing field | `{}` | `422`, "Field required" |
| Empty string | `{"text": ""}` | `422`, "String should have at least 1 character" |

## Known limitations

- **No out-of-distribution detection at the vocabulary level was originally
  present** — early testing found that any input with no vocabulary overlap
  with the training data (gibberish like `"hhhhh"`, numbers like
  `"2222222"`, random keyboard input) produced an all-zero TF-IDF vector.
  Logistic Regression still had to output a class in this case, and it
  defaulted to whichever class had the highest baseline weight
  (`top_up_reverted`) — returning it as an ordinary, confident-looking
  prediction with no indication the model had no real signal to work with.

  **Fix applied:** the `/predict` endpoint now checks whether the
  vectorized input has any non-zero features. If it doesn't, the endpoint
  returns `predicted_intent: "unknown"` and `confident: false` instead of a
  fabricated guess, so callers can distinguish a real prediction from a
  fallback.

  ```json
  // Before the fix
  {"input_text": "2222222.", "predicted_intent": "top_up_reverted"}

  // After the fix
  {"input_text": "2222222.", "predicted_intent": "unknown", "confident": false}
  ```

- This guardrail only catches inputs with **zero** vocabulary overlap. A
  short or ambiguous but still recognizable message can still be
  misclassified with `confident: true`, since the model has no true
  confidence/probability threshold applied — it only flags total absence of
  signal, not low-confidence predictions.
- The classifier is TF-IDF + Logistic Regression, chosen for speed and low
  resource use. A fine-tuned transformer (e.g. DistilBERT) trained on the
  same dataset achieves higher accuracy (~92%) at the cost of a larger model
  and slower CPU inference.

## Tools used

FastAPI • Pydantic • Uvicorn • joblib • scikit-learn • NLTK
