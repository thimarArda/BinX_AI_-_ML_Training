# Banking77 Intent Classifier — FastAPI Serving + Streamlit Demo

A REST API that serves a trained Banking77 intent classification model,
plus an interactive Streamlit demo built on top of it. Send it a customer
banking message (e.g. *"I lost my card"*) and it returns the predicted
intent (e.g. `lost_or_stolen_card`) as JSON — or, via the Streamlit app,
as a friendly UI anyone can try without touching the API directly.

This project builds on a previously serialized model (TF-IDF + Logistic
Regression, ~85.6% accuracy, trained on the
[Banking77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data))
by wrapping it in a live FastAPI service — turning a saved model file into
something any application (a website, mobile app, or another backend) can
actually call. The Streamlit app wraps that same model in a non-technical,
point-and-click interface, suitable for a live demo to a mentor, recruiter,
or stakeholder.

## What's inside

```
.
├── main.py              # FastAPI app: loads the model and exposes /predict
├── streamlit_app.py      # Streamlit demo: interactive UI for the same model
├── test_api.py           # Script to manually test the running API
├── requirements.txt      # Pinned dependencies
└── artifacts/
    ├── model.joblib            # Trained LogisticRegression classifier
    ├── vectorizer.joblib       # Fitted TF-IDF vectorizer
    ├── label_encoder.joblib    # Maps the 77 intent labels <-> class indices
    └── known_check.txt         # Known input/output pair used for verification
```

The artifacts are the same serialized files produced and verified in the
previous serialization step — they were not retrained here, and both
`main.py` and `streamlit_app.py` load them exactly as-is.

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
   This installs everything needed for both the FastAPI service and the
   Streamlit demo, including Streamlit itself.

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

## Running the Streamlit Demo

The Streamlit app (`streamlit_app.py`) loads the same three artifacts as
the FastAPI service and runs the exact same preprocessing and guardrail
logic — it's the same model, just presented as a point-and-click UI
instead of a JSON API. No separate server or API call is needed; the app
loads the model artifacts directly.

Start it with:

```bash
streamlit run streamlit_app.py
```

Run this from the project's root folder (the one containing `artifacts/`),
so the app can find `artifacts/model.joblib` and the other files using
their relative paths.

Streamlit will print a local URL in the terminal and should open it in
your default browser automatically:

```
http://localhost:8501
```

If that port is already in use, Streamlit automatically picks the next
free one and prints the URL it's using instead.

### Using the app

1. Type a customer banking message into the text box (e.g. *"I lost my
   card and need a replacement"*).
2. Click **Predict intent**.
3. The predicted intent appears in a green success box.
4. Below it, a bar chart shows the model's top 5 candidate intents by
   probability — useful for seeing how confident the model actually was,
   not just which label won.

### Guardrail behavior (same as the API)

If the message shares no vocabulary at all with the training data
(gibberish like `"asdkjh"`, or all-numeric input), the app shows a warning
instead of a fabricated prediction — the same `confident: false` case
documented for the API below, just surfaced as an `st.warning` message
rather than a JSON field.

### Stopping the app

Go back to the terminal running Streamlit and press `Ctrl+C`.

## Input validation

The `/predict` endpoint uses Pydantic to validate every request. Invalid
input is rejected with a clear `422 Unprocessable Entity` response instead
of crashing the model:

| Invalid input | Example | Response |
|---|---|---|
| Wrong type | `{"text": 12345}` | `422`, "Input should be a valid string" |
| Missing field | `{}` | `422`, "Field required" |
| Empty string | `{"text": ""}` | `422`, "String should have at least 1 character" |

The Streamlit app doesn't use Pydantic (it's a UI, not an API), but
handles the equivalent cases directly: an empty text box shows a prompt
to type a message first, and whitespace-only or punctuation-only input is
caught by the same "no usable words after preprocessing" check used in
`main.py`.

## Known limitations

- **No out-of-distribution detection at the vocabulary level was originally
  present** — early testing found that any input with no vocabulary overlap
  with the training data (gibberish like `"hhhhh"`, numbers like
  `"2222222"`, random keyboard input) produced an all-zero TF-IDF vector.
  Logistic Regression still had to output a class in this case, and it
  defaulted to whichever class had the highest baseline weight
  (`top_up_reverted`) — returning it as an ordinary, confident-looking
  prediction with no indication the model had no real signal to work with.

  **Fix applied:** both the `/predict` endpoint and the Streamlit app now
  check whether the vectorized input has any non-zero features. If it
  doesn't, they report `predicted_intent: "unknown"` / a warning message
  and `confident: false` instead of a fabricated guess, so callers (or
  demo viewers) can distinguish a real prediction from a fallback.

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
  signal, not low-confidence predictions. The Streamlit app's probability
  bar chart makes this visible at a glance (e.g. a close spread across the
  top 5 candidates signals genuine ambiguity), even though the underlying
  guardrail doesn't act on it.
- The classifier is TF-IDF + Logistic Regression, chosen for speed and low
  resource use. A fine-tuned transformer (e.g. DistilBERT) trained on the
  same dataset achieves higher accuracy (~92%) at the cost of a larger model
  and slower CPU inference.

## Tools used

FastAPI • Streamlit • Pydantic • Uvicorn • joblib • scikit-learn • NLTK