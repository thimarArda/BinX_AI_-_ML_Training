import re
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = FastAPI(
    title="Banking77 Intent Classifier",
    description="Predicts the customer intent behind a banking-related message.",
    version="1.0.0",
)

# --- Load the Day 1 serialized artifacts ---
model = joblib.load("artifacts/model.joblib")
vectorizer = joblib.load("artifacts/vectorizer.joblib")
label_encoder = joblib.load("artifacts/label_encoder.joblib")

# --- Same preprocessing used at training time (Day 1) — loaded logic, not reinvented ---
stop_words = set(stopwords.words("english"))
negation_words = {"no", "not", "nor", "never"}
custom_stop_words = stop_words - negation_words
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in custom_stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


# --- Pydantic input schema: validates every incoming request ---
class InputData(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="A customer banking message, e.g. 'I lost my card'.",
        examples=["I do not recognize this payment."],
    )


class PredictionResponse(BaseModel):
    input_text: str
    predicted_intent: str
    confident: bool


@app.get("/")
def root():
    return {"message": "Banking77 Intent Classifier API. Visit /docs to test the /predict endpoint."}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    cleaned_tokens = preprocess_text(data.text)
    cleaned_text = " ".join(cleaned_tokens)

    if not cleaned_text.strip():
        # Text became empty after preprocessing (e.g. input was only punctuation)
        raise HTTPException(
            status_code=422,
            detail="Input text contained no usable words after preprocessing.",
        )

    X = vectorizer.transform([cleaned_text])

    # Guardrail: if the input shares no vocabulary with training data, the
    # TF-IDF vector is all zeros and the model has no real signal to work
    # with. Rather than silently returning its default fallback class as if
    # it were a confident prediction, flag it explicitly.
    if X.nnz == 0:
        return PredictionResponse(
            input_text=data.text,
            predicted_intent="unknown",
            confident=False,
        )

    pred_idx = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    return PredictionResponse(
        input_text=data.text,
        predicted_intent=pred_label,
        confident=True,
    )
