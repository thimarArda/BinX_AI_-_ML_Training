import re

import joblib
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- Page setup ---
st.set_page_config(page_title="Banking Intent Classifier", page_icon="🏦")


# --- Load the same serialized artifacts the FastAPI service uses ---
# @st.cache_resource means this only runs once, not on every widget interaction
# (Streamlit re-runs the whole script top-to-bottom every time you touch a widget).
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/model.joblib")
    vectorizer = joblib.load("artifacts/vectorizer.joblib")
    label_encoder = joblib.load("artifacts/label_encoder.joblib")
    return model, vectorizer, label_encoder


model, vectorizer, label_encoder = load_artifacts()

# --- Same preprocessing used at training/serving time (Day 1 & 2) ---
stop_words = set(stopwords.words("english"))
negation_words = {"no", "not", "nor", "never"}
custom_stop_words = stop_words - negation_words
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in custom_stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens


# --- UI ---
st.title("🏦 Banking Intent Classifier")
st.write(
    "Type a customer banking message below and the model will predict what "
    "the customer is trying to do — e.g. report a lost card, dispute a "
    "payment, or top up an account."
)

message = st.text_area(
    "Customer message",
    placeholder="e.g. I lost my card and need a replacement",
    height=100,
)

if st.button("Predict intent", type="primary"):
    if not message.strip():
        st.warning("Type a message first.")
    else:
        cleaned_tokens = preprocess_text(message)
        cleaned_text = " ".join(cleaned_tokens)

        if not cleaned_text.strip():
            st.error(
                "That input has no usable words after preprocessing — try "
                "rephrasing."
            )
        else:
            X = vectorizer.transform([cleaned_text])

            # Same guardrail as the FastAPI /predict endpoint: an all-zero
            # TF-IDF vector means no real signal, so don't show a fabricated
            # confident-looking guess.
            if X.nnz == 0:
                st.warning(
                    "⚠️ This message shares no vocabulary with the training "
                    "data, so the model has no real signal here. Showing "
                    "this as **unknown / not confident** rather than a "
                    "guess."
                )
            else:
                pred_idx = model.predict(X)[0]
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                st.success(f"**Predicted intent:** `{pred_label}`")

                # --- Supporting visualization: top-5 predicted probabilities ---
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    top5_idx = proba.argsort()[-5:][::-1]
                    top5_labels = label_encoder.inverse_transform(top5_idx)
                    top5_scores = proba[top5_idx]

                    st.caption("Top 5 candidate intents by model confidence")
                    chart_df = pd.DataFrame(
                        {"probability": top5_scores}, index=top5_labels
                    )
                    st.bar_chart(chart_df)
