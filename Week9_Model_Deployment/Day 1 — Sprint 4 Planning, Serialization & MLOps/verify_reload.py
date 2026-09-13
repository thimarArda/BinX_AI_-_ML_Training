import re
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- Load artifacts from disk only (nothing in memory from training) ---
model = joblib.load("artifacts/model.joblib")
vectorizer = joblib.load("artifacts/vectorizer.joblib")
le = joblib.load("artifacts/label_encoder.joblib")

# --- Same preprocessing must be reimplemented identically in the serving app ---
stop_words = set(stopwords.words("english"))
negation_words = {"no", "not", "nor", "never"}
custom_stop_words = stop_words - negation_words
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in custom_stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def predict(raw_text):
    cleaned = " ".join(preprocess_text(raw_text))
    vec = vectorizer.transform([cleaned])
    pred_idx = model.predict(vec)[0]
    return le.inverse_transform([pred_idx])[0]

# --- Compare against the known result recorded during training ---
with open("artifacts/known_check.txt") as f:
    known_text, expected_label = f.read().strip().split("\t")

actual_label = predict(known_text)

print(f"Input:    {known_text!r}")
print(f"Expected: {expected_label}")
print(f"Actual:   {actual_label}")
assert actual_label == expected_label, "MISMATCH — training/serving skew detected!"
print("MATCH — serialization verified, no skew.")

# A couple extra sanity checks
for text in ["Why was my card payment declined?", "How do I top up my account?"]:
    print(f"{text!r} -> {predict(text)}")
