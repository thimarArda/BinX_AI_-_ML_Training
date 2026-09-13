import re
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Load data (same source as the notebook) ---
train_df = pd.read_csv(
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
)
test_df = pd.read_csv(
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
)

# --- Label encoding ---
le = LabelEncoder()
train_df["label"] = le.fit_transform(train_df["category"])
test_df["label"] = le.transform(test_df["category"])

# --- Preprocessing (identical to notebook) ---
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

train_df["processed_text_joined"] = train_df["text"].apply(preprocess_text).apply(lambda t: " ".join(t))
test_df["processed_text_joined"] = test_df["text"].apply(preprocess_text).apply(lambda t: " ".join(t))

# --- Vectorize ---
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df["processed_text_joined"])
X_test = vectorizer.transform(test_df["processed_text_joined"])

y_train = train_df["label"]
y_test = test_df["label"]

# --- Train ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, preds))

# --- Serialize model + every preprocessing object ---
joblib.dump(model, "artifacts/model.joblib")
joblib.dump(vectorizer, "artifacts/vectorizer.joblib")
joblib.dump(le, "artifacts/label_encoder.joblib")

print("Saved model.joblib, vectorizer.joblib, label_encoder.joblib to artifacts/")

# Record one known prediction to check reload consistency against
known_text = "I do not recognize this payment."
known_cleaned = " ".join(preprocess_text(known_text))
known_vec = vectorizer.transform([known_cleaned])
known_pred_idx = model.predict(known_vec)[0]
known_pred_label = le.inverse_transform([known_pred_idx])[0]
print(f"Known check -> {known_text!r} => {known_pred_label}")

with open("artifacts/known_check.txt", "w") as f:
    f.write(f"{known_text}\t{known_pred_label}\n")
