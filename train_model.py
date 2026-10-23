"""
train_model.py
Downloads the SMS Spam Collection dataset (if not already present),
trains a Naive Bayes spam classifier using TF-IDF features, and saves
the trained model + vectorizer to disk for use by app.py.
"""
import os
import urllib.request
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_PATH = "data/SMSSpamCollection.tsv"
MODEL_PATH = "model/spam_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
METRICS_PATH = "model/metrics.json"
DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


def ensure_dataset():
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        print("Dataset not found locally — downloading...")
        urllib.request.urlretrieve(DATASET_URL, DATA_PATH)
        print(f"Downloaded dataset to {DATA_PATH}")


def main():
    ensure_dataset()
    os.makedirs("model", exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    print(f"Loaded {len(df)} messages ({(df.label=='spam').sum()} spam, {(df.label=='ham').sum()} ham)")

    X = df["message"]
    y = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- Evaluation on held-out test set ---")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {cm}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "confusion_matrix": cm,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved vectorizer -> {VECTORIZER_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()