"""
app.py
Flask backend for SpamShield. Loads the trained Naive Bayes + TF-IDF model
and serves:
  - GET  /                 -> the web UI
  - POST /api/predict      -> classify a message, return verdict + score + explanation
  - GET  /api/history      -> list of past scans (in-memory, per server run)
  - GET  /api/metrics      -> model evaluation metrics (from training)
"""
import json
import re
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")
with open("model/metrics.json") as f:
    METRICS = json.load(f)

feature_names = np.array(vectorizer.get_feature_names_out())
spam_log_prob = model.feature_log_prob_[1]
ham_log_prob = model.feature_log_prob_[0]
word_spamminess = dict(zip(feature_names, spam_log_prob - ham_log_prob))

HISTORY = []


def analyze_message(text):
    vec = vectorizer.transform([text])
    proba = model.predict_proba(vec)[0]
    spam_prob = float(proba[1])
    score = round(spam_prob * 100, 1)
    is_spam = spam_prob >= 0.5

    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    seen = set()
    contributions = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        if tok in word_spamminess:
            contributions.append((tok, word_spamminess[tok]))
    contributions.sort(key=lambda x: x[1], reverse=True)
    top_words = [w for w, val in contributions[:6] if val > 0]

    return {
        "isSpam": bool(is_spam),
        "score": score,
        "flaggedWords": top_words,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = (data.get("message") or "").strip()
    if not text:
        return jsonify({"error": "message is required"}), 400

    result = analyze_message(text)
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": text[:120],
        "verdict": "spam" if result["isSpam"] else "safe",
        "score": result["score"],
    }
    HISTORY.insert(0, entry)
    del HISTORY[50:]

    return jsonify(result)


@app.route("/api/history")
def history():
    return jsonify(HISTORY)


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


if __name__ == "__main__":
    app.run(debug=False, port=5000)