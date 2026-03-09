from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

with open('model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model/best_model_name.txt', 'r') as f:
    model_name = f.read()

analyzer = SentimentIntensityAnalyzer()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def truncate(text, limit=150):
    return " ".join(str(text).split()[:limit])

def punctuation_features(text):
    words = str(text).split()
    total = len(words) + 1
    exclamation = text.count('!') / total
    caps = sum(1 for w in words if w.isupper()) / total
    return [exclamation, caps]

def rating_sentiment_mismatch(text, rating):
    score = analyzer.polarity_scores(str(text))['compound']
    normalized_rating = (float(rating) - 3) / 2
    return abs(normalized_rating - score)

def classify_review(text, rating):
    if len(text.split()) < 5:
        return "Suspicious"
    text = truncate(text)
    tfidf_features = tfidf.transform([text])
    punct = np.array([punctuation_features(text)])
    mismatch = np.array([[rating_sentiment_mismatch(text, rating)]])
    X = hstack([tfidf_features, csr_matrix(punct), csr_matrix(mismatch)])
    prediction = model.predict(X)[0]
    return "Fake" if prediction == 1 else "Real"

def get_sentiment_label(text):
    score = analyzer.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def get_overall_sentiment(reviews):
    scores = [analyzer.polarity_scores(r)['compound'] for r in reviews]
    avg = sum(scores) / len(scores) if scores else 0
    positive = round((avg + 1) / 2 * 100)
    negative = 100 - positive
    return positive, negative

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    reviews = data.get('reviews', [])

    results = []
    real_reviews = []

    for r in reviews:
        text = r.get('text', '')
        rating = r.get('rating', 3)
        label = classify_review(text, rating)
        sentiment = get_sentiment_label(text)
        results.append({"text": text, "label": label, "sentiment": sentiment})
        if label == "Real":
            real_reviews.append(text)

    positive, negative = get_overall_sentiment(real_reviews) if real_reviews else (0, 0)

    return jsonify({
        "reviews": results,
        "counts": {
            "total": len(results),
            "real": sum(1 for r in results if r['label'] == 'Real'),
            "fake": sum(1 for r in results if r['label'] == 'Fake'),
            "suspicious": sum(1 for r in results if r['label'] == 'Suspicious')
        },
        "sentiment": {
            "positive": positive,
            "negative": negative
        },
        "real_reviews": real_reviews
    })

@app.route('/summary', methods=['POST'])
def summary():
    data = request.json
    real_reviews = data.get('real_reviews', [])

    if not real_reviews:
        return jsonify({"summary": "Not enough real reviews to summarize."})

    top_reviews = real_reviews[:10]
    combined = "\n".join([f"- {r}" for r in top_reviews])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": f"Based on these genuine customer reviews, write a 2-3 sentence summary of what people honestly think about this place. Be concise and neutral.\n\nReviews:\n{combined}\n\nSummary:"
            }
        ]
    )

    return jsonify({"summary": response.choices[0].message.content.strip()})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)