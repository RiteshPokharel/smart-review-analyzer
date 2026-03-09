import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import os

# load data
df = pd.read_csv('dataset/dataset.csv')
df = df[['text_', 'rating', 'label']].dropna()

# OR = real (0), CG = fake (1)
df['label'] = df['label'].map({'OR': 0, 'CG': 1})

# cap at 150 words
def truncate(text, limit=150):
    return " ".join(str(text).split()[:limit])

df['text_'] = df['text_'].apply(truncate)

# convert text to numbers
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df['text_'])

# exclamation and caps ratio per review
def punctuation_features(text):
    words = str(text).split()
    total = len(words) + 1
    exclamation = text.count('!') / total
    caps = sum(1 for w in words if w.isupper()) / total
    return [exclamation, caps]

punct_features = np.array([punctuation_features(t) for t in df['text_']])

# check if rating matches sentiment
analyzer = SentimentIntensityAnalyzer()

def rating_sentiment_mismatch(row):
    score = analyzer.polarity_scores(str(row['text_']))['compound']
    normalized_rating = (float(row['rating']) - 3) / 2
    return abs(normalized_rating - score)

print("Computing mismatch features...")
df['mismatch'] = df.apply(rating_sentiment_mismatch, axis=1)
mismatch_features = df['mismatch'].values.reshape(-1, 1)

# combine all features
X = hstack([X_text, csr_matrix(punct_features), csr_matrix(mismatch_features)])
y = df['label'].values

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train and compare 5 models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes":         MultinomialNB(),
    "SVM":                 LinearSVC(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100),
    "Decision Tree":       DecisionTreeClassifier()
}

results = {}
best_model = None
best_model_name = ""
best_f1 = 0

print("\nModel Comparison\n")
print(f"{'Model':<25} {'Accuracy':<12} {'F1':<10} {'Precision':<12} {'Recall':<10}")
print("-" * 70)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    results[name] = {
        "accuracy": round(acc, 3),
        "f1": round(f1, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3)
    }

    print(f"{name:<25} {acc:<12.3f} {f1:<10.3f} {prec:<12.3f} {rec:<10.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name} with F1: {best_f1:.3f}")

# save best model and vectorizer
os.makedirs('model', exist_ok=True)

with open('model/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model/results.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('model/best_model_name.txt', 'w') as f:
    f.write(best_model_name)
