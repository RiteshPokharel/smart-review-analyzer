import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
import numpy as np

df = pd.read_csv('dataset/dataset.csv')
df['label'] = (df['label'] == 'CG').astype(int)
texts = df['text_'].fillna('').tolist()
ratings = df['rating'].tolist()
labels = df['label'].tolist()

X_train_texts, X_test_texts, y_train, y_test, r_train, r_test = train_test_split(
    texts, labels, ratings, test_size=0.2, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
X_train_tfidf = vectorizer.fit_transform(X_train_texts)
X_test_tfidf = vectorizer.transform(X_test_texts)

def stylometric(texts):
    feats = []
    for t in texts:
        words = t.split()
        n = len(words) + 1
        excl = t.count('!') / n
        caps = sum(1 for w in words if w.isupper()) / n
        feats.append([excl, caps])
    return np.array(feats)

analyzer = SentimentIntensityAnalyzer()
def mismatch(texts, ratings):
    feats = []
    for t, r in zip(texts, ratings):
        vader = analyzer.polarity_scores(t)['compound']
        norm_r = (r - 3) / 2
        feats.append([abs(norm_r - vader)])
    return np.array(feats)

scaler1 = MinMaxScaler()
train_stylo = csr_matrix(scaler1.fit_transform(stylometric(X_train_texts)))
test_stylo = csr_matrix(scaler1.transform(stylometric(X_test_texts)))

scaler2 = MinMaxScaler()
train_mis = csr_matrix(scaler2.fit_transform(mismatch(X_train_texts, r_train)))
test_mis = csr_matrix(scaler2.transform(mismatch(X_test_texts, r_test)))

feature_sets = {
    'TF-IDF only':                     (hstack([X_train_tfidf]),                         hstack([X_test_tfidf])),
    'TF-IDF + Stylometric':            (hstack([X_train_tfidf, train_stylo]),             hstack([X_test_tfidf, test_stylo])),
    'TF-IDF + Stylometric + Mismatch': (hstack([X_train_tfidf, train_stylo, train_mis]), hstack([X_test_tfidf, test_stylo, test_mis])),
}

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes':         MultinomialNB(),
    'SVM':                 LinearSVC(),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
}

print(f"{'Model':<22} {'Features':<40} {'F1':>6}")
print("-" * 70)
for model_name, model in models.items():
    for feat_name, (X_tr, X_te) in feature_sets.items():
        model.fit(X_tr, y_train)
        f1 = f1_score(y_test, model.predict(X_te))
        print(f"{model_name:<22} {feat_name:<40} {f1:.3f}")
    print()