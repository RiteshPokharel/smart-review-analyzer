import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix,
                              classification_report, roc_curve, auc,
                              precision_recall_curve)
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import os

os.makedirs('model', exist_ok=True)
os.makedirs('model/plots', exist_ok=True)

# load and clean 
df = pd.read_csv('dataset/dataset.csv')
df = df[['text_', 'rating', 'label']].dropna()
df['label'] = df['label'].map({'OR': 0, 'CG': 1})

print("=" * 70)
print("DATASET STATISTICS")
print("=" * 70)
print(f"Total samples      : {len(df)}")
print(f"Real reviews (OR)  : {(df['label'] == 0).sum()}")
print(f"Fake reviews (CG)  : {(df['label'] == 1).sum()}")
print(f"Class balance      : {df['label'].value_counts(normalize=True).round(3).to_dict()}")

imbalance_ratio = df['label'].value_counts(normalize=True).min()
is_imbalanced   = imbalance_ratio < 0.35
print(f"Dataset            : {'IMBALANCED' if is_imbalanced else 'BALANCED'} "
      f"(minority: {imbalance_ratio:.1%})\n")

# preprocessing 
def truncate(text, limit=150):
    return " ".join(str(text).split()[:limit])

df['text_'] = df['text_'].apply(truncate)

analyzer = SentimentIntensityAnalyzer()

def rating_sentiment_mismatch(row):
    score = analyzer.polarity_scores(str(row['text_']))['compound']
    normalized_rating = (float(row['rating']) - 3) / 2
    return abs(normalized_rating - score)

print("Computing mismatch features...")
df['mismatch'] = df.apply(rating_sentiment_mismatch, axis=1)

def punctuation_features(text):
    words = str(text).split()
    total = len(words) + 1
    exclamation = text.count('!') / total
    caps = sum(1 for w in words if w.isupper()) / total
    return [exclamation, caps]

# train/test split 
texts = df['text_'].values
y     = df['label'].values

X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = train_test_split(
    texts, y, np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
)

# fit TF-IDF only on train (bigrams, 8000 features)
tfidf        = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X_train_text = tfidf.fit_transform(X_train_raw)
X_test_text  = tfidf.transform(X_test_raw)

train_punct   = np.array([punctuation_features(t) for t in X_train_raw])
test_punct    = np.array([punctuation_features(t) for t in X_test_raw])
all_mismatch  = df['mismatch'].values
train_mismatch = all_mismatch[train_idx].reshape(-1, 1)
test_mismatch  = all_mismatch[test_idx].reshape(-1, 1)

X_train = hstack([X_train_text, csr_matrix(train_punct), csr_matrix(train_mismatch)])
X_test  = hstack([X_test_text,  csr_matrix(test_punct),  csr_matrix(test_mismatch)])

cw  = 'balanced' if is_imbalanced else None
spw = (df['label'].value_counts()[0] / df['label'].value_counts()[1]) if is_imbalanced else 1

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight=cw), False),
    "Naive Bayes":         (MultinomialNB(), False),
    "SVM":                 (LinearSVC(max_iter=1000, class_weight=cw), False),
    "Random Forest":       (RandomForestClassifier(n_estimators=100, class_weight=cw), False),
    "Decision Tree":       (DecisionTreeClassifier(class_weight=cw), False),
    "XGBoost":             (XGBClassifier(n_estimators=100, eval_metric='logloss',
                                          verbosity=0, scale_pos_weight=spw), True),
}

def build_features(text_arr, mismatch_arr, tfidf_vec, fit=False):
    X_text = tfidf_vec.fit_transform(text_arr) if fit else tfidf_vec.transform(text_arr)
    punct  = np.array([punctuation_features(t) for t in text_arr])
    mm     = mismatch_arr.reshape(-1, 1)
    return hstack([X_text, csr_matrix(punct), csr_matrix(mm)])

#  1. HOLDOUT 
print("\n" + "=" * 70)
print("1. HOLDOUT (80/20 split)")
print("=" * 70)
print(f"{'Model':<25} {'Accuracy':<12} {'F1':<10} {'Precision':<12} {'Recall':<10} {'AUC':<8}")
print("-" * 78)

holdout_results = {}
best_model      = None
best_model_name = ""
best_f1         = 0
roc_data        = {}

for name, (model, needs_dense) in models.items():
    Xtr = X_train.toarray() if needs_dense else X_train
    Xte = X_test.toarray()  if needs_dense else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    # get probabilities for ROC calibrate linearsvc 
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(Xte)[:, 1]
    else:
        calibrated_model = CalibratedClassifierCV(
            estimator=LinearSVC(max_iter=1000, class_weight=cw),
            cv=5
        )

        calibrated_model.fit(Xtr, y_train)
        y_prob = calibrated_model.predict_proba(Xte)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = auc(fpr, tpr)

    roc_data[name] = (fpr, tpr, auc_score)
    holdout_results[name] = {
        "accuracy": round(acc, 3), "f1": round(f1, 3),
        "precision": round(prec, 3), "recall": round(rec, 3),
        "auc": round(auc_score, 3)
    }
    print(f"{name:<25} {acc:<12.3f} {f1:<10.3f} {prec:<12.3f} {rec:<10.3f} {auc_score:<8.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

print(f"\nBest (Holdout): {best_model_name}  F1={best_f1:.3f}")

# confusion matrix for best model 
Xte_best = X_test.toarray() if isinstance(best_model, XGBClassifier) else X_test
y_pred_best = best_model.predict(Xte_best)
cm = confusion_matrix(y_test, y_pred_best)

print(f"\nConfusion Matrix ({best_model_name}):")
print(f"                 Predicted Real  Predicted Fake")
print(f"Actual Real      {cm[0][0]:<16} {cm[0][1]}")
print(f"Actual Fake      {cm[1][0]:<16} {cm[1][1]}")

print(f"\nClassification Report ({best_model_name}):")
print(classification_report(y_test, y_pred_best, target_names=['Real', 'Fake']))

# save confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set(xticks=[0,1], yticks=[0,1],
       xticklabels=['Real','Fake'], yticklabels=['Real','Fake'],
       xlabel='Predicted', ylabel='Actual',
       title=f'Confusion Matrix — {best_model_name}')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i][j], ha='center', va='center',
                color='white' if cm[i][j] > cm.max()/2 else 'black', fontsize=14)
plt.tight_layout()
plt.savefig('model/plots/confusion_matrix.png', dpi=150)
plt.close()
print("Saved: model/plots/confusion_matrix.png")

# ROC curve 
fig, ax = plt.subplots(figsize=(7, 5))
for name, (fpr, tpr, auc_score) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")
ax.plot([0,1],[0,1],'k--', label='Random')
ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
       title='ROC Curves — All Models')
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('model/plots/roc_curve.png', dpi=150)
plt.close()
print("Saved: model/plots/roc_curve.png")

# Precision-Recall curve
fig, ax = plt.subplots(figsize=(7, 5))
for name, (model, needs_dense) in models.items():
    Xte = X_test.toarray() if needs_dense else X_test

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(Xte)[:, 1]
    else:
        calibrated_model = CalibratedClassifierCV(
            estimator=LinearSVC(max_iter=1000, class_weight=cw),
            cv=5
        )

        calibrated_model.fit(
            X_train.toarray() if needs_dense else X_train,
            y_train
        )

        y_prob = calibrated_model.predict_proba(Xte)[:, 1]

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(rec_arr, prec_arr, label=name)

ax.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curves — All Models')
ax.legend(loc='lower left', fontsize=8)
plt.tight_layout()
plt.savefig('model/plots/pr_curve.png', dpi=150)
plt.close()
print("Saved: model/plots/pr_curve.png")

# 2. K-FOLD 
print("\n" + "=" * 70)
print("2. K-FOLD CROSS VALIDATION (k=5)")
print("=" * 70)
print(f"{'Model':<25} {'Mean Acc':<12} {'Mean F1':<10} {'Std F1':<10}")
print("-" * 55)

kf            = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kfold_results = {}
full_indices  = np.arange(len(df))

for name, (model, needs_dense) in models.items():
    fold_acc, fold_f1 = [], []

    for fold_train_idx, fold_test_idx in kf.split(full_indices, y):
        fresh_model = model.__class__(**model.get_params())

        fold_tfidf   = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
        X_fold_train = build_features(texts[fold_train_idx],
                                      all_mismatch[fold_train_idx], fold_tfidf, fit=True)
        X_fold_test  = build_features(texts[fold_test_idx],
                                      all_mismatch[fold_test_idx],  fold_tfidf, fit=False)
        ytr, yte = y[fold_train_idx], y[fold_test_idx]

        if needs_dense:
            fresh_model.fit(X_fold_train.toarray(), ytr)
            y_pred = fresh_model.predict(X_fold_test.toarray())
        else:
            fresh_model.fit(X_fold_train, ytr)
            y_pred = fresh_model.predict(X_fold_test)

        fold_acc.append(accuracy_score(yte, y_pred))
        fold_f1.append(f1_score(yte, y_pred))

    mean_acc = np.mean(fold_acc)
    mean_f1  = np.mean(fold_f1)
    std_f1   = np.std(fold_f1)
    kfold_results[name] = {"mean_accuracy": round(mean_acc, 3),
                            "mean_f1": round(mean_f1, 3), "std_f1": round(std_f1, 3)}
    print(f"{name:<25} {mean_acc:<12.3f} {mean_f1:<10.3f} {std_f1:<10.3f}")

# 3. BOOTSTRAPPING 
print("\n" + "=" * 70)
print("3. BOOTSTRAPPING (100 iterations)")
print("=" * 70)
print(f"{'Model':<25} {'Mean Acc':<12} {'Mean F1':<10} {'95% CI F1':<15}")
print("-" * 65)

N_BOOTSTRAP       = 100
bootstrap_results = {}

for name, (model, needs_dense) in models.items():
    boot_acc, boot_f1 = [], []

    for i in range(N_BOOTSTRAP):
        fresh_model = model.__class__(**model.get_params())

        boot_idx = resample(np.arange(len(df)), random_state=i)
        oob_idx  = np.array(list(set(range(len(df))) - set(boot_idx)))
        if len(oob_idx) == 0:
            continue

        boot_tfidf  = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
        X_boot_tr   = build_features(texts[boot_idx],  all_mismatch[boot_idx],
                                     boot_tfidf, fit=True)
        X_boot_test = build_features(texts[oob_idx],   all_mismatch[oob_idx],
                                     boot_tfidf, fit=False)

        if needs_dense:
            fresh_model.fit(X_boot_tr.toarray(), y[boot_idx])
            y_pred = fresh_model.predict(X_boot_test.toarray())
        else:
            fresh_model.fit(X_boot_tr, y[boot_idx])
            y_pred = fresh_model.predict(X_boot_test)

        boot_acc.append(accuracy_score(y[oob_idx], y_pred))
        boot_f1.append(f1_score(y[oob_idx], y_pred, zero_division=0))

    mean_acc = np.mean(boot_acc)
    mean_f1  = np.mean(boot_f1)
    ci_low   = np.percentile(boot_f1, 2.5)
    ci_high  = np.percentile(boot_f1, 97.5)

    bootstrap_results[name] = {"mean_accuracy": round(mean_acc, 3),
                                "mean_f1": round(mean_f1, 3),
                                "ci_95": (round(ci_low, 3), round(ci_high, 3))}
    print(f"{name:<25} {mean_acc:<12.3f} {mean_f1:<10.3f} [{ci_low:.3f}, {ci_high:.3f}]")

# save everything 
with open('model/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model/best_model_name.txt', 'w') as f:
    f.write(best_model_name)

all_results = {
    "holdout": holdout_results,
    "kfold": kfold_results,
    "bootstrap": bootstrap_results
}

with open('model/all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print(f"\n{'=' * 70}")
print(f"Best Model (by Holdout F1): {best_model_name}  F1={best_f1:.3f}")
print(f"Plots saved to: model/plots/")
print(f"Models saved to: model/")