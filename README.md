# Smart Review Analyzer

A Chrome extension that detects fake and AI-generated Google Maps reviews, analyzes sentiment from genuine reviews only, and summarizes what real customers actually think.

---

## Why I built this

Most review analyzers and sentiment tools process all reviews — including fake ones. This skews the output. A place with 50 paid 5-star reviews will show as highly positive even if real customers had a terrible experience.

Smart Review Analyzer solves this by filtering out fake and suspicious reviews first, then running sentiment analysis and summary generation only on what's left. The result is a more honest picture of what genuine customers think.

I built this as a college minor project, but also because this is something I've genuinely wanted to build for a long time.

---

## What it does

- **Fake Review Detection** — classifies each review as Real, Fake, or Suspicious using a trained ML classifier
- **Sentiment Analysis** — calculates positive and negative sentiment based on real reviews only
- **Summary Generation** — generates a 2-3 sentence summary of what genuine customers think, powered by Groq (LLaMA 3.3)

---

## How it works

```
Google Maps Page
      ↓
Chrome Extension scrapes reviews (text + rating + date)
      ↓
Preprocessing (truncate to 150 words, auto-flag under 5 words)
      ↓
Feature Extraction
  → TF-IDF (text patterns)
  → Punctuation Density (exclamation marks, caps ratio)
  → Rating-Sentiment Mismatch
      ↓
SVM Classifier → Real / Fake / Suspicious
      ↓
Fake + Suspicious filtered out
      ↓
Sentiment Analysis on Real reviews only (VADER)
      ↓
Summary Generation (Groq API — LLaMA 3.3)
      ↓
Results displayed in Extension popup
```

---

## Model Comparison

Five models were trained and evaluated on the same dataset. The best performing model was selected based on F1 score.

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 89.8% | 0.897 |
| Naive Bayes | 87.6% | 0.879 |
| **SVM** | **90.6%** | **0.905** |
| Random Forest | 88.6% | 0.885 |
| Decision Tree | 75.3% | 0.752 |

---

## Why this is different

Other sentiment analyzers and summarizers don't remove fake reviews before processing. This means their output reflects a mix of real and fabricated opinions. Smart Review Analyzer removes fake and suspicious reviews first — so the sentiment score and summary reflect only what genuine customers said.

---

## Research Backing

Each feature used in the classifier is backed by published research:

- **Rating-Sentiment Mismatch** — Shan et al. (2021), ScienceDirect — https://www.sciencedirect.com/science/article/abs/pii/S0167923621000233
- **Punctuation & Linguistic Features** — Abri et al. (2020), arXiv — https://arxiv.org/abs/2010.04260
- **Temporal Burst Patterns** — arXiv — https://arxiv.org/pdf/1611.06625
- **Semantic Similarity Detection** — Sandulescu & Ester, arXiv — https://arxiv.org/pdf/1609.02727

---

## Project Structure

```
smart-review-analyzer/
│
├── dataset/          # Fake review dataset (not included in repo)
├── model/            # Saved trained model files (not included in repo)
├── backend/
│   └── app.py        # Flask API
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── content.js
└── train.py          # Model training script
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/RiteshPokharel/smart-review-analyzer.git
cd smart-review-analyzer
```

### 2. Install dependencies
```bash
pip install scikit-learn pandas numpy flask flask-cors nltk vaderSentiment scipy groq python-dotenv
```

### 3. Add dataset
Download the fake reviews dataset and place it in the `dataset/` folder as `dataset.csv`.

### 4. Train the model
```bash
python train.py
```

### 5. Add your Groq API key
Create a `.env` file in the root:
```
GROQ_API_KEY=your_key_here
```

### 6. Run the Flask backend
```bash
python backend/app.py
```

### 7. Load the Chrome Extension
- Go to `chrome://extensions`
- Enable Developer Mode
- Click Load Unpacked
- Select the `extension/` folder

---

## Tech Stack

- Python, scikit-learn, Flask
- VADER Sentiment Analysis
- Groq API (LLaMA 3.3) for summary generation
- Chrome Extension (HTML, CSS, JavaScript)

---

## License

MIT