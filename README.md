# SemEval-2026 Task 13 — Subtask A: Binary Machine-Generated Code Detection

**Team:** FMI_SU_Yotkova_Kastreva  
**Authors:** Elitsa Yotkova, Violeta Kastreva  
**Result:** Macro-F1 **0.67347** — Rank **13 / 82** (baseline: 0.308)

---

## Problem

Given a code snippet, predict whether it was **fully human-written** or **fully machine-generated**.

| Split | Samples |
|-------|---------|
| Train | 500K (238K human / 262K AI) |
| Validation | 100K |
| Test | 500K |

**Training languages:** C++, Python, Java (algorithmic domain).  
**Evaluation includes unseen languages** (Go, PHP, C#, C, JS) **and unseen domains** (research, production).

---

## Approach

Pretrained code-model embeddings (GraphCodeBERT, CodeT5+, CodeRankLLM) alone overfit badly on this task (test F1 0.25–0.49). Our winning approach uses **custom ratio-based features** extracted via AST parsing and two purpose-built auxiliary classifiers, combined with lightweight ML models and rule-based heuristics.

### Key ideas

1. **Ratio-based features** instead of raw counts — code snippets range from 1 to 100+ lines, so absolute counts are unreliable. We compute ratios like `comment_ratio`, `verb_comment_ratio`, and `text_like_ratio`.
2. **Two auxiliary classifiers** built from scratch:
   - **Text-vs-code line classifier** — distinguishes natural-language lines from code lines inside a snippet (TF-IDF char n-grams + Logistic Regression, trained on a custom 52K-sample dataset from StackOverflow + Twitch Chat).
   - **Language guesser** — predicts the programming language from a snippet (TF-IDF char n-grams + Multinomial Naive Bayes, trained on Rosetta Code, accuracy 0.94).
3. **Tree-sitter AST parsing** across 7 languages for precise comment extraction, with spaCy POS tagging to compute verb ratios in comments.
4. **Size-aware bucketing** — snippets are split into small (<20 LOC), medium (20–70), and large (>70) buckets, each with its own scaler.
5. **Final classifier** — Decision Tree (max_depth=2) on `comment_ratio` + `verb_comment_ratio`, augmented with heuristic rules on `text_like_ratio` and first-line language detection.

---

## Repository Structure

```
SemEvalTask13-SubtaskA/
│
├── data/
│   ├── semeval/                        # SemEval competition data
│   │   ├── train.parquet               # 500K training samples
│   │   ├── validation.parquet          # 100K validation samples
│   │   ├── test_new.parquet            # Test set
│   │   ├── processed/                  # Pre-computed feature CSVs
│   │   │   ├── train_feats_300k.csv
│   │   │   ├── val_feats.csv
│   │   │   ├── test_feats.csv
│   │   │   └── ...
│   │   └── additional-test-sets/       # Cross-domain test data
│   ├── text-vs-code/                   # Text-vs-code classifier dataset
│   │   ├── gamer.csv                   # Twitch Chat (text samples)
│   │   └── StackOverflow_questions_*.csv
│   └── submissions/                    # Final submission CSVs
│
├── models/
│   ├── ai_detector.joblib              # Trained AI detection model (LogReg, F1=0.62)
│   ├── textvscode_classifier.joblib    # Text-vs-code line classifier
│   └── language_classifier.joblib      # Programming language classifier
│
├── app/                                # Streamlit demo application
│   ├── app.py                          # Main UI — paste code, get prediction
│   ├── requirements.txt                # App dependencies
│   ├── src/
│   │   ├── features.py                 # Feature extraction from code snippets
│   │   ├── pipeline.py                 # Model loading & inference pipeline
│   │   ├── adapters.py                 # LangModel / DetectorModel interfaces
│   │   └── explain.py                  # Feature-based reasoning display
│   └── .streamlit/config.toml
│
├── data_analysis.ipynb                 # EDA on train/val/test sets
├── feature_analysis.ipynb              # Feature extraction & analysis pipeline
├── text_vs_code_line_classifier.ipynb  # Train the text-vs-code classifier
├── language_classifier.ipynb           # Train the language guesser (Rosetta Code)
├── models_with_features.ipynb          # Train ML classifiers on custom features
├── models_with_embeddings.ipynb        # Train classifiers on transformer embeddings
└── training-codebert.ipynb             # GraphCodeBERT embedding generation & models
```

---

## Notebooks

### `data_analysis.ipynb`
Exploratory data analysis of the SemEval datasets. Visualizes label distributions, language distributions, code length statistics (character count, LOC, token count), and cross-split comparisons.

### `text_vs_code_line_classifier.ipynb`
Builds the binary text-vs-code classifier from a custom dataset:
- **Text samples:** StackOverflow question bodies (HTML-parsed with BeautifulSoup) + Twitch Chat messages.
- **Code samples:** `<pre><code>` blocks from the same StackOverflow data.
- **Model:** TF-IDF (3,5)-char n-grams → Logistic Regression.
- **Performance:** Validation accuracy 0.97, F1-macro 0.956.
- **Output:** `models/textvscode_classifier.joblib`

### `language_classifier.ipynb`
Builds the programming language classifier:
- **Data:** Scraped from Rosetta Code (MediaWiki API) for 7 languages.
- **Model:** Heuristic regex rules + TF-IDF (3,8)-char n-grams → Multinomial Naive Bayes.
- **Performance:** Accuracy 0.94.
- **Output:** `models/language_classifier.joblib`

### `feature_analysis.ipynb`
Extracts semantic features from code snippets using the two auxiliary classifiers above:
- Loads tree-sitter parsers for Python, JavaScript, Java, C#, PHP, C/C++, Go.
- Extracts comments via AST, classifies lines as text/code, POS-tags comment words with spaCy.
- Computes per-sample features: `comment_ratio`, `verb_comment_ratio`, `text_like_ratio`, and others.
- Saves feature CSVs to `data/semeval/processed/`.
- Includes feature distribution analysis, correlation heatmaps, and per-label comparisons.

### `models_with_features.ipynb`
Trains and evaluates classifiers on the extracted features:
- Loads feature CSVs, engineers LOC-based buckets (small / medium / large), applies per-bucket StandardScaler.
- Trains Logistic Regression, Decision Tree, Random Forest, MLP, LinearSVC.
- Performs threshold tuning and rule-based post-processing (`text_like_ratio` heuristic, first-line language name detection).
- Exports the best-performing model (LogReg with 10 features) to `models/ai_detector.joblib` for use in the Streamlit app.
- Generates final submission CSVs.

### `models_with_embeddings.ipynb` / `training-codebert.ipynb`
Embedding-based pipelines for three pretrained code models:
- **GraphCodeBERT** (`microsoft/graphcodebert-base`) — [CLS] token embeddings, 768-dim.
- **CodeT5+** (`Salesforce/codet5p-110m-embedding`) — encoder embeddings.
- **CodeRankLLM** (`nomicai/CodeRankLLM`) — encoder embeddings.

For each model: generate embeddings in chunks → train LR / XGBoost / LightGBM / MLP → optimize ensemble weights (SLSQP) → generate submissions. These models overfit on the test set (F1 0.25–0.49) and were ultimately not used in the final submission.

---

## Streamlit App

An interactive demo for AI-generated code detection:

```bash
python -m streamlit run app/app.py
```

**Features:**
- Paste any code snippet and get instant predictions
- Shows **AI-generated** or **Human-written** classification with probability
- Detects programming language automatically
- Displays feature-level reasoning explaining the prediction
- Uses the trained LogisticRegression model (`models/ai_detector.joblib`)

The app extracts 10 semantic features from the code and passes them through the trained classifier. If spaCy is not installed, the app falls back to heuristic-based verb ratio computation.

---

## Features

The trained model uses 10 semantic features extracted from code snippets:

| Feature | Description |
|---------|-------------|
| `comment_ratio` | Comment lines / total lines (via tree-sitter AST) |
| `verb_ratio_comments` | Verb tokens / total word tokens in comments (via spaCy or heuristic) |
| `text_like_ratio` | Text-classified lines / code-classified lines (via text-vs-code classifier) |
| `comments_code_like_ratio_to_total` | Code-like comment lines / total lines |
| `comments_text_like_ratio_to_total` | Text-like comment lines / total lines |
| `comments_code_like_ratio_comments` | Code-like comment lines / total comment lines |
| `comments_text_like_ratio_comments` | Text-like comment lines / total comment lines |
| `bucket_small` | 1 if snippet < 20 LOC, else 0 |
| `bucket_medium` | 1 if snippet 20–70 LOC, else 0 |
| `bucket_large` | 1 if snippet > 70 LOC, else 0 |

**Findings:** AI-generated code has higher `text_like_ratio` (+0.09), higher `comment_ratio` (+0.14), and higher `error_near_eof_ratio` (+0.08) compared to human-written code.

---

## Setup

### Full Environment (for notebooks)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install pandas numpy scikit-learn matplotlib seaborn
pip install torch transformers
pip install xgboost lightgbm
pip install joblib pyarrow
pip install tree-sitter tree-sitter-python tree-sitter-java tree-sitter-javascript
pip install tree-sitter-cpp tree-sitter-c-sharp tree-sitter-go tree-sitter-php
pip install spacy && python -m spacy download en_core_web_sm
pip install streamlit          # for the demo app
```

### Streamlit App Only

```bash
cd app
pip install -r requirements.txt
python -m streamlit run app.py
```

> **Note:** spaCy is optional for the Streamlit app — verb ratio computation falls back to heuristics if spaCy is unavailable. For full notebook functionality, spaCy requires Python ≤ 3.13.
