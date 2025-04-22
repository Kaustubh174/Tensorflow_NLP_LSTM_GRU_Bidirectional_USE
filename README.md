# 🧠 NLP Disaster Tweet Classification with TensorFlow, USE, and RNNs

This repository contains a comprehensive Natural Language Processing (NLP) project using **TensorFlow**, **TensorFlow Hub**, and **Scikit-learn** to classify disaster-related tweets.

The dataset used is **[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)** from Kaggle, where the goal is to predict whether a given tweet is about a real disaster or not.

---

## 📦 Dataset
- **Source:** Kaggle Competition — *NLP with Disaster Tweets*
- **Task:** Binary classification (`1 = real disaster`, `0 = not disaster`)
- **Fields:** `id`, `keyword`, `location`, `text`, `target`

---

## 🔍 Project Overview

| Model Type            | Description                                       |
|-----------------------|---------------------------------------------------|
| Baseline Model        | `TfidfVectorizer` + `MultinomialNB` (Scikit-learn) |
| FFNN Model 1          | Embedding + `GlobalAveragePooling1D`              |
| FFNN Model 2          | Embedding + `GlobalMaxPooling1D`                  |
| USE Model             | Pre-trained Universal Sentence Encoder from TF Hub |
| RNN (LSTM)            | Single LSTM layer model                           |
| RNN (Stacked LSTM)    | Two stacked LSTM layers + Dense                   |
| RNN (GRU)             | Single GRU layer model                            |

---

## 🚀 Highlights

- ✅ Clean implementation of baseline and advanced models
- ✅ Real disaster tweet dataset with preprocessing, training, and evaluation
- ✅ TF-IDF + Naive Bayes baseline for quick benchmarking
- ✅ Embedding visualization for TensorBoard/Projector
- ✅ Use of pre-trained **Universal Sentence Encoder (USE)**
- ✅ RNN-based models: **LSTM**, **Stacked LSTM**, **GRU**
- ✅ Model comparison using `accuracy`, `precision`, `recall`, and `F1-score`

---

## 🛠️ Dependencies

```bash
pip install tensorflow tensorflow-hub scikit-learn pandas matplotlib
