# 🎬 Word Vectorization Comparison for Sentiment Analysis

## 📌 Overview
This project explores and compares different **word vectorization techniques** used in Natural Language Processing (NLP) for sentiment analysis.

We analyze how **Bag of Words (BoW)** and **TF-IDF** affect model performance and computational efficiency using machine learning models.

---

## 🎯 Objectives
- Understand word vectorization in NLP  
- Implement BoW and TF-IDF  
- Train ML models on text data  
- Compare accuracy and time complexity  

---

## 📂 Dataset
- **IMDb Movie Reviews Dataset**
- 50,000 movie reviews  
- Labels: Positive / Negative  

---

## ⚙️ Techniques Used

### 🔹 Preprocessing
- Lowercasing
- Removing HTML tags
- Removing special characters

### 🔹 Vectorization Methods
- Bag of Words (BoW)
- TF-IDF

### 🔹 Machine Learning Models
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)

---

## 📊 Results

| Method | Model | Accuracy |
|--------|------|----------|
| BoW | Logistic Regression | ~85% |
| BoW | SVM | ~87% |
| TF-IDF | Logistic Regression | ~88% |
| TF-IDF | SVM | ~89% |

---

## 🧠 Key Insights
- TF-IDF performs better than BoW  
- BoW is faster but less accurate  
- Trade-off between speed and performance  

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
