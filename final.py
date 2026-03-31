# ==========================================
# WORD VECTORIZER COMPARISON PROJECT
# ==========================================

import pandas as pd
import re
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score


# ==========================================
# Step 1: Load Dataset
# ==========================================
df = pd.read_csv("IMDB Dataset.csv")

print("\nDataset Loaded ✅")
print(df.head())


# ==========================================
# Step 2: Preprocessing
# ==========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df["review"] = df["review"].apply(clean_text)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

print("\nPreprocessing Done ✅")


# ==========================================
# Step 3: Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

print("\nData Split Done ✅")


# ==========================================
# Step 4: Bag of Words (BoW)
# ==========================================
print("\n--- Bag of Words ---")

start = time.time()

bow_vectorizer = CountVectorizer(max_features=5000)

X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

end = time.time()

bow_time = end - start
print(f"BoW Vectorization Time: {bow_time:.4f} sec")


# Train models on BoW
lr_bow = LogisticRegression(max_iter=1000)
lr_bow.fit(X_train_bow, y_train)

nb_bow = MultinomialNB()
nb_bow.fit(X_train_bow, y_train)

svm_bow = LinearSVC()
svm_bow.fit(X_train_bow, y_train)

# Evaluate
print("\nBoW Results:")
print("LR Accuracy:", accuracy_score(y_test, lr_bow.predict(X_test_bow)))
print("NB Accuracy:", accuracy_score(y_test, nb_bow.predict(X_test_bow)))
print("SVM Accuracy:", accuracy_score(y_test, svm_bow.predict(X_test_bow)))


# ==========================================
# Step 5: TF-IDF
# ==========================================
print("\n--- TF-IDF ---")

start = time.time()

tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

end = time.time()

tfidf_time = end - start
print(f"TF-IDF Vectorization Time: {tfidf_time:.4f} sec")


# Train models on TF-IDF
lr_tfidf = LogisticRegression(max_iter=1000)
lr_tfidf.fit(X_train_tfidf, y_train)

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)

svm_tfidf = LinearSVC()
svm_tfidf.fit(X_train_tfidf, y_train)

# Evaluate
print("\nTF-IDF Results:")
print("LR Accuracy:", accuracy_score(y_test, lr_tfidf.predict(X_test_tfidf)))
print("NB Accuracy:", accuracy_score(y_test, nb_tfidf.predict(X_test_tfidf)))
print("SVM Accuracy:", accuracy_score(y_test, svm_tfidf.predict(X_test_tfidf)))


# ==========================================
# Step 6: Final Comparison
# ==========================================
print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

print(f"BoW Time: {bow_time:.4f} sec")
print(f"TF-IDF Time: {tfidf_time:.4f} sec")

print("\nBest Accuracy (BoW):", max(
    accuracy_score(y_test, lr_bow.predict(X_test_bow)),
    accuracy_score(y_test, nb_bow.predict(X_test_bow)),
    accuracy_score(y_test, svm_bow.predict(X_test_bow))
))

print("Best Accuracy (TF-IDF):", max(
    accuracy_score(y_test, lr_tfidf.predict(X_test_tfidf)),
    accuracy_score(y_test, nb_tfidf.predict(X_test_tfidf)),
    accuracy_score(y_test, svm_tfidf.predict(X_test_tfidf))
))


# ==========================================
# Step 7: User Input (OPTIONAL SCREENSHOT)
# ==========================================
def predict(text):
    text = clean_text(text)
    vec = tfidf_vectorizer.transform([text])
    result = lr_tfidf.predict(vec)[0]
    return "Positive 😊" if result == 1 else "Negative 😡"


print("\n🎬 Try Your Own Review!")

while True:
    user_input = input("\nEnter review (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    print("Prediction:", predict(user_input))