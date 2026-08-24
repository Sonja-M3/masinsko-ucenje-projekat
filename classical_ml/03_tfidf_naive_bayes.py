import pandas as pd
import matplotlib.pyplot as plt

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay


# Ucitavanje ociscenih podataka
train = pd.read_csv("data/processed/train.txt", sep=";", header=None, names=["text", "emotion"])
val = pd.read_csv("data/processed/val.txt", sep=";", header=None, names=["text", "emotion"])

x_train = train["text"]
y_train = train["emotion"]
x_val = val["text"]
y_val = val["emotion"]


# TF-IDF reprezentacija teksta
tfidf = TfidfVectorizer()
x_train_tfidf = tfidf.fit_transform(x_train)
x_val_tfidf = tfidf.transform(x_val)


# Treniranje Multinomial Naive Bayes modela
model = MultinomialNB(alpha=1.0)
model.fit(x_train_tfidf, y_train)

# Cuvanje modela i TF-IDF vektorizatora
joblib.dump(
    {"tfidf": tfidf, "model": model},
    "models/classical_ml/naive_bayes.joblib"
)

predictions = model.predict(x_val_tfidf)


# Evaluacija na validation skupu
print("Accuracy:", accuracy_score(y_val, predictions))
print("Macro F1:", f1_score(y_val, predictions, average="macro"))
print("\n", classification_report(y_val, predictions, zero_division=0))


# Matrica konfuzije
ConfusionMatrixDisplay.from_predictions(y_val, predictions)
plt.title("Matrica konfuzije - Naive Bayes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/naive_bayes/confusion_matrix_naive_bayes.png")
plt.close()
