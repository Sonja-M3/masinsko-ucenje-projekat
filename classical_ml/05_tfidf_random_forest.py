import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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


# Treniranje Random Forest modela
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(x_train_tfidf, y_train)

predictions = model.predict(x_val_tfidf)


# Evaluacija na validation skupu
print("Accuracy:", accuracy_score(y_val, predictions))
print("Macro F1:", f1_score(y_val, predictions, average="macro"))
print("\n", classification_report(y_val, predictions, zero_division=0))


# Matrica konfuzije
ConfusionMatrixDisplay.from_predictions(y_val, predictions)
plt.title("Matrica konfuzije - Random Forest")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/confusion_matrix_random_forest.png")
plt.close()
