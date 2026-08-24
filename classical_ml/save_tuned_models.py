import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# Ucitavanje trening skupa
train = pd.read_csv("data/processed/train.txt", sep=";", header=None, names=["text", "emotion"])

x_train = train["text"]
y_train = train["emotion"]


# Podeseni SVM
svm_tuned = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 1))),
    ("model", LinearSVC(C=0.5, class_weight="balanced", max_iter=3000))
])

svm_tuned.fit(x_train, y_train)
joblib.dump(svm_tuned, "models/classical_ml/svm_tuned.joblib")


# Podesena logisticka regresija
logistic_tuned = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 1))),
    ("model", LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000))
])

logistic_tuned.fit(x_train, y_train)
joblib.dump(logistic_tuned, "models/classical_ml/logistic_regression_tuned.joblib")

print("Podeseni modeli su sacuvani.")
