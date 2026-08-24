import pandas as pd
import matplotlib.pyplot as plt

import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay


# Ucitavanje train i validation skupa
train = pd.read_csv("data/processed/train.txt", sep=";", header=None, names=["text", "emotion"])
val = pd.read_csv("data/processed/val.txt", sep=";", header=None, names=["text", "emotion"])

x_train = train["text"]
y_train = train["emotion"]
x_val = val["text"]
y_val = val["emotion"]



# Podesavanje linearnog SVM modela


# Pipeline je vazan da se TF-IDF ponovo uci
# samo na trening delu svakog CV folda
svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LinearSVC(max_iter=3000))
])


# Parametri koje GridSearchCV ispituje
svm_parameters = {
    # (1, 1) - samo pojedinacne reci
    # (1, 2) - pojedinacne reci i parovi susednih reci
    "tfidf__ngram_range": [(1, 1), (1, 2)],

    # C kontrolise jacinu regularizacije SVM modela
    "model__C": [0.5, 1.0, 2.0],

    # balanced daje vecu tezinu redjim klasama
    "model__class_weight": [None, "balanced"]
}


# GridSearch koristi 3-fold unakrsnu validaciju
# samo unutar trening skupa
svm_grid = GridSearchCV(
    svm_pipeline,
    svm_parameters,
    scoring="f1_macro",
    cv=3,
    n_jobs=1,
    verbose=1
)

print("Podesavanje SVM modela")
svm_grid.fit(x_train, y_train)

print("\nNajbolji parametri:")
print(svm_grid.best_params_)
print("CV Macro F1:", svm_grid.best_score_)


# Najbolja kombinacija parametara se zatim
# proverava na posebnom validation skupu
best_svm = svm_grid.best_estimator_

joblib.dump(
    best_svm,
    "models/classical_ml/svm_tuned.joblib"
)

svm_predictions = best_svm.predict(x_val)

svm_accuracy = accuracy_score(y_val, svm_predictions)
svm_macro_f1 = f1_score(y_val, svm_predictions, average="macro")

print("\nSVM na validation skupu")
print("Accuracy:", svm_accuracy)
print("Macro F1:", svm_macro_f1)
print(classification_report(y_val, svm_predictions, zero_division=0))


# Matrica konfuzije podesenog SVM modela
ConfusionMatrixDisplay.from_predictions(y_val, svm_predictions)
plt.title("Matrica konfuzije - podeseni SVM")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/SVM/confusion_matrix_svm_tuned.png")
plt.close()



# Podesavanje logisticke regresije


logistic_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression(max_iter=1000))
])


# Ispitujemo isti TF-IDF izbor, kao i
# C i class_weight logisticke regresije
logistic_parameters = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "model__C": [0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced"]
}

logistic_grid = GridSearchCV(
    logistic_pipeline,
    logistic_parameters,
    scoring="f1_macro",
    cv=3,
    n_jobs=1,
    verbose=1
)

print("\nPodesavanje logisticke regresije")
logistic_grid.fit(x_train, y_train)

print("\nNajbolji parametri:")
print(logistic_grid.best_params_)
print("CV Macro F1:", logistic_grid.best_score_)


# Provera najbolje logisticke regresije na validation skupu
best_logistic = logistic_grid.best_estimator_

joblib.dump(
    best_logistic,
    "models/classical_ml/logistic_regression_tuned.joblib"
)

logistic_predictions = best_logistic.predict(x_val)

logistic_accuracy = accuracy_score(y_val, logistic_predictions)
logistic_macro_f1 = f1_score(y_val, logistic_predictions, average="macro")

print("\nLogisticka regresija na validation skupu")
print("Accuracy:", logistic_accuracy)
print("Macro F1:", logistic_macro_f1)
print(classification_report(y_val, logistic_predictions, zero_division=0))


# Matrica konfuzije podesene logisticke regresije
ConfusionMatrixDisplay.from_predictions(y_val, logistic_predictions)
plt.title("Matrica konfuzije - podesena logisticka regresija")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/logistic_regression/confusion_matrix_logistic_regression_tuned.png")
plt.close()


# Cuvanje rezultata podesenih modela radi poredjenja
results = pd.DataFrame({
    "model": ["Linear SVM tuned", "Logistic Regression tuned"],
    "accuracy": [svm_accuracy, logistic_accuracy],
    "macro_f1": [svm_macro_f1, logistic_macro_f1]
})

results.to_csv("results/tuned_results.csv", index=False)

print("\nRezultati podesenih modela:")
print(results)
