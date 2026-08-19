import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay


# Ucitavanje sva tri skupa
train = pd.read_csv("data/processed/train.txt", sep=";", header=None, names=["text", "emotion"])
val = pd.read_csv("data/processed/val.txt", sep=";", header=None, names=["text", "emotion"])
test = pd.read_csv("data/processed/test.txt", sep=";", header=None, names=["text", "emotion"])


# Model i hiperparametri su vec izabrani pomocu validation skupa,
# pa sada mozemo spojiti train i validation za finalno treniranje
train_val = pd.concat([train, val], ignore_index=True)

x_train_val = train_val["text"]
y_train_val = train_val["emotion"]

x_test = test["text"]
y_test = test["emotion"]


# Konacni model sa najboljim parametrima
# dobijenim pomocu GridSearchCV
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 1))),
    ("model", LinearSVC(C=0.5, class_weight="balanced", max_iter=3000))
])


# Finalno treniranje na train + validation skupu
model.fit(x_train_val, y_train_val)


# Test skup se koristi samo jednom za konacnu evaluaciju
predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")

print("KONACNI REZULTAT NA TEST SKUPU")
print("Accuracy:", accuracy)
print("Macro F1:", macro_f1)

print("\nKlasifikacioni izvestaj:")
print(classification_report(y_test, predictions, zero_division=0))


# Matrica konfuzije finalnog modela
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Matrica konfuzije - konacni SVM")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix_final_svm.png")
plt.close()


# Cuvanje pogresno klasifikovanih primera
# za kasniju analizu gresaka
errors = test.copy()
errors["prediction"] = predictions
errors = errors[errors["emotion"] != errors["prediction"]]

errors.to_csv("misclassified_examples.csv", index=False)

print("\nBroj pogresno klasifikovanih primera:", len(errors))
