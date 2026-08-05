import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


# Učitavanje očišćenih podataka
train_data = pd.read_csv(
    "data/processed/train.txt",
    sep=";",
    header=None,
    names=["text", "emotion"]
)

val_data = pd.read_csv(
    "data/processed/val.txt",
    sep=";",
    header=None,
    names=["text", "emotion"]
)


# Odvajanje tekstova i tačnih emocija
x_train_text = train_data["text"]
y_train = train_data["emotion"]

x_val_text = val_data["text"]
y_val = val_data["emotion"]


# Pretvaranje tekstova u TF-IDF brojeve
vectorizer = TfidfVectorizer()

x_train = vectorizer.fit_transform(x_train_text)
x_val = vectorizer.transform(x_val_text)


print("Dimenzija trening podataka:", x_train.shape)
print("Dimenzija validacionih podataka:", x_val.shape)


# Pravljenje i obučavanje modela
model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)


# Predikcije na validacionom skupu
predictions = model.predict(x_val)


# Evaluacija modela
accuracy = accuracy_score(y_val, predictions)
macro_f1 = f1_score(y_val, predictions, average="macro")

print("\nAccuracy:", accuracy)
print("Macro F1:", macro_f1)

print("\nKlasifikacioni izveštaj:")
print(
    classification_report(
        y_val,
        predictions
    )
)
    
# Matrica konfuzije
ConfusionMatrixDisplay.from_predictions(
    y_val,
    predictions
)

plt.title("Matrica konfuzije - logistička regresija")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("confusion_matrix_logistic_regression.png")
plt.show()
