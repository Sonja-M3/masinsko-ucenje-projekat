import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# Učitavanje trening, validacionog i test skupa
train_data = pd.read_csv(
    "data/raw/train.txt",
    sep=";",
    header=None,
    names=["text", "emotion"]
)

val_data = pd.read_csv(
    "data/raw/val.txt",
    sep=";",
    header=None,
    names=["text", "emotion"]
)

test_data = pd.read_csv(
    "data/raw/test.txt",
    sep=";",
    header=None,
    names=["text", "emotion"]
)


# Pregled osnovnih informacija o podacima
print(train_data.head())
print(train_data.shape)


# Provera nedostajućih vrednosti
print(train_data.isnull().sum())
print(val_data.isnull().sum())
print(test_data.isnull().sum())


# Provera dupliranih redova
print(train_data.duplicated().sum())
print(val_data.duplicated().sum())
print(test_data.duplicated().sum())


# Provera broja primera po emocijama
print(train_data["emotion"].value_counts())


# Uklanjanje duplikata unutar svakog skupa
train_clean = train_data.drop_duplicates()
val_clean = val_data.drop_duplicates()
test_clean = test_data.drop_duplicates()


# Uklanjanje tekstova koji se pojavljuju i u validacionom i test skupu
test_texts = set(test_clean["text"])

val_clean = val_clean[~val_clean["text"].isin(test_texts)]


# Uklanjanje tekstova iz trening skupa koji postoje u validacionom ili test skupu
val_texts = set(val_clean["text"])
test_texts = set(test_clean["text"])

train_clean = train_clean[~train_clean["text"].isin(val_texts)]

train_clean = train_clean[~train_clean["text"].isin(test_texts)]


# Ponovno numerisanje indeksa nakon uklanjanja redova
train_clean = train_clean.reset_index(drop=True)
val_clean = val_clean.reset_index(drop=True)
test_clean = test_clean.reset_index(drop=True)


# Čuvanje očišćenih podataka u posebnom folderu
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

train_clean.to_csv(
    processed_dir / "train.txt",
    sep=";",
    header=False,
    index=False
)

val_clean.to_csv(
    processed_dir / "val.txt",
    sep=";",
    header=False,
    index=False
)

test_clean.to_csv(
    processed_dir / "test.txt",
    sep=";",
    header=False,
    index=False
)


# Broj primera za svaku emociju
emotion_counts = train_clean["emotion"].value_counts()

print("\nBroj primera po emocijama nakon čišćenja:")
print(emotion_counts)


# Crtanje grafikona
emotion_counts.plot(kind="bar")

plt.title("Raspodela emocija u trening skupu")
plt.xlabel("Emocija")
plt.ylabel("Broj primera")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("raspodela_emocija.png")

plt.show()
