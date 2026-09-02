# Prepoznavanje emocija iz teksta

Projekat iz predmeta Mašinsko učenje posvećen automatskom prepoznavanju emocija u tekstu.

## Cilj projekta

Cilj aplikacije je da na osnovu teksta na engleskom jeziku prepozna jednu od šest emocija:

- `joy` (radost)
- `sadness` (tuga)
- `anger` (bes)
- `fear` (strah)
- `love` (ljubav)
- `surprise` (iznenađenje)

Pored same klasifikacije, cilj projekta je poređenje različitih pristupa obradi prirodnog jezika: klasičnih metoda mašinskog učenja, rekurentnih neuronskih mreža i transformer modela. Modeli su poređeni pomoću metrika Accuracy i Macro F1.

## Korišćeni modeli

U projektu su implementirani i testirani sledeći modeli:

### Klasični modeli

Tekst se kod klasičnih modela predstavlja pomoću TF-IDF vektorizacije.

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Random Forest

Za najperspektivnije modele izvršeno je podešavanje hiperparametara pomoću `GridSearchCV`. Kao finalni klasični model izabran je Linear SVM.

### Rekurentne neuronske mreže

- LSTM
- GRU

### Transformer modeli

- BERT (`bert-base-uncased`)
- DistilBERT (`distilbert-base-uncased`)

Pretrenirani transformer modeli dodatno su trenirani za klasifikaciju tekstova u šest klasa emocija.

## Skup podataka

Korišćen je skup [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) sa platforme Kaggle. Skup sadrži tekstove na engleskom jeziku označene jednom od šest emocija i originalno je podeljen na trening, validacioni i test skup.

Svaki red ima format:

```text
tekst;emocija
```

Pre modelovanja proverene su nedostajuće vrednosti, uklonjeni su duplikati i sprečeno je preklapanje tekstova između skupova. Obrađeni podaci koji se nalaze u direktorijumu `data/processed` sadrže:

- 15.983 primera za trening
- 1.997 primera za validaciju
- 2.000 primera za testiranje

## Članovi tima

- Matija Stanković
- Sonja Mijailović
- Dimitrije Vranić

## Pokretanje projekta

Za najjednostavnije pokretanje koristite završnu Jupyter svesku `final_demo.ipynb`. Ona učitava već obrađene podatke i sačuvane modele, prikazuje rezultate i demonstrira predikcije.

### 1. Kloniranje repozitorijuma

Veliki modeli čuvaju se pomoću Git LFS-a, pa je potrebno da [Git LFS](https://git-lfs.com/) bude instaliran.

```bash
git clone https://github.com/Sonja-M3/masinsko-ucenje-projekat.git
cd masinsko-ucenje-projekat
git lfs pull
```

### 2. Kreiranje virtuelnog okruženja

Potrebna je instalacija Python-a. Zatim kreirajte virtuelno okruženje:

```bash
python -m venv .venv
```

Aktiviranje na Windows PowerShell-u:

```powershell
.\.venv\Scripts\Activate.ps1
```

Aktiviranje na Linux-u ili macOS-u:

```bash
source .venv/bin/activate
```

### 3. Instaliranje biblioteka

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Pokretanje demonstracije

Komandu pokrenite iz osnovnog direktorijuma projekta:

```bash
jupyter notebook final_demo.ipynb
```

Kada se sveska otvori u pregledaču, izaberite **Kernel > Restart & Run All** kako biste izvršili sve ćelije redom.

Pri prvom pokretanju BERT-a i DistilBERT-a potrebna je internet veza, jer biblioteka Transformers preuzima njihove osnovne konfiguracije, tokenizatore i pretrenirane osnovne modele sa Hugging Face-a. Težine modela dodatno treniranih u okviru projekta već se nalaze u direktorijumu `models`.

## Struktura projekta

```text
masinsko-ucenje-projekat/
├── classical_ml/       # Klasični TF-IDF modeli i podešavanje parametara
├── data/processed/     # Obrađeni trening, validacioni i test podaci
├── models/             # Sačuvani modeli
├── results/            # Rezultati, izveštaji i matrice konfuzije
├── RNN/                # LSTM i GRU modeli
├── tranformer_models/  # BERT i DistilBERT modeli
├── final_demo.ipynb    # Završna demonstracija projekta
└── requirements.txt    # Potrebne Python biblioteke
```

Detalji treniranja i evaluacije pojedinačnih modela nalaze se u odgovarajućim Python skriptama i Jupyter sveskama.
