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

## Redosled Jupyter svezaka

Sveske su numerisane prema preporučenom redosledu pokretanja:

1. [`01_klasicni_modeli.ipynb`](classical_ml/01_klasicni_modeli.ipynb) – klasični modeli mašinskog učenja
2. [`02_lstm.ipynb`](RNN/02_lstm.ipynb) – LSTM model
3. [`03_gru.ipynb`](RNN/03_gru.ipynb) – GRU model
4. [`04_bert.ipynb`](tranformer_models/04_bert.ipynb) – BERT model
5. [`05_distilbert.ipynb`](tranformer_models/05_distilbert.ipynb) – DistilBERT model
6. [`06_final_demo.ipynb`](06_final_demo.ipynb) – završna demonstracija projekta

## Pokretanje projekta

Za najjednostavnije pokretanje koristite završnu Jupyter svesku `06_final_demo.ipynb`. Ona učitava već obrađene podatke i sačuvane modele, prikazuje rezultate i demonstrira predikcije.

### 1. Git LFS i kloniranje repozitorijuma

Zbog veličine pojedinih istreniranih modela projekat koristi [Git LFS (Git Large File Storage)](https://git-lfs.com/). Nakon instalacije Git LFS-a pokrenite:

```bash
git lfs install
git clone https://github.com/Sonja-M3/masinsko-ucenje-projekat.git
cd masinsko-ucenje-projekat
git lfs pull
```

Nakon toga svi dostupni sačuvani modeli nalaze se u direktorijumu `models/`.

### 2. Podešavanje Python okruženja

Preporučuje se korišćenje novije verzije Python-a. Klasični deo projekta i završna demo sveska testirani su sa Python-om 3.13.

Provera instalirane verzije Python-a:

```bash
python --version
```

Kreiranje virtuelnog okruženja:

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
jupyter notebook 06_final_demo.ipynb
```

Kada se sveska otvori u pregledaču, izaberite **Kernel > Restart & Run All** kako biste izvršili sve ćelije redom.

Pri prvom pokretanju BERT-a i DistilBERT-a potrebna je internet veza, jer biblioteka Transformers preuzima njihove osnovne konfiguracije, tokenizatore i pretrenirane osnovne modele sa Hugging Face-a. Težine modela dodatno treniranih u okviru projekta već se nalaze u direktorijumu `models`.

## Struktura projekta

```text
masinsko-ucenje-projekat/
├── classical_ml/
│   └── 01_klasicni_modeli.ipynb  # Klasični TF-IDF modeli
├── RNN/
│   ├── 02_lstm.ipynb              # LSTM model
│   └── 03_gru.ipynb               # GRU model
├── tranformer_models/
│   ├── 04_bert.ipynb              # BERT model
│   └── 05_distilbert.ipynb        # DistilBERT model
├── data/processed/                # Obrađeni podaci
├── models/                        # Sačuvani modeli
├── results/                       # Rezultati, izveštaji i matrice konfuzije
├── 06_final_demo.ipynb            # Završna demonstracija projekta
└── requirements.txt               # Potrebne Python biblioteke
```

Detalji treniranja i evaluacije pojedinačnih modela nalaze se u odgovarajućim Python skriptama i Jupyter sveskama.

## Literatura

- [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp), Kaggle
- [Scikit-learn dokumentacija](https://scikit-learn.org/stable/)
- [PyTorch dokumentacija](https://docs.pytorch.org/docs/stable/index.html)
- [Transformers dokumentacija](https://huggingface.co/docs/transformers/)
- [Materijali i literatura sa kursa Mašinsko učenje](https://github.com/matf-ml/materijali-sa-vezbi-2025)
