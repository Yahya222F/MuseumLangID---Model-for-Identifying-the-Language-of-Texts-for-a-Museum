# MuseumLangID — Identificazione automatica della lingua di testi museali

> **Progetto del Master in AI Development — Modulo “A.I. applicata per Sviluppatori”**  
> **Studente:** Giacomo Latini

Questo repository contiene un progetto di **Language Identification** per classificare le descrizioni di opere e manufatti museali nella loro lingua di riferimento. Il modello utilizza una pipeline **TF‑IDF + Multinomial Naive Bayes** per distinguere tra **italiano (`it`)**, **inglese (`en`)** e **tedesco (`de`)**.

---

## 🧭 Obiettivi
- Automatizzare l’**identificazione della lingua** di testi brevi (descrizioni museali).  
- Produrre un **baseline forte** e facilmente deployabile con Scikit‑learn.  
- Fornire una pipeline **riproducibile e spiegabile**, con metriche e grafici (matrici di confusione, ROC/AUC).

---

## 📦 Dataset
- **Nome file:** `museo_descrizioni.csv`  
- **Colonne principali:**
  - `Testo` → descrizione testuale (feature)
  - `Codice Lingua` → etichetta (`it`, `en`, `de`)
- **Origine:** il notebook carica il CSV da un percorso remoto (`BASE_PATH`) ospitato su GitHub (raw).  
  Se preferisci usare un file locale, imposta `BASE_PATH = ""` e assicurati che `museo_descrizioni.csv` sia nella stessa cartella del notebook.

> **Nota:** il dataset è trattato come **multi‑classe (3 classi)**. Il notebook mostra anche distribuzioni e controlli di qualità (schema colonne, valori nulli, etichette uniche).

---

## 🏗️ Architettura della soluzione
1. **Import & Config**
   - Librerie: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
   - Costanti: `RANDOM_SEED` (riproducibilità), `BASE_PATH` (sorgente dati).

2. **Caricamento & QA dati**
   - `pd.read_csv(BASE_PATH + "museo_descrizioni.csv")`
   - Controlli: `df.info()`, `df.isna().sum()`, `df["Codice Lingua"].unique()`

3. **Preprocessing testuale**
   - Funzione `data_cleaner(description)` con:
     - rimozione di **numeri** (`\d+`)
     - rimozione di **URL** (`http/https/www`)
     - rimozione **punteggiatura**
     - **lowercasing**
   - Applicata alla colonna `Testo` → `descriptions_cleaned`

4. **Train/Test split**
   - `train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)`  
   - Dove `X` è l’array dei testi puliti e `y` le etichette.

5. **Rappresentazione: TF‑IDF**
   - Funzione `tfidfvectorizer(data, tfidf_vectorizer=None)`:
     - in **train** esegue `fit_transform`
     - in **test** esegue `transform` con lo stesso vocabolario

6. **Modello**
   - **Multinomial Naive Bayes** (`MultinomialNB`)
   - Addestramento su matrice TF‑IDF di train

7. **Valutazione**
   - **Classification Report** (precision, recall, f1 per classe)
   - **Matrici di confusione** (train e test) con `seaborn.heatmap`
   - **Curve ROC** in schema **One‑vs‑Rest** + **AUC** per ciascuna classe

> Nel notebook è presente anche un’analisi qualitativa delle **matrici di confusione**. Per il set di test, la diagonale è dominante (ad es. valori come `it: 22`, `en: 19`, `de: 15`) con pochi errori fuori diagonale, indicando buona separabilità tra le tre lingue.

---

## ▶️ Come eseguire
### Prerequisiti
- **Python 3.10+**
- Pacchetti: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

### Setup rapido
```bash
# 1) Clona il repo
git clone <URL_DEL_REPO>
cd <NOME_REPO>

# 2) (opzionale) Crea un ambiente virtuale
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Installa le dipendenze
pip install -r requirements.txt
# oppure:
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Esecuzione del notebook
Apri `progetto_museum_langid.ipynb` (o il nome del notebook) con Jupyter/VS Code e riesegui tutte le celle.  
Se usi un CSV locale, imposta:
```python
BASE_PATH = ""  # e metti museo_descrizioni.csv accanto al notebook
```

---

## 📊 Output principali
- **Report di classificazione** (per classe e macro/micro average)
- **Matrice di confusione (train/test)** per visualizzare TP/FP/FN/TN
- **Curve ROC + AUC** per ciascuna classe (`it`, `en`, `de`)

Gli script/utility inclusi:
- `data_cleaner(description)` → normalizzazione testi
- `tfidfvectorizer(data, tfidf_vectorizer=None)` → TF‑IDF train/test
- `classifier_report(model, (X, y))` → stampa metriche
- `plot_confusion_matrix(model, (X, y))` → heatmap confusion matrix
- `plot_roc_curve(model, (X, y))` → ROC OVR + AUC

---

## ✅ Scelte progettuali
- **TF‑IDF + MultinomialNB**: baseline classica, veloce ed efficace per testi brevi e vocabolari distintivi tra lingue.
- **Preprocessing minimale**: manteniamo i tratti discriminanti delle lingue (accenti, stopword) evitando stem/lemmatizzazione.
- **Fit su train, transform su test**: corretta separazione per evitare leakage.
- **Seed fisso**: riproducibilità degli split.

---

## ⚠️ Note & miglioramenti suggeriti
- **Ordinamento parametri in `classification_report`**: nel notebook è chiamato come `classification_report(y_pred, y)`, ma la firma corretta è `classification_report(y_true, y_pred)`. Consigliato correggere per evitare metriche invertite.
- **Docstring `plot_confusion_matrix`**: menziona “modello K‑NN”, ma il modello usato è **MultinomialNB** → aggiornare la docstring.
- **Memoria TF‑IDF**: il codice fa `toarray()`. Su dataset grandi può saturare la RAM. Meglio usare la matrice **sparsa** direttamente (scikit‑learn lo supporta nativamente).
- **Caratteri e n‑grammi**: per Language ID spesso performano meglio **n‑grammi di caratteri** (`analyzer='char'`, `ngram_range=(3,5)`), da valutare come upgrade.
- **Stopword/diacritici**: in Language ID spesso è utile **non** rimuovere stopword e **preservare accenti**; già coerente con l’approccio corrente.
- **Valutazione**: aggiungere **cross‑validation**, **stratificazione** nello split (`stratify=y`) e report **macro‑averaged** completo.
- **Packaging**: possibile migrazione verso una **`Pipeline` scikit‑learn** (`Pipeline([('tfidf', TfidfVectorizer(...)), ('clf', MultinomialNB())])`) con salvataggio di modello e vettorizzatore (`joblib`).

---

## 🗂️ Struttura consigliata del repo
```
.
├─ notebooks/
│  └─ progetto_museum_langid.ipynb
├─ data/              # (opzionale) dataset locale non versionato
├─ src/               # (opzionale) funzioni riusabili
├─ reports/           # (opzionale) grafici ed export metriche
├─ requirements.txt
└─ README.md
```

Esempio `requirements.txt` minimale:
```
scikit-learn>=1.2
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
```

---

## 📌 Esempio d’uso (inferenza rapida)
```python
# addestramento (schematico)
X_train_vec, vec = tfidfvectorizer(X_train)
clf = MultinomialNB().fit(X_train_vec, y_train)

# inferenza su nuovo testo
X_new_vec = vec.transform(["Descrizione breve dell'opera in italiano."])
pred = clf.predict(X_new_vec)   # -> ['it']
```

---

## 📄 Licenza
Scegli e aggiungi una licenza (es. MIT).

## 🙏 Crediti
- Docenti e materiali del **Master in AI Development** (Modulo A.I. applicata per Sviluppatori).  
- Dataset `museo_descrizioni.csv` (sorgente GitHub raw, referenziato nel notebook).  
- Librerie open‑source citate sopra.
