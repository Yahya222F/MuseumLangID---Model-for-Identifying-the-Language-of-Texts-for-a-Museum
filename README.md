![image](https://github.com/user-attachments/assets/1b701899-f179-4f08-a4cf-d50720bc827b)
> **Progetto del Master in AI Development — Modulo “A.I. applicata per Sviluppatori”**  
> **Studente:** Giacomo Latini
# MuseumLangID — Identificazione automatica della lingua di testi museali

Questo repository contiene un progetto di **Language Identification** per classificare le descrizioni di opere e manufatti museali nella loro lingua di riferimento. Il modello utilizza **TF‑IDF** e **Multinomial Naive Bayes** per distinguere tra **italiano (`it`)**, **inglese (`en`)** e **tedesco (`de`)**.

---

## Dataset
- **Nome file:** `[museo_descrizioni.csv](https://raw.githubusercontent.com/Profession-AI/progetti-ml/refs/heads/main/Modello%20per%20l'identificazione%20della%20lingua%20dei%20testi%20di%20un%20museo/museo_descrizioni.csv`  
- **Colonne principali:**
  - `Testo` → descrizione testuale (feature)
  - `Codice Lingua` → etichetta (`it`, `en`, `de`)

> **Nota:** il dataset è trattato come **multi‑classe (3 classi)**

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
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Esecuzione del notebook
Apri `nome_notebook.ipynb` con Jupyter/VS Code e riesegui tutte le celle.  

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

## 🙏 Crediti
- Docenti e materiali del **Master in AI Development** (Modulo A.I. applicata per Sviluppatori).  
- Dataset `museo_descrizioni.csv`.  
- Librerie open‑source citate sopra.

## 📄 Licenza
Questo progetto è rilasciato con licenza GNU GPL v3.
Vedi il file LICENSE per i dettagli. 
