# MAPA-AI

Projet de data science applique a l'analyse de rapports MAPA , de l'extraction PDF jusqu'a la prediction assistee par modele de machine learning.

MAPA (Mesure Ambulatoire de la Pression Artérielle) — un appareil que le patient porte 24h pour mesurer sa tension automatiquement toutes les 15-30 minutes, le jour et la nuit.


## 1. Contexte et objectif pedagogique


- transformer des rapports PDF medicaux en donnees exploitables,
- construire une variable cible a partir de texte clinique,
- entrainer un modele de classification,
- exposer la prediction via API,
- proposer des interfaces de demonstration (web HTML et Streamlit).



## 2. Demarche suivie

### 2.1 Exploration initiale

- Exploration dans les notebooks:
  - `notebooks/01_EDA.ipynb`
  - `notebooks/02_Modeling.ipynb`
- Analyse des variables tensionnelles (moyennes, charges, variabilite, rythmes circadiens, AASI).

### 2.2 Extraction automatique depuis PDF

Le script `src/extract_data.py` parse les rapports MAPA (4 pages) et produit:

- un CSV de synthese patient/examen,
- un CSV des mesures detaillees,
- un JSON complet des rapports parses.

Informations extraites (exemples):

- age, sexe, date de naissance,
- moyennes PAS/PAD globales, diurnes et nocturnes,
- charges tensionnelles,
- maxima/minima avec horodatage,
- indicateurs de variabilite et AASI,
- champs textuels cliniques (`contents`, `diagnostics`).

### 2.3 Construction de la cible via LLM local

Le script `src/classify_health_ollama.py` enrichit le fichier de synthese en ajoutant un label binaire:

- `healthy`
- `not`

Le classement repose sur les notes textuelles cliniques et un appel a Ollama (modele local), afin de produire une cible supervisée pour l'apprentissage.

### 2.4 Entrainement du modele

Le script `src/train_model.py`:

- charge `data/bp_summary_classified.csv`,
- selectionne des variables numeriques pertinentes,
- encode la cible,
- entraine un RandomForestClassifier,
- sauvegarde les artefacts dans `models/`:
  - `random_forest.pkl`


### 2.5 Inference et mise a disposition

- API Flask (`src/api.py`):
  - `POST /predict` recoit un PDF, extrait les metriques, calcule la prediction et retourne un JSON de resultat.
  - `GET /` endpoint de sante.
- Application Streamlit (`streamlit_app.py`): interface utilisateur pour charger un PDF et visualiser prediction + probabilites.
- Interface HTML (`interface.html`): front-end leger en glisser-deposer connecte a l'API Flask.

### 2.6 Traitement de confidentialite

Le script `src/remove_names.py` permet d'anonymiser des PDF (suppression d'identifiants et du nom patient) avant exploitation des donnees.

## 3. Organisation du depot

- `data/`: donnees tabulaires intermediaires et finales.
- `models/`: modeles et encodeurs sauvegardes.
- `notebooks/`: exploration et experimentation.
- `src/`: scripts du pipeline de traitement et d'inference.
- `streamlit_app.py`: application de demonstration Streamlit.
- `interface.html`: interface web statique reliee a l'API.
- `requirements.txt` et `environment.yml`: gestion des dependances.

## 4. Reproduction du projet

### 4.0 Initialisation GitHub (optionnel)

```bash
git remote add origin https://github.com/mariembouhlel/Classification-Automatique-du-Statut-Tensionnel-MAPA-.git
git push -u origin main
```

### 4.1 Installation (pip)

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 4.2 Installation (conda)

```bash
conda env create -f environment.yml
conda activate mapaai
```

### 4.3 Execution du pipeline

1. Extraction de PDF vers jeux de donnees structures:

```bash
python -m src.extract_data "dossier_des_pdfs" --summary-csv data/bp_summary.csv --readings-csv data/bp_readings.csv --json-out data/bp_extracted.json
```

2. Enrichissement de la cible par LLM local (Ollama):

```bash
python -m src.classify_health_ollama --input-csv data/bp_summary.csv --output-csv data/bp_summary_classified.csv --text-field diagnostics
```

3. Entrainement du modele:

```bash
python -m src.train_model
```

4. Lancement de l'API:

```bash
python -m src.api
```

5. Lancement de l'interface Streamlit (optionnel):

```bash
streamlit run streamlit_app.py
```

6. Interface HTML (optionnel): ouvrir `interface.html` dans un navigateur, avec l'API active sur `http://localhost:5000`.

## 5. Resultat attendu

Pour un PDF MAPA valide, le systeme retourne:

- un statut tensionnel predit,
- un niveau de confiance/probabilites par classe,
- un resume des principales metriques hemodynamiques extraites.

## 6. Limites et perspectives

Limites actuelles:

- robustesse dependante du format exact des PDF,
- absence de suite de tests automatisee,
- travail centre sur une classification simple et interpretable.

Ameliorations envisagees:

- ajouter des tests unitaires et d'integration,
- elargir la compatibilite a d'autres templates de rapports,
- comparer plusieurs modeles (ex: gradient boosting, XGBoost) avec protocole de validation plus strict,
- enrichir l'interface de suivi des predictions.

