# Classification Automatique du Statut Tensionnel MAPA

Pipeline IA de bout en bout pour analyser des rapports MAPA (PDF), extraire les indicateurs hemodynamiques, classifier le statut tensionnel, puis exposer le resultat via API et interface utilisateur.

![MAPA Cover Placeholder](docs/images/mapa-cover-placeholder.png)

## Vue d'ensemble

- Extraction automatique des donnees cliniques depuis PDF MAPA
- Enrichissement de cible a partir de texte clinique
- Entrainement d'un modele de classification
- Inference via API Flask
- Demo utilisateur avec Streamlit et interface HTML

## Architecture

![MAPA Architecture Placeholder](docs/images/mapa-architecture-placeholder.png)

1. `src/extract_data.py` transforme les rapports PDF en jeux de donnees structures.
2. `src/classify_health_ollama.py` ajoute les labels cibles via Ollama.
3. `src/train_model.py` entraine et sauvegarde le modele.
4. `src/api.py` sert la prediction en HTTP.
5. `streamlit_app.py` et `interface.html` fournissent des interfaces de demonstration.

## Structure du projet

```text
Classification-Automatique-du-Statut-Tensionnel-MAPA-/
|- notebooks/
|- src/
|- streamlit_app.py
|- interface.html
|- requirements.txt
|- environment.yml
```

## Demarrage rapide

### Option 1: pip

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate mapaai
```

## Execution

```bash
python -m src.extract_data "dossier_des_pdfs" --summary-csv data/bp_summary.csv --readings-csv data/bp_readings.csv --json-out data/bp_extracted.json
python -m src.classify_health_ollama --input-csv data/bp_summary.csv --output-csv data/bp_summary_classified.csv --text-field diagnostics
python -m src.train_model
python -m src.api
streamlit run streamlit_app.py
```

## Endpoints API

- `GET /` : health check
- `POST /predict` : recoit un PDF et retourne prediction + probabilites

## Resultats et captures

![MAPA UI Placeholder](docs/images/mapa-ui-placeholder.png)
![MAPA Charts Placeholder](docs/images/mapa-charts-placeholder.png)

## Depot GitHub

- URL: `https://github.com/Marie-bouhlel/Classification-Automatique-du-Statut-Tensionnel-MAPA`
- Branche principale: `main`

## Limites actuelles

- Performance dependante de la qualite et du format des PDF
- Couverture de tests automatises a renforcer
- Pipeline d'annotation texte a monitorer selon le modele Ollama utilise

Ameliorations envisagees:

- ajouter des tests unitaires et d'integration,
- elargir la compatibilite a d'autres templates de rapports,
- comparer plusieurs modeles (ex: gradient boosting, XGBoost) avec protocole de validation plus strict,
- enrichir l'interface de suivi des predictions.

