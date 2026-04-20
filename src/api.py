from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path

try:
    from src.extract_data import parse_report
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from extract_data import parse_report

app = Flask(__name__)
CORS(app)

# Charger le modèle et les encodeurs
model         = joblib.load("models/random_forest.pkl")
le            = joblib.load("models/label_encoder.pkl")
feature_names = joblib.load("models/feature_names.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier reçu"}), 400

    pdf_file = request.files["file"]
    temp_path = Path("temp_upload.pdf")
    pdf_file.save(temp_path)

    try:
        report = parse_report(temp_path)
        row    = report["summary"]
    except Exception as e:
        return jsonify({"error": f"Erreur extraction PDF : {str(e)}"}), 500
    finally:
        if temp_path.exists():
            temp_path.unlink()

    # Préparer les features
    df_row = pd.DataFrame([row])

    # Encoder sex
    df_row['sex'] = 1 if str(row.get('sex','')).lower() == 'male' else 0

    # Remplir NaN
    df_row['age']    = df_row['age'].fillna(50)
    df_row['aasi_sys'] = df_row['aasi_sys'].fillna(0.35)
    df_row['circadian_sys_night_des_pct'] = df_row['circadian_sys_night_des_pct'].fillna(0)
    df_row['circadian_dia_night_des_pct'] = df_row['circadian_dia_night_des_pct'].fillna(0)

    # Garder seulement les features du modèle
    df_features = df_row[feature_names]

    # Prédiction
    proba  = model.predict_proba(df_features)[0]
    classe_encoded = int(np.argmax(proba))
    classe = le.inverse_transform([classe_encoded])[0]

    # Mapping classes → label lisible
    label_map = {0: "HTA vraie", 1: "Normale", 2: "Limite"}
    statut = label_map.get(classe_encoded, classe)

    return jsonify({
        "statut":     statut,
        "confidence": f"{proba.max()*100:.0f}%",
        "proba_hta":  float(proba[0]),
        "proba_norm": float(proba[1]),
        "proba_limit":float(proba[2]),
        "all_avg_sys": row.get("all_avg_sys"),
        "all_avg_dia": row.get("all_avg_dia"),
        "day_avg_sys": row.get("day_avg_sys"),
        "day_avg_dia": row.get("day_avg_dia"),
        "night_avg_sys": row.get("night_avg_sys"),
        "night_avg_dia": row.get("night_avg_dia"),
        "day_sys_load_pct":   row.get("day_sys_load_pct"),
        "night_sys_load_pct": row.get("night_sys_load_pct"),
        "day_dia_load_pct":   row.get("day_dia_load_pct"),
        "night_dia_load_pct": row.get("night_dia_load_pct"),
        "circadian_sys_night_des_pct": row.get("circadian_sys_night_des_pct"),
        "aasi_sys": row.get("aasi_sys"),
        "max_sys":  row.get("max_sys"),
        "min_sys":  row.get("min_sys"),
    })

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "MAPA·AI API en ligne ✅"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)