import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from src.extract_data import parse_report

MODEL_PATH = Path("models/random_forest.pkl")
LABEL_ENCODER_PATH = Path("models/label_encoder.pkl")
FEATURE_NAMES_PATH = Path("models/feature_names.pkl")

STATUS_MAP = {
    "healthy": "Normale",
    "not": "HTA probable",
}

MALE_VALUES = {"male", "m", "homme", "man"}
FEMALE_VALUES = {"female", "f", "femme", "woman"}


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, label_encoder, feature_names


def normalize_sex(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, (int, np.integer)):
        return 1 if int(value) == 1 else 0

    text = str(value).strip().lower()
    if text in MALE_VALUES:
        return 1
    if text in FEMALE_VALUES:
        return 0
    return None


def sex_to_label(value: Any) -> str:
    sex_numeric = normalize_sex(value)
    if sex_numeric == 1:
        return "Homme"
    if sex_numeric == 0:
        return "Femme"
    return "N/A"


def prepare_features(summary: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    # Keep the same preprocessing logic as in the modeling notebook: feature subset + fillna(0).
    df_row = pd.DataFrame([summary])

    for feature in feature_names:
        if feature not in df_row.columns:
            df_row[feature] = np.nan

    df_features = df_row[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df_features


def predict_report(model, label_encoder, feature_names, summary: dict[str, Any]) -> dict[str, Any]:
    df_features = prepare_features(summary, feature_names)
    proba = model.predict_proba(df_features)[0]
    predicted_encoded = int(model.predict(df_features)[0])
    predicted_label = str(label_encoder.inverse_transform([predicted_encoded])[0])
    statut = STATUS_MAP.get(predicted_label.lower(), predicted_label)

    # Build probabilities by true class labels from the fitted encoder/model classes.
    probabilities = {}
    for encoded_class, prob in zip(model.classes_, proba):
        class_label = label_encoder.inverse_transform([int(encoded_class)])[0]
        probabilities[str(class_label)] = float(prob)

    return {
        "statut": statut,
        "health_label": predicted_label,
        "confidence": float(proba.max()),
        "probabilities": probabilities,
        "summary": summary,
    }


def main():
    st.set_page_config(
        page_title="MAPA·AI - Analyse Tensionnelle",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header avec style moderne
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    .status-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .confidence-card {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <div class="main-title">🩺 MAPA·AI</div>
        <div class="subtitle">Analyse intelligente de rapports de mesure ambulatoire de la pression artérielle</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.title("📋 Instructions")
    st.sidebar.markdown("""
    **Comment utiliser l'application :**

    1. **Téléversez** un rapport PDF MAPA
    2. **Analysez** automatiquement les métriques
    3. **Obtenez** une prédiction du statut tensionnel
    4. **Visualisez** les probabilités et données patient
    """)

    uploaded_file = st.file_uploader(
        "📎 Sélectionnez votre rapport PDF MAPA",
        type=["pdf"],
        help="Formats acceptés : PDF de rapports de mesure ambulatoire de pression artérielle"
    )

    if uploaded_file is not None:
        with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = Path(tmp_file.name)

        try:
            report = parse_report(tmp_path)
            summary = report["summary"]
        except Exception as exc:
            st.error(f"❌ Erreur lors de l'extraction du PDF : {exc}")
            return
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass

        model, label_encoder, feature_names = load_model_artifacts()

        with st.spinner("🔄 Analyse en cours... Veuillez patienter"):
            result = predict_report(model, label_encoder, feature_names, summary)

        # Résultats principaux avec style moderne
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="status-card">Statut Tensionnel<br><span style="font-size: 2rem;">{}</span></div>'.format(result["statut"]), unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="confidence-card">Confiance<br><span style="font-size: 2rem;">{:.1f}%</span></div>'.format(result['confidence'] * 100), unsafe_allow_html=True)

        # Graphique des probabilités
        st.subheader("📊 Probabilités des classes")
        ordered_classes = ["healthy", "not"]
        ordered_proba = {k: result["probabilities"].get(k, 0.0) for k in ordered_classes}
        prob_df = pd.DataFrame(
            {
                "Classe": [STATUS_MAP.get(k, k) for k in ordered_proba.keys()],
                "Probabilité (%)": [v * 100 for v in ordered_proba.values()],
            }
        )
        st.bar_chart(data=prob_df.set_index("Classe"), y="Probabilité (%)", use_container_width=True)

        # Tableau des données du patient
        st.subheader("📋 Données du patient")
        patient_data = {
            "Métrique": [
                "Âge",
                "Sexe",
                "Tension systolique moyenne (24h)",
                "Tension diastolique moyenne (24h)",
                "Tension systolique moyenne (jour)",
                "Tension diastolique moyenne (jour)",
                "Tension systolique moyenne (nuit)",
                "Tension diastolique moyenne (nuit)",
                "Charge systolique jour (%)",
                "Charge systolique nuit (%)",
                "Charge diastolique jour (%)",
                "Charge diastolique nuit (%)",
                "Rythme circadien systolique (%)",
                "Rythme circadien diastolique (%)",
                "AASI systolique",
                "Tension systolique maximale",
                "Tension systolique minimale",
            ],
            "Valeur": [
                summary.get("age", "N/A"),
                sex_to_label(summary.get("sex")),
                summary.get("all_avg_sys", "N/A"),
                summary.get("all_avg_dia", "N/A"),
                summary.get("day_avg_sys", "N/A"),
                summary.get("day_avg_dia", "N/A"),
                summary.get("night_avg_sys", "N/A"),
                summary.get("night_avg_dia", "N/A"),
                summary.get("day_sys_load_pct", "N/A"),
                summary.get("night_sys_load_pct", "N/A"),
                summary.get("day_dia_load_pct", "N/A"),
                summary.get("night_dia_load_pct", "N/A"),
                summary.get("circadian_sys_night_des_pct", "N/A"),
                summary.get("circadian_dia_night_des_pct", "N/A"),
                summary.get("aasi_sys", "N/A"),
                summary.get("max_sys", "N/A"),
                summary.get("min_sys", "N/A"),
            ],
        }
        patient_df = pd.DataFrame(patient_data)
        st.dataframe(patient_df, use_container_width=True, hide_index=True)

    else:
        # Message d'accueil quand aucun fichier n'est uploadé
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: #667eea; margin-bottom: 1rem;">Bienvenue sur MAPA·AI</h2>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
                Téléversez un rapport PDF de mesure ambulatoire de la pression artérielle<br>
                pour obtenir une analyse automatique et une prédiction du statut tensionnel.
            </p>
            <div style="font-size: 4rem;">📎</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
