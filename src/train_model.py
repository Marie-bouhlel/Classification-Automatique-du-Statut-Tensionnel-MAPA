import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Charger les données
df = pd.read_csv('data/bp_summary_classified.csv')

# Sélectionner les features numériques
features = ['age', 'all_avg_sys', 'all_avg_dia', 'day_avg_sys', 'day_avg_dia', 'night_avg_sys', 'night_avg_dia', 
            'day_sys_load_pct', 'night_sys_load_pct', 'day_dia_load_pct', 'night_dia_load_pct', 
            'circadian_sys_night_des_pct', 'circadian_dia_night_des_pct', 'aasi_sys']

# Encoder la cible
le = LabelEncoder()
df['health_encoded'] = le.fit_transform(df['health_label'])

# Préparer X et y
X = df[features].fillna(0)  # Remplir NaN avec 0
y = df['health_encoded']

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sauvegarder
joblib.dump(model, 'models/random_forest.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(features, 'models/feature_names.pkl')

print("Modèle entraîné et sauvegardé.")