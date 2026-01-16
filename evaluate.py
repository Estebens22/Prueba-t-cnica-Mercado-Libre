import pandas as pd
import joblib
from new_or_used import build_dataset
from features.feature_builder import build_features
from sklearn.metrics import accuracy_score, recall_score, classification_report

# -----------------------------
# Cargar modelo entrenado
# -----------------------------
model = joblib.load("models/new_used_model.pkl")

# -----------------------------
# Cargar dataset (hold-out real)
# -----------------------------
X_train, y_train, X_test, y_test = build_dataset()

# Convertir X_test (list[dict]) a DataFrame
df_test = pd.DataFrame(X_test)

# Feature engineering
df_test = build_features(df_test)

# Ground truth
y_true = pd.Series(y_test).map({"new": 0, "used": 1})

# Predicción
y_proba = model.predict_proba(df_test)[:, 1]

# Ajuste de threshold (más estricto)
threshold = 0.58
y_pred = (y_proba >= threshold).astype(int)

# -----------------------------
# Métricas
# -----------------------------
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Recall (USED):", recall_score(y_true, y_pred))
print("\nClassification report:\n")
print(classification_report(y_true, y_pred))