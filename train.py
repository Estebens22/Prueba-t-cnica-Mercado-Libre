import pandas as pd
import joblib

from new_or_used import build_dataset
from features.feature_builder import build_features

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier


# --------------------------------------------------
# 1. Cargar dataset usando el helper oficial
# --------------------------------------------------
X_train_raw, y_train_raw, X_test_raw, y_test_raw = build_dataset()

# X_train_raw es una LISTA de diccionarios → DataFrame
df = pd.DataFrame(X_train_raw)

# --------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------
df = build_features(df)

# Target
y = pd.Series(y_train_raw).map({"new": 0, "used": 1})

# --------------------------------------------------
# 3. Split train / validation
# --------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    df,
    y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

# --------------------------------------------------
# 4. Preprocesamiento
# --------------------------------------------------
num_features = [
    "price_log",
    "initial_quantity",
    "num_images_log",
    "sold_quantity_log",
    "title_len",
]

bool_features = [
    "has_stock",
    "free_shipping",
    "accepts_mp",
    "kw_new",
    "kw_used",
]

cat_features = [
    "category_id",
    "listing_type_id",
    "shipping_mode",
]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", TargetEncoder(), cat_features),
        ("bool", "passthrough", bool_features),
    ]
)

# --------------------------------------------------
# 5. Modelo
# --------------------------------------------------
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

# --------------------------------------------------
# 6. Entrenamiento
# --------------------------------------------------
pipeline.fit(X_train, y_train)

# --------------------------------------------------
# 7. Guardar modelo
# --------------------------------------------------
joblib.dump(pipeline, "models/new_used_model.pkl")

print("✅ Modelo entrenado y guardado en models/new_used_model.pkl")