import pandas as pd
import numpy as np

KEYWORDS_NEW = ["nuevo", "sellado", "sin uso", "original"]
KEYWORDS_USED = ["usado", "segunda mano", "reacondicionado", "refurbished"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # Visual / Stock
    # -----------------------------
    df["num_images"] = df["pictures"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    df["num_images_log"] = df["num_images"].apply(lambda x: np.log1p(x))

    df["has_stock"] = df.get("available_quantity", 0).fillna(0) > 0
    df["sold_quantity_log"] = df.get("sold_quantity", 0).fillna(0).apply(lambda x: np.log1p(x))

    # -----------------------------
    # Shipping / Pago
    # -----------------------------
    df["free_shipping"] = df["shipping"].apply(
        lambda x: x.get("free_shipping", False) if isinstance(x, dict) else False
    )

    df["shipping_mode"] = df["shipping"].apply(
        lambda x: x.get("mode") if isinstance(x, dict) else "unknown"
    )

    df["accepts_mp"] = df.get("accepts_mercadopago", False).fillna(False)

    # -----------------------------
    # Precio
    # -----------------------------
    df["price_log"] = df["price"].apply(
        lambda x: 0 if pd.isna(x) or x <= 0 else np.log1p(x)
    )

    # -----------------------------
    # Texto (solo señales indirectas)
    # -----------------------------
    text = df.get("title", "").fillna("").str.lower()

    df["title_len"] = text.apply(len)
    df["kw_new"] = text.apply(lambda x: any(k in x for k in KEYWORDS_NEW))
    df["kw_used"] = text.apply(lambda x: any(k in x for k in KEYWORDS_USED))

    # -----------------------------
    # Marketplace structure
    # -----------------------------
    df["listing_type_id"] = df.get("listing_type_id", "unknown")

    # -----------------------------
    # Feature set final
    # -----------------------------
    features = [
        "price_log",
        "initial_quantity",
        "num_images_log",
        "has_stock",
        "sold_quantity_log",
        "free_shipping",
        "accepts_mp",
        "kw_new",
        "kw_used",
        "title_len",
        "category_id",
        "listing_type_id",
        "shipping_mode",
    ]

    if "condition" in df.columns:
        features.append("condition")

    return df[features]