# Dataset – Mercado Libre Listings (MLA)

## 📌 Descripción general

Este directorio contiene el dataset utilizado para entrenar y evaluar el modelo de clasificación **New vs Used**.

El archivo principal está en formato **JSON Lines**, donde cada línea representa una publicación individual de Mercado Libre con su información asociada.

---

## 📂 Archivos

### `MLA_100k.jsonlines`

- **Formato:** JSON Lines (`.jsonlines`)
- **Cantidad de registros:** ~100.000 publicaciones
- **Granularidad:** 1 publicación por línea
- **Idioma:** Español
- **Site:** Mercado Libre Argentina (MLA)

Cada línea contiene un objeto JSON con información estructural y semiestructural del marketplace.

---

## 🧾 Campos relevantes utilizados

A continuación se listan los principales campos del dataset utilizados directa o indirectamente en el modelo:

### 🎯 Target
- `condition`: estado del ítem (`new` | `used`)

### 💰 Precio y performance
- `price`
- `sold_quantity`
- `available_quantity`

### 📦 Logística y pago
- `shipping.free_shipping`
- `shipping.mode`
- `accepts_mercadopago`

### 🖼️ Visual
- `pictures` (cantidad de imágenes)

### 🧩 Estructura del marketplace
- `category_id`
- `listing_type_id`

### 📝 Texto
- `title`

---

## 🔄 Preprocesamiento

- Los registros se cargan línea por línea para optimizar memoria.
- Los campos anidados (ej. `shipping`, `pictures`) se normalizan durante el feature engineering.
- No se utilizan campos que puedan generar **data leakage** (ej. eventos posteriores a la publicación).

---

## ⚠️ Notas importantes

- El dataset se utiliza **exclusivamente con fines de evaluación técnica**.
- No se realizan modificaciones sobre el archivo original.
- El split de entrenamiento y evaluación se define a nivel de código (`build_dataset`).

---

## 📌 Observaciones

El dataset refleja escenarios reales del marketplace, incluyendo:
- Heterogeneidad de categorías
- Variabilidad en precios y logística
- Diferentes tipos de publicaciones

Esto permite evaluar la capacidad del modelo para generalizar en condiciones realistas.

---