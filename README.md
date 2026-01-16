# Mercado Libre – New vs Used Classification

## 📌 Objetivo

Construir un modelo de Machine Learning capaz de clasificar publicaciones de Mercado Libre como **`new`** o **`used`**, utilizando exclusivamente información disponible al momento de la publicación y priorizando un enfoque **escalable, interpretable y production-ready**.

---

## 📂 Estructura del proyecto
```
ml/
├── data/
│   ├── MLA_100k.jsonlines       # Dataset original
│   └── README.md                # Descripción del dataset
├── features/
│   └── feature_builder.py       # Feature engineering
├── models/
│   └── new_used_model.pkl       # Modelo entrenado
├── train.py                     # Entrenamiento
├── evaluate.py                  # Evaluación
└── report.md                    # Análisis y conclusiones
```

---

## 📊 Dataset

- **Fuente:** Publicaciones reales de Mercado Libre (MLA)
- **Formato:** JSON Lines
- **Tamaño:** ~100.000 publicaciones
- **Target:** `condition` → `{ new, used }`

El dataset contiene información estructural del marketplace como precio, logística, tipo de publicación, imágenes, categoría, texto del título, entre otros.

---

## 🧠 Enfoque de la solución

Se priorizó un enfoque **práctico y realista**, alineado con escenarios de producción:

- ❌ Sin TF-IDF pesado ni modelos deep learning
- ✅ Features estructurales y de dominio del marketplace
- ✅ Modelo interpretable y eficiente
- ✅ Sin data leakage

---

## ⚙️ Feature Engineering

Las features se agrupan en cuatro grandes categorías:

### 1️⃣ Precio y performance
- `price_log`: log(1 + price)
- `sold_quantity_log`: log(1 + sold_quantity)

### 2️⃣ Visual y stock
- `num_images_log`
- `has_stock`

### 3️⃣ Logística y pago
- `free_shipping`
- `accepts_mp`
- `shipping_mode`

### 4️⃣ Señales semánticas livianas
- `kw_new`, `kw_used` (keywords en título)
- `title_len`

### 5️⃣ Estructura del marketplace
- `category_id`
- `listing_type_id`

Estas features son **baratas de computar**, estables en el tiempo y altamente explicativas del comportamiento real del marketplace.

---

## 🤖 Modelo

- **Algoritmo:** LightGBM (Gradient Boosted Trees)
- **Motivos de elección:**
  - Manejo eficiente de variables heterogéneas
  - Buen desempeño sin heavy feature scaling
  - Interpretabilidad
  - Escalabilidad en producción

- **Preprocesamiento:**
  - Numéricas: `StandardScaler`
  - Categóricas: `TargetEncoder`
  - Booleanas: passthrough

---

## 📈 Evaluación

La evaluación se realizó sobre un **hold-out fijo de 10.000 publicaciones**, separado del entrenamiento.

### Resultados finales:

```
| Métrica | Valor |
|------|------|
| Accuracy | **0.8862** |
| Recall (USED) | **0.8742** |
| F1-score macro | **0.89** |

          precision    recall  f1-score   support

       new       0.89      0.90      0.89
      used       0.88      0.87      0.88

accuracy                           0.89
```

Estos resultados superan holgadamente el umbral solicitado en la consigna.

---

## ▶️ Ejecución

### Entrenamiento
```
python train.py
```

### Evaluación
```
python evaluate.py
```

🧩 Decisiones clave
	•	Se evitó el uso de texto pesado para garantizar latencia baja y escalabilidad.
	•	Se utilizaron señales reales del negocio (logística, tipo de publicación, performance histórica).
	•	Se mantuvo una separación clara entre feature engineering, modelo y evaluación.
	•	Se priorizó un diseño reproducible y fácil de mantener.

⸻

📌 Conclusión

El modelo logra un desempeño sólido utilizando únicamente información disponible al momento de la publicación, con un enfoque alineado a estándares reales de producción en marketplaces de gran escala.

La solución es:
	•	✔️ Precisa
	•	✔️ Escalable
	•	✔️ Interpretable
	•	✔️ Fácil de mantener

⸻

✍️ Autor

Esteban Baquero
Tech Lead / AI Engineer
Prueba técnica – Mercado Libre