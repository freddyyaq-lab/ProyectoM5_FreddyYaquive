# 📊 Proyecto de Machine Learning – Predicción de Pago a Tiempo

## 🎯 Objetivo del proyecto
El objetivo de este proyecto es construir un modelo de **Machine Learning** capaz de predecir si un cliente realizará el **pago de un crédito a tiempo**, utilizando información financiera, crediticia y sociodemográfica.

El proyecto sigue una estructura modular y buenas prácticas de ciencia de datos, desde la carga de datos hasta el entrenamiento, evaluación y selección del mejor modelo.

---

## 🧩 Descripción de los módulos

### 1️⃣ Carga de datos (`cargar_datos.py`)
- Lectura del archivo Excel.
- Normalización de nombres de columnas.
- Conversión de fechas.
- Validación de existencia de la variable objetivo `Pago_atiempo`.

---

### 2️⃣ Feature Engineering (`ft_engineering.py`)
- Separación de variables numéricas y categóricas.
- Creación de un `ColumnTransformer` con:
  - Imputación de valores faltantes.
  - Escalado de variables numéricas.
  - Codificación de variables categóricas (One-Hot Encoding).
- Prevención de errores comunes como:
  - Columnas inexistentes.
  - Valores infinitos o demasiado grandes.

---

### 3️⃣ Entrenamiento y Evaluación (`model_training_evaluation.py`)
- Separación de datos en entrenamiento y prueba.
- Entrenamiento de múltiples modelos de clasificación:
  - Regresión Logística
  - Random Forest
  - Gradient Boosting
- Comparación de métricas:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Selección del mejor modelo basado en desempeño.
- Guardado de:
  - Modelo final (`.pkl`)

---

## 📈 Variable objetivo

- **Pago_atiempo**
  - `1` → Cliente paga a tiempo
  - `0` → Cliente no paga a tiempo

---

## 🛠️ Herramientas y tecnologías utilizadas

- **Python**
- **Pandas** – Manipulación de datos
- **NumPy** – Operaciones numéricas
- **Scikit-learn** – Modelado, pipelines y evaluación
- **Joblib** – Persistencia del modelo
- **Excel** – Fuente de datos

---

## ✅ Resultado final

El proyecto genera:
- Un **modelo entrenado y evaluado** listo para ser usado en predicción.
- Un archivo `.pkl` para ser ejecutado en aplicacion de Streamlit.
- Un flujo reproducible y modular siguiendo buenas prácticas de Machine Learning.

