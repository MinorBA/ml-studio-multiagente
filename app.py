# =============================================================================
# app.py  — Punto de entrada Streamlit
# Ejecutar:  streamlit run app.py
# =============================================================================

import streamlit as st

st.set_page_config(
    page_title="ML Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 ML Studio")
st.sidebar.caption("Machine Learning interactivo")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Páginas disponibles**

    Usa el menú de la izquierda para navegar:

    1. 📊 **EDA** — Exploración de datos
    2. 🔬 **No Supervisado** — ACP, HAC, K-Medias, UMAP, t-SNE
    3. 🎯 **Clasificación** — KNN, DT, RF, XGBoost, AdaBoost
    4. 📈 **Regresión** — Lineal, Lasso, Ridge, SVM, Árboles
    5. 🧮 **Matriz Confusión** — Evaluación manual 2×2
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧠 ML Studio — Dashboard Interactivo")
st.markdown("---")

st.markdown(
    """
    ### Bienvenido al laboratorio de Machine Learning

    Este proyecto integra un **backend modularizado** con una interfaz
    **Streamlit** que permite ejecutar análisis de forma interactiva.

    **Arquitectura del proyecto:**
    """
)

st.code("""
ml_studio/
├── app.py                        ← Este archivo (punto de entrada)
├── .streamlit/config.toml        ← Configuración de tema
├── backend/
│   ├── __init__.py               ← Re-exportaciones centralizadas
│   ├── eda_base.py               ← Clase raíz: analisisEDA
│   ├── no_supervisado.py         ← Clustering + ACP (hereda de EDA)
│   ├── supervisado.py            ← Clase padre Supervisado
│   ├── clasificacion.py          ← Clasificación (hereda de Supervisado)
│   ├── regresion.py              ← Regresión   (hereda de Supervisado)
│   └── matriz_confusion.py       ← Utilidad MC 2×2
├── frontend/
│   └── pages/
│       ├── 1_📊_EDA.py
│       ├── 2_🔬_No_Supervisado.py
│       ├── 3_🎯_Clasificacion.py
│       ├── 4_📈_Regresion.py
│       └── 5_🧮_Matriz_Confusion.py
└── requirements.txt
""", language="text")

st.markdown(
    """
    ### Jerarquía de clases

    ```
    analisisEDA
    ├── NoSupervisado       → ACP, HAC, K-Medias, UMAP, t-SNE
    └── Supervisado
        ├── Clasificacion   → KNN, DT, RF, XGBoost, AdaBoost
        └── Regresion       → Lineal, Lasso, Ridge, SVM, Árboles
    ```

    **Instrucciones:**
    1. Suba un archivo CSV desde cualquier página
    2. Configure los parámetros en la barra lateral
    3. Explore los resultados interactivamente
    """
)
