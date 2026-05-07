# =============================================================================
# frontend/pages/3_🎯_Clasificacion.py
# Página de Clasificación Supervisada
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.clasificacion import Clasificacion

st.set_page_config(page_title="Clasificación", page_icon="🎯", layout="wide")
st.title("🎯 Clasificación Supervisada")
st.markdown("---")

# ──────────────────────── CARGA ────────────────────────────────────────────
archivo = st.sidebar.file_uploader("📁 Subir CSV", type=["csv"],
                                    key="clf_csv")
sep = st.sidebar.radio("Separador", [",", ";"], horizontal=True, key="clf_sep")

if archivo is None and "df_limpio" not in st.session_state:
    st.info("⬅️ Suba un CSV o ejecute primero la página EDA.")
    st.stop()

if archivo is not None:
    df = pd.read_csv(archivo, sep=sep, decimal=".")
else:
    df = st.session_state["df_limpio"]

st.sidebar.markdown("---")
col_target = st.sidebar.selectbox("🎯 Variable objetivo:", df.columns.tolist())
test_size  = st.sidebar.slider("% Test", 10, 50, 25) / 100

clf = Clasificacion(df, columna_target=col_target, test_size=test_size)

# ──────────────────────── MODELO INDIVIDUAL ────────────────────────────────
st.subheader("⚙️ Ejecutar un modelo")

modelo_sel = st.selectbox("Algoritmo:", [
    "KNN", "Árbol de Decisión", "Random Forest",
    "Gradient Boosting", "AdaBoost",
])

mapeo = {
    "KNN": "KNN",
    "Árbol de Decisión": "DT",
    "Random Forest": "RF",
    "Gradient Boosting": "XG",
    "AdaBoost": "ADA",
}

# Hiperparámetros dinámicos
params = {}
c1, c2 = st.columns(2)
if modelo_sel == "KNN":
    with c1:
        params["n_neighbors"] = st.slider("k (vecinos)", 1, 20, 5)
    with c2:
        params["algorithm"] = st.selectbox(
            "Algoritmo búsqueda",
            ["auto", "ball_tree", "kd_tree", "brute"])
elif modelo_sel in ("Árbol de Decisión", "Random Forest", "Gradient Boosting"):
    with c1:
        params["max_depth"] = st.slider("max_depth", 1, 15, 4)
    with c2:
        params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2)
    if modelo_sel in ("Random Forest", "Gradient Boosting"):
        params["n_estimators"] = st.slider("n_estimators", 10, 500, 100,
                                            step=10)
elif modelo_sel == "AdaBoost":
    params["n_estimators"] = st.slider("n_estimators", 10, 300, 100, step=10)

if st.button("▶️ Ejecutar modelo"):
    with st.spinner(f"Entrenando {modelo_sel}..."):
        r = clf.ejecutar_modelo(mapeo[modelo_sel], **params)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Precisión Global", f"{r['Precisión Global']:.4f}")
    col_b.metric("Error Global", f"{r['Error Global']:.4f}")
    col_c.metric("Modelo", modelo_sel)

    t1, t2 = st.tabs(["Matriz de Confusión", "Precisión por categoría"])
    with t1:
        clases = list(np.unique(r["y_test"]))
        fig = Clasificacion.fig_matriz_confusion(
            r["y_test"], r["y_pred"], clases)
        st.pyplot(fig)
    with t2:
        st.dataframe(r["Precisión por categoría"], use_container_width=True)

# ──────────────────────── BARRIDOS ─────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Barrido de hiperparámetros")

barrido_sel = st.selectbox("Algoritmo para barrido:", [
    "KNN", "Árbol de Decisión", "Random Forest",
    "Gradient Boosting", "AdaBoost",
], key="barrido_alg")

if st.button("▶️ Ejecutar barrido"):
    with st.spinner(f"Barriendo hiperparámetros de {barrido_sel}..."):
        if barrido_sel == "KNN":
            tabla = clf.barrido_knn()
        elif barrido_sel == "Árbol de Decisión":
            tabla = clf.barrido_dt()
        elif barrido_sel == "Random Forest":
            tabla = clf.barrido_rf()
        elif barrido_sel == "Gradient Boosting":
            tabla = clf.barrido_xg()
        elif barrido_sel == "AdaBoost":
            tabla = clf.barrido_ada()
    st.dataframe(tabla, use_container_width=True)
    st.line_chart(tabla.set_index(tabla.columns[0])["PG"])

# ──────────────────────── BENCHMARK ────────────────────────────────────────
st.markdown("---")
st.subheader("🏆 Benchmark — Comparación de todos los algoritmos")

clases_unicas = df[col_target].unique().tolist()
c1, c2 = st.columns(2)
with c1:
    cp = st.selectbox("Clase positiva:", clases_unicas,
                       index=len(clases_unicas) - 1, key="bm_cp")
with c2:
    cn = st.selectbox("Clase negativa:", clases_unicas, index=0, key="bm_cn")

if st.button("▶️ Ejecutar Benchmark"):
    with st.spinner("Ejecutando todos los modelos..."):
        tabla_bm = clf.benchmark(clase_positiva=cp, clase_negativa=cn)
    st.dataframe(tabla_bm.style.highlight_max(
        subset=["PG"], color="#d4edda").highlight_min(
        subset=["EG"], color="#d4edda"), use_container_width=True)
