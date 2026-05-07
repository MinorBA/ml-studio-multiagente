# =============================================================================
# frontend/pages/4_📈_Regresion.py
# Página de Regresión Supervisada
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.regresion import Regresion
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Regresión", page_icon="📈", layout="wide")
st.title("📈 Regresión Supervisada")
st.markdown("---")

# ──────────────────────── CARGA ────────────────────────────────────────────
archivo = st.sidebar.file_uploader("📁 Subir CSV", type=["csv"],
                                    key="reg_csv")
sep = st.sidebar.radio("Separador", [",", ";"], horizontal=True, key="reg_sep")

if archivo is None and "df_limpio" not in st.session_state:
    st.info("⬅️ Suba un CSV o ejecute primero la página EDA.")
    st.stop()

if archivo is not None:
    df = pd.read_csv(archivo, sep=sep, decimal=".")
else:
    df = st.session_state["df_limpio"]

df_num = df.select_dtypes(include="number")
if df_num.shape[1] < 2:
    st.error("Se necesitan al menos 2 columnas numéricas.")
    st.stop()

st.sidebar.markdown("---")
col_target = st.sidebar.selectbox("🎯 Variable a predecir:",
                                    df_num.columns.tolist())
test_pct   = st.sidebar.slider("% Test", 10, 50, 25) / 100

# Preparar datos
X = df_num.drop(columns=[col_target])
y = df_num[col_target].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_pct, random_state=42)

reg = Regresion(df_num, columna_target=col_target, test_size=test_pct)

# ──────────────────────── MODELO INDIVIDUAL ────────────────────────────────
st.subheader("⚙️ Ejecutar un modelo")

modelo_sel = st.selectbox("Modelo:", [
    "Regresión Lineal Múltiple", "Lasso", "Lasso CV",
    "Ridge", "Ridge CV", "SVM (rbf)", "SVM (linear)", "SVM (poly)",
    "Árbol de Decisión", "Random Forest", "Gradient Boosting",
])

if st.button("▶️ Ejecutar modelo"):
    with st.spinner(f"Entrenando {modelo_sel}..."):
        if modelo_sel == "Regresión Lineal Múltiple":
            err = reg.RegLinMult(X_train, X_test, y_train, y_test)
        elif modelo_sel == "Lasso":
            err = reg.RegLasso(X_train, X_test, y_train, y_test)
        elif modelo_sel == "Lasso CV":
            err = reg.RegLassoCV(X_train, X_test, y_train, y_test)
        elif modelo_sel == "Ridge":
            err = reg.RegRidge(X_train, X_test, y_train, y_test)
        elif modelo_sel == "Ridge CV":
            err = reg.RegRidgeCV(X_train, X_test, y_train, y_test)
        elif modelo_sel == "SVM (rbf)":
            err, _, _ = reg.SVM(X_train, X_test, y_train, y_test)
        elif modelo_sel == "SVM (linear)":
            _, err, _ = reg.SVM(X_train, X_test, y_train, y_test)
        elif modelo_sel == "SVM (poly)":
            _, _, err = reg.SVM(X_train, X_test, y_train, y_test)
        elif modelo_sel == "Árbol de Decisión":
            err = reg.DecisionTreeReg(X_train, X_test, y_train, y_test)
        elif modelo_sel == "Random Forest":
            err = reg.RandomForestReg(X_train, X_test, y_train, y_test)
        elif modelo_sel == "Gradient Boosting":
            err = reg.XGBoostingReg(X_train, X_test, y_train, y_test)

    # Mostrar métricas
    c1, c2, c3, c4 = st.columns(4)
    vals = err.set_index("Tipo")["Error"]
    c1.metric("RMSE", f"{vals['RMSE']:.4f}")
    c2.metric("MAE",  f"{vals['MAE']:.4f}")
    c3.metric("ER",   f"{vals['ER']:.4f}")
    c4.metric("R²",   f"{vals['R2']:.4f}")
    st.dataframe(err, use_container_width=True)

# ──────────────────────── COMPARACIONES ────────────────────────────────────
st.markdown("---")
st.subheader("📊 Tablas comparativas")

comparacion = st.selectbox("Comparación:", [
    "Lasso vs Lasso CV", "Ridge vs Ridge CV",
    "SVM (3 kernels)", "Comparación Final (todos los modelos)",
])

if st.button("▶️ Ejecutar comparación"):
    with st.spinner("Calculando..."):
        if comparacion == "Lasso vs Lasso CV":
            tabla = reg.comparacion_lasso(X_train, X_test, y_train, y_test)
        elif comparacion == "Ridge vs Ridge CV":
            tabla = reg.comparacion_ridge(X_train, X_test, y_train, y_test)
        elif comparacion == "SVM (3 kernels)":
            tabla = reg.comparacion_svm(X_train, X_test, y_train, y_test)
        elif comparacion == "Comparación Final (todos los modelos)":
            tabla = reg.comparacion_final(X_train, X_test, y_train, y_test)

    st.dataframe(tabla.style.highlight_min(
        subset=["RMSE", "MAE", "ER"], color="#d4edda").highlight_max(
        subset=["R2"], color="#d4edda"), use_container_width=True)

    # Gráfico de barras
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    tabla[["RMSE", "R2"]].plot(kind="bar", ax=ax)
    ax.set_title("RMSE y R² por modelo")
    ax.set_ylabel("Valor")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# ──────────────────────── INFO VARIABLE OBJETIVO ───────────────────────────
st.markdown("---")
with st.expander("📊 Estadísticas de la variable objetivo"):
    info = reg.res_var_pred(y)
    for k, v in info.items():
        st.write(f"**{k}:** {v}")
