# =============================================================================
# frontend/pages/1_📊_EDA.py
# Página de Análisis Exploratorio de Datos
# =============================================================================

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.eda_base import analisisEDA

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Análisis Exploratorio de Datos (EDA)")
st.markdown("---")

# ──────────────────────── CARGA DEL DATASET ────────────────────────────────
archivo = st.sidebar.file_uploader("📁 Subir CSV", type=["csv"], key="eda_csv")
sep     = st.sidebar.radio("Separador", [",", ";"], horizontal=True)

if archivo is None:
    st.info("⬅️ Suba un archivo CSV desde la barra lateral para comenzar.")
    st.stop()

sep_num = 1 if sep == "," else 2
try:
    df = pd.read_csv(archivo, sep=sep, decimal=".")
except Exception as e:
    st.error(f"Error al leer el archivo: {e}")
    st.stop()

eda = analisisEDA()
eda.cargar_dataframe(df)

# Guardar en session_state para las demás páginas
st.session_state["df_original"] = df.copy()

# ──────────────────────── INFORMACIÓN GENERAL ──────────────────────────────
st.subheader("🔍 Información General")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Filas", df.shape[0])
col2.metric("Columnas", df.shape[1])
col3.metric("Nulos totales", int(df.isnull().sum().sum()))
col4.metric("Duplicados", int(df.duplicated().sum()))

with st.expander("📋 Tipos de datos"):
    st.dataframe(eda.tipoDatos().reset_index().rename(
        columns={"index": "Columna", 0: "Tipo"}))

with st.expander("📊 Estadísticas descriptivas"):
    r = eda.resumen()
    st.dataframe(r["describe"])

with st.expander("🔢 Valores faltantes por columna"):
    falt = eda.valores_faltantes()
    st.dataframe(falt.reset_index().rename(
        columns={"index": "Columna", 0: "Nulos"}))

with st.expander("🗄️ Vista previa del DataFrame"):
    st.dataframe(df.head(50))

# ──────────────────────── LIMPIEZA INTERACTIVA ─────────────────────────────
st.markdown("---")
st.subheader("🧹 Limpieza de Datos")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("🗑️ Eliminar filas duplicadas"):
        n = eda.eliminarDuplicados()
        st.success(f"Se eliminaron {n} filas duplicadas.")
        st.session_state["df_original"] = eda.df.copy()

with col_b:
    if st.button("🗑️ Eliminar filas con nulos"):
        n = eda.eliminarNulos()
        st.success(f"Se eliminaron {n} filas con valores nulos.")
        st.session_state["df_original"] = eda.df.copy()

cols_eliminar = st.multiselect("Seleccione columnas a eliminar:",
                                eda.df.columns.tolist())
if cols_eliminar and st.button("Eliminar columnas seleccionadas"):
    eda.eliminarColumnas(cols_eliminar)
    st.success(f"Columnas eliminadas: {cols_eliminar}")
    st.session_state["df_original"] = eda.df.copy()

if st.button("🔄 Convertir categóricas a dummies"):
    eda.analisisCompleto()
    st.success("Variables categóricas convertidas a dummies.")
    st.session_state["df_original"] = eda.df.copy()

# ──────────────────────── GRÁFICOS ─────────────────────────────────────────
st.markdown("---")
st.subheader("📉 Visualizaciones")

grafico = st.selectbox("Seleccione un gráfico:", [
    "Boxplots", "Histogramas", "Distribuciones (con KDE)",
    "Densidad", "Mapa de Correlación", "Distribución de Clase",
])

if grafico == "Boxplots":
    st.pyplot(eda.fig_boxplot())
elif grafico == "Histogramas":
    st.pyplot(eda.fig_histogramas())
elif grafico == "Distribuciones (con KDE)":
    st.pyplot(eda.fig_distribucion())
elif grafico == "Densidad":
    st.pyplot(eda.fig_densidad())
elif grafico == "Mapa de Correlación":
    st.pyplot(eda.fig_correlacion())
elif grafico == "Distribución de Clase":
    col_clase = st.selectbox("Variable objetivo:", eda.df.columns.tolist())
    st.pyplot(eda.fig_histograma_clase(col_clase))

# Actualizar session_state con df final limpio
st.session_state["df_limpio"] = eda.df.copy()
