# =============================================================================
# frontend/pages/2_🔬_No_Supervisado.py
# Página de Análisis No Supervisado (ACP + Clustering)
# =============================================================================

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.no_supervisado import NoSupervisado

st.set_page_config(page_title="No Supervisado", page_icon="🔬", layout="wide")
st.title("🔬 Análisis No Supervisado")
st.markdown("---")

# ──────────────────────── CARGA ────────────────────────────────────────────
archivo = st.sidebar.file_uploader("📁 Subir CSV", type=["csv"],
                                    key="ns_csv")
sep = st.sidebar.radio("Separador", [",", ";"], horizontal=True, key="ns_sep")

if archivo is None and "df_limpio" not in st.session_state:
    st.info("⬅️ Suba un CSV o ejecute primero la página EDA.")
    st.stop()

if archivo is not None:
    df = pd.read_csv(archivo, sep=sep, decimal=".")
else:
    df = st.session_state["df_limpio"]

# Solo numéricas
df_num = df.select_dtypes(include="number")
if df_num.shape[1] < 2:
    st.error("Se necesitan al menos 2 columnas numéricas.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parámetros")

tecnica = st.sidebar.selectbox(
    "Técnica",
    ["ACP", "HAC", "K-Medias / K-Medoids", "UMAP", "t-SNE"],
)

ns = NoSupervisado(df_num)

# ──────────────────────── ACP ──────────────────────────────────────────────
if tecnica == "ACP":
    n_comp = st.sidebar.slider("Componentes", 2, min(10, df_num.shape[1]), 2)
    if st.button("▶️ Ejecutar ACP"):
        with st.spinner("Calculando ACP..."):
            r = ns.ACP(n_comp)
        tab1, tab2, tab3 = st.tabs(
            ["Plano Principal", "Círculo Correlación", "Sobreposición"])
        with tab1:
            st.pyplot(r["plano"])
        with tab2:
            st.pyplot(r["circulo"])
        with tab3:
            st.pyplot(r["sobreposicion"])

        st.subheader("📊 Varianza explicada")
        st.dataframe(r["var_explicada"])
        st.subheader("📊 Correlación de variables")
        st.dataframe(r["correlacion_var"])

# ──────────────────────── HAC ──────────────────────────────────────────────
elif tecnica == "HAC":
    n_cl = st.sidebar.slider("Clusters para centroides", 2, 10, 3)
    if st.button("▶️ Ejecutar HAC"):
        with st.spinner("Generando dendrogramas y Silhouette Scores..."):
            r = ns.HAC(n_cl)

        st.subheader("🌳 Dendrogramas")
        for nombre, fig in r["dendrogramas"].items():
            st.markdown(f"**Método: {nombre}**")
            st.pyplot(fig)

        st.subheader("📊 Centroides (Ward)")
        st.pyplot(r["centroides"])

        st.subheader("📈 Silhouette Scores")
        for nombre, datos in r["silhouette"].items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.pyplot(datos["fig"])
            with col2:
                st.metric(f"Mejor k ({nombre})", datos["mejor_k"])
                st.metric("Score", f"{datos['score']:.4f}")

# ──────────────────────── K-MEDIAS ─────────────────────────────────────────
elif tecnica == "K-Medias / K-Medoids":
    n_cl = st.sidebar.slider("Número de clusters", 2, 10, 3, key="km_k")
    if st.button("▶️ Ejecutar K-Medias y K-Medoids"):
        with st.spinner("Entrenando modelos..."):
            r = ns.KMedia(n_cl)

        tab1, tab2 = st.tabs(["K-Medias", "K-Medoids"])
        with tab1:
            st.pyplot(r["kmedias_scatter"])
            st.pyplot(r["kmedias_radar"])
            st.metric("Silhouette Score", f"{r['score_kmedias']:.4f}")
        with tab2:
            st.pyplot(r["kmedoids_scatter"])
            st.pyplot(r["kmedoids_radar"])
            st.metric("Silhouette Score", f"{r['score_kmedoids']:.4f}")

# ──────────────────────── UMAP ─────────────────────────────────────────────
elif tecnica == "UMAP":
    n_comp = st.sidebar.slider("Componentes", 2, 5, 2, key="umap_comp")
    n_neigh = st.sidebar.slider("Vecinos", 5, 50, 15, key="umap_neigh")
    if st.button("▶️ Ejecutar UMAP"):
        with st.spinner("Aplicando UMAP..."):
            r = ns.UMAP(n_comp, n_neigh)
        st.pyplot(r["fig"])

# ──────────────────────── T-SNE ────────────────────────────────────────────
elif tecnica == "t-SNE":
    seed = st.sidebar.number_input("Random state", 0, 999, 42, key="tsne_rs")
    if st.button("▶️ Ejecutar t-SNE"):
        with st.spinner("Aplicando t-SNE..."):
            r = ns.TSNE(random_state=seed)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.pyplot(r["fig_silhouette"])
            st.pyplot(r["fig_scatter"])
        with col2:
            st.metric("Mejor k", r["mejor_k"])
            st.metric("Silhouette Score", f"{r['mejor_score']:.4f}")

# ──────────────────────── BENCHMARK ────────────────────────────────────────
st.markdown("---")
bm = ns.benchmark()
if not bm.empty:
    st.subheader("🏆 Benchmark — Comparación de modelos")
    st.dataframe(bm, use_container_width=True)
