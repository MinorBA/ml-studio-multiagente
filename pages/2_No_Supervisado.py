# ==================================================================
# pages/2_No_Supervisado.py — AgenteClustering: Análisis No Supervisado
# =============================================================================

import streamlit as st
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.no_supervisado import NoSupervisado
from shared_utils import obtener_df, observaciones_clustering

st.set_page_config(page_title="AgenteClustering", page_icon="🔬", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CSS — mismo sistema visual que 1_EDA.py
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.paso-card {
    background: #f8fafc;
    border-left: 4px solid #8b5cf6;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px; margin: 8px 0;
}
.paso-num   { font-size:11px; font-weight:700; color:#8b5cf6;
              letter-spacing:1.5px; text-transform:uppercase; margin-bottom:2px; }
.paso-titulo{ font-size:16px; font-weight:700; color:#1e293b; margin:0; }
.badge { display:inline-block; padding:4px 12px; border-radius:20px;
         font-size:13px; font-weight:600; margin:4px 4px 4px 0; }
.badge-ok   { background:#dcfce7; color:#166534; }
.badge-warn { background:#fef9c3; color:#854d0e; }
.badge-alg  { background:#ede9fe; color:#5b21b6; }
.alerta-card{ background:#fff7ed; border:1px solid #fb923c; border-radius:8px;
              padding:10px 14px; margin:6px 0; font-size:14px; color:#7c2d12; }
.rec-card   { background:#f0fdf4; border:1px solid #4ade80; border-radius:8px;
              padding:10px 14px; margin:6px 0; font-size:14px; color:#14532d; }
.metric-mini{ text-align:center; padding:12px; background:#f8fafc;
              border-radius:8px; border:1px solid #e2e8f0; }
.metric-mini .val{ font-size:24px; font-weight:700; color:#1e293b; }
.metric-mini .lbl{ font-size:12px; color:#64748b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🔬 Agente Clustering — Análisis No Supervisado")
st.caption("Ejecuta los 7 pasos del agente: percepción → análisis → razonamiento "
           "→ decisión → entrenamiento → comunicación → ejecución")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────────────────────────────────────
df = obtener_df()
if df is None:
    st.info("📂 Primero cargue un CSV en la página **EDA**.")
    st.stop()

df_num = df.select_dtypes(include="number")
if df_num.shape[1] < 2:
    st.error("Se necesitan al menos 2 columnas numéricas. "
             "Vaya a EDA y convierta las categóricas a dummies.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuración del Agente")
st.sidebar.info(
    "🤖 El agente selecciona automáticamente las columnas, determina "
    "el K óptimo y decide qué algoritmos ejecutar según el dataset."
)

col_excluir = st.sidebar.multiselect(
    "Excluir columnas (opcional)",
    options=df_num.columns.tolist(),
    key="cluster_excluir"
)
columnas_usar = [c for c in df_num.columns if c not in col_excluir]

ejecutar = st.sidebar.button("▶ Ejecutar AgenteClustering",
                              type="primary", use_container_width=True)

if "clustering_resultado" not in st.session_state:
    st.session_state["clustering_resultado"] = None

if ejecutar:
    if len(columnas_usar) < 2:
        st.sidebar.error("Necesita al menos 2 columnas para clustering.")
    else:
        with st.spinner("🤖 AgenteClustering ejecutándose…"):
            plan_eda = st.session_state.get("eda_plan", {})
            ns       = NoSupervisado(df_num[columnas_usar])
            resultado = ns.ejecutar(df_num, columnas=columnas_usar,
                                    plan_eda=plan_eda)
            st.session_state["clustering_resultado"] = resultado

resultado = st.session_state["clustering_resultado"]
if resultado is None:
    st.warning("Presiona **▶ Ejecutar AgenteClustering** para comenzar.")
    st.stop()

# ── Extraer datos ─────────────────────────────────────────────────────────
msg    = resultado["mensaje"]["contenido"]
conoc  = resultado["conocimiento"]
bench  = resultado["benchmark"]
figs   = resultado["figuras"]
alertas = conoc.get("alertas", [])
recs    = conoc.get("recomendaciones", [])
errores = msg.get("errores_etapas", [])

# ═════════════════════════════════════════════════════════════════════════════
# PASO 1 — PERCEPCIÓN
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 1</div>
  <div class="paso-titulo">📡 Percepción del Dataset</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas",             conoc.get("n_filas", 0))
c2.metric("Columnas usadas",   conoc.get("n_columnas_usadas", 0))
c3.metric("Filas tras limpieza", conoc.get("n_filas_limpias", 0))
c4.metric("Correlación media", conoc.get("correlacion_media", 0))

with st.expander("📋 Columnas seleccionadas"):
    cols_sel  = conoc.get("columnas_seleccionadas", [])
    ids_excl  = conoc.get("columnas_excluidas_ids", [])
    suf_cols  = conoc.get("tiene_suficientes_cols", True)
    suf_filas = conoc.get("tiene_suficientes_filas", True)

    # Mostrar columnas excluidas automáticamente
    if ids_excl:
        st.markdown(
            f'''<div style="background:#fff7ed;border:1px solid #fb923c;
            border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:14px;
            color:#7c2d12;">
            🪪 <b>Columnas excluidas automáticamente (posibles IDs):</b><br>
            {", ".join(f"<code>{c}</code>" for c in ids_excl)} — 
            más del 95% de valores únicos, no aptos para clustering.
            </div>''',
            unsafe_allow_html=True
        )

    st.markdown("**Columnas usadas:** " + " · ".join(f"`{c}`" for c in cols_sel))

    if suf_cols and suf_filas:
        st.success("✅ Dataset apto para clustering.")
    else:
        if not suf_cols:
            st.error("⛔ Menos de 2 columnas numéricas disponibles tras excluir IDs.")
        if not suf_filas:
            st.warning("⚠️ Menos de 30 filas — resultados pueden no ser fiables.")

# ═════════════════════════════════════════════════════════════════════════════
# PASO 2 — ANÁLISIS + PASO 3 — RAZONAMIENTO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASOS 2 Y 3</div>
  <div class="paso-titulo">📊 Análisis y Razonamiento</div>
</div>
""", unsafe_allow_html=True)

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown(f"**🚨 Alertas** ({len(alertas)})")
    if alertas:
        for a in alertas:
            st.markdown(f'<div class="alerta-card">{a}</div>',
                        unsafe_allow_html=True)
    else:
        st.success("Sin alertas críticas.")

with col_r2:
    st.markdown(f"**💡 Recomendaciones** ({len(recs)})")
    for r in recs:
        st.markdown(f'<div class="rec-card">{r}</div>',
                    unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PASO 4 — DECISIÓN DE ALGORITMOS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 4</div>
  <div class="paso-titulo">🧩 Algoritmos Seleccionados</div>
</div>
""", unsafe_allow_html=True)

algs = msg.get("algoritmos_ejecutados", [])
k    = msg.get("k_optimo", "—")
st.markdown(
    " ".join(f'<span class="badge badge-alg">⚙️ {a}</span>' for a in algs),
    unsafe_allow_html=True
)
st.caption(f"K óptimo determinado por Silhouette Score: **{k} clusters**")

# ═════════════════════════════════════════════════════════════════════════════
# PASO 5 — BENCHMARK (tabla + gráficos)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 5</div>
  <div class="paso-titulo">🏆 Benchmark de Algoritmos</div>
</div>
""", unsafe_allow_html=True)

# Métricas del ganador
mejor_alg = msg.get("mejor_algoritmo", "N/A")
mejor_sil  = msg.get("mejor_silhouette", float("nan"))

mc1, mc2, mc3 = st.columns(3)
mc1.metric("Mejor algoritmo",   mejor_alg)
mc2.metric("Silhouette (mejor)", f"{mejor_sil:.4f}" if mejor_sil == mejor_sil else "N/A")
mc3.metric("DBSCAN ruido",      msg.get("dbscan_ruido", 0))

# Tabla benchmark coloreada
if not bench.empty:
    st.markdown("**Tabla comparativa:**")

    def color_silhouette(val):
        try:
            v = float(val)
            if v >= 0.7:   return "background-color:#dcfce7;color:#166534"
            if v >= 0.5:   return "background-color:#fef9c3;color:#854d0e"
            if v >= 0.25:  return "background-color:#ffedd5;color:#9a3412"
            return "background-color:#fee2e2;color:#991b1b"
        except:
            return ""

    def color_db(val):
        try:
            v = float(val)
            if v < 0.5:  return "background-color:#dcfce7"
            if v < 1.0:  return "background-color:#fef9c3"
            return "background-color:#fee2e2"
        except:
            return ""

    styled = bench.style\
        .applymap(color_silhouette, subset=["Silhouette"])\
        .applymap(color_db,         subset=["Davies_Bouldin"])\
        .format(precision=4)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Leyenda de interpretación
    with st.expander("ℹ️ Interpretación de métricas"):
        st.markdown("""
        | Métrica | Mejor cuando | Verde | Amarillo | Rojo |
        |---------|-------------|-------|----------|------|
        | **Silhouette** | Mayor (max 1.0) | ≥ 0.7 | 0.5–0.7 | < 0.5 |
        | **Davies-Bouldin** | Menor (min 0) | < 0.5 | 0.5–1.0 | > 1.0 |
        | **Calinski-Harabasz** | Mayor | — | — | — |
        """)

# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES POR ALGORITMO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">VISUALIZACIONES</div>
  <div class="paso-titulo">📉 Gráficos por Algoritmo</div>
</div>
""", unsafe_allow_html=True)

# ── Método del Codo ──────────────────────────────────────────────────────
if "codo" in figs:
    with st.expander("📐 Método del Codo — K-Means", expanded=True):
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.pyplot(figs["codo"], use_container_width=True)
        with col_c2:
            k_opt = msg.get("k_optimo", "N/A")
            st.markdown(f"**K óptimo seleccionado: {k_opt} clusters**")
            st.markdown(
                f'''<div style="background:#f0fdf4;border-radius:8px;
                padding:10px;border:1px solid #4ade80;font-size:13px;
                color:#166534;">
                El punto de inflexión de la curva indica el número óptimo
                de clusters. A partir de ese punto, agregar más clusters
                no reduce significativamente la inercia.
                </div>''', unsafe_allow_html=True)

# ── K-Medias ─────────────────────────────────────────────────────────────
if "kmedias_scatter" in figs or "kmedias_radar" in figs:
    with st.expander("🔵 K-Medias / K-Medoids", expanded=True):
        col_km1, col_km2 = st.columns(2)
        with col_km1:
            if figs.get("kmedias_scatter"):
                st.markdown("**Scatter de Clusters**")
                st.pyplot(figs["kmedias_scatter"], use_container_width=True)
        with col_km2:
            if figs.get("kmedias_radar"):
                st.markdown("**Radar de Centroides**")
                st.pyplot(figs["kmedias_radar"], use_container_width=True)

# ── HAC ──────────────────────────────────────────────────────────────────
hac_keys = [k for k in figs if k.startswith("hac_dendro_")]
if hac_keys:
    with st.expander("🌲 Clustering Jerárquico (HAC)", expanded=True):
        metodos = [k.replace("hac_dendro_","").capitalize() for k in hac_keys]
        tabs    = st.tabs(metodos)
        for tab, key in zip(tabs, hac_keys):
            with tab:
                if figs.get(key):
                    st.pyplot(figs[key], use_container_width=True)
        if figs.get("hac_centroides"):
            st.markdown("**Centroides HAC:**")
            st.pyplot(figs["hac_centroides"], use_container_width=True)

# ── DBSCAN ────────────────────────────────────────────────────────────────
if "dbscan_scatter" in figs:
    with st.expander("🌑 DBSCAN — Clustering por Densidad"):
        col_db1, col_db2 = st.columns([2, 1])
        with col_db1:
            st.pyplot(figs["dbscan_scatter"], use_container_width=True)
        with col_db2:
            st.metric("Puntos de ruido", msg.get("dbscan_ruido", 0))
            st.caption("DBSCAN detecta clusters de forma arbitraria "
                       "y marca como ruido los puntos aislados.")

# ── ACP ───────────────────────────────────────────────────────────────────
if "acp_plano" in figs or "acp_circulo" in figs:
    with st.expander("📐 ACP — Reducción Dimensional"):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            if figs.get("acp_plano"):
                st.markdown("**Plano Principal**")
                st.pyplot(figs["acp_plano"], use_container_width=True)
        with col_a2:
            if figs.get("acp_circulo"):
                st.markdown("**Círculo de Correlación**")
                st.pyplot(figs["acp_circulo"], use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PASO 6 — MENSAJE MAS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 6 — COMUNICACIÓN</div>
  <div class="paso-titulo">📨 Mensaje al AgenteCoordinador</div>
</div>
""", unsafe_allow_html=True)

col_m1, col_m2 = st.columns([1, 2])
with col_m1:
    ts = resultado["mensaje"].get("timestamp","")[:19].replace("T"," ")
    st.metric("Agente",           resultado["mensaje"].get("agente",""))
    st.metric("Timestamp",        ts)
    st.metric("Figuras generadas",len(figs))
    st.metric("Errores en etapas",len(errores))
    if errores:
        with st.expander("⚠️ Ver errores"):
            for e in errores:
                st.error(f"**{e.get('paso','')}**: {e.get('error','')}")

with col_m2:
    with st.expander("📋 Mensaje completo (JSON)", expanded=False):
        st.json(msg)

# Resumen ejecutivo
# Observaciones del benchmark de clustering
if not bench.empty:
    obs_cl = observaciones_clustering(
        bench, msg.get("k_optimo", 2), msg.get("mejor_algoritmo", ""))
    if obs_cl:
        st.markdown("**Observaciones:**")
        for o in obs_cl:
            st.markdown(f"• {o}")
        st.markdown("")

st.markdown("**📊 Resumen ejecutivo:**")
cs1, cs2, cs3, cs4 = st.columns(4)
for col_c, val, lbl in [
    (cs1, mejor_alg,                      "Mejor algoritmo"),
    (cs2, f"{mejor_sil:.4f}",             "Silhouette ganador"),
    (cs3, k,                              "K óptimo"),
    (cs4, len(algs),                      "Algoritmos ejecutados"),
]:
    col_c.markdown(
        f'<div class="metric-mini"><div class="val">{val}</div>'
        f'<div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True
    )
