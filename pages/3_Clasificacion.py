# =============================================================================
# pages/3_Clasificacion.py — AgenteClasificacion
# =============================================================================

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.clasificacion import Clasificacion
from shared_utils import obtener_df, observaciones_clasificacion

st.set_page_config(page_title="AgenteClasificacion",
                   page_icon="🎯", layout="wide")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.paso-card { background:#f8fafc; border-left:4px solid #10b981;
             border-radius:0 8px 8px 0; padding:12px 18px; margin:8px 0; }
.paso-num  { font-size:11px; font-weight:700; color:#10b981;
             letter-spacing:1.5px; text-transform:uppercase; margin-bottom:2px; }
.paso-titulo { font-size:16px; font-weight:700; color:#1e293b; margin:0; }
.badge { display:inline-block; padding:4px 12px; border-radius:20px;
         font-size:13px; font-weight:600; margin:4px 4px 4px 0; }
.badge-clf   { background:#dbeafe; color:#1d4ed8; }
.badge-bin   { background:#fef9c3; color:#854d0e; }
.badge-multi { background:#ffedd5; color:#9a3412; }
.badge-ok    { background:#dcfce7; color:#166534; }
.badge-warn  { background:#fef9c3; color:#854d0e; }
.badge-model { background:#ede9fe; color:#5b21b6; }
.alerta-card { background:#fff7ed; border:1px solid #fb923c; border-radius:8px;
               padding:10px 14px; margin:6px 0; font-size:14px; color:#7c2d12; }
.rec-card    { background:#f0fdf4; border:1px solid #4ade80; border-radius:8px;
               padding:10px 14px; margin:6px 0; font-size:14px; color:#14532d; }
.metric-mini { text-align:center; padding:12px; background:#f8fafc;
               border-radius:8px; border:1px solid #e2e8f0; }
.metric-mini .val { font-size:22px; font-weight:700; color:#1e293b; }
.metric-mini .lbl { font-size:12px; color:#64748b; }
</style>
""", unsafe_allow_html=True)

# ─── ENCABEZADO ───────────────────────────────────────────────────────────────
st.markdown("## 🎯 Agente Clasificación — Aprendizaje Supervisado")
st.caption("Ejecuta los 7 pasos del agente: percepción → análisis → razonamiento "
           "→ decisión → inferencia → comunicación → ejecución")
st.markdown("---")

# ─── DATOS ────────────────────────────────────────────────────────────────────
df = obtener_df()
if df is None:
    st.info("📂 Primero cargue un CSV en la página **EDA**.")
    st.stop()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuración del Agente")

# Variable objetivo: preferir la inferida por EDA, sino selector manual
var_eda = st.session_state.get("eda_variable_objetivo")
tipo_eda = st.session_state.get("eda_tipo_problema", "")

if var_eda and tipo_eda == "clasificacion" and var_eda in df.columns:
    st.sidebar.success(f"🤖 Variable inferida por AgenteEDA: **{var_eda}**")
    variable_objetivo = var_eda
else:
    variable_objetivo = st.sidebar.selectbox(
        "Variable objetivo (Y)",
        options=df.columns.tolist(),
        key="clf_target"
    )
    st.sidebar.caption("Seleccione la columna a predecir.")

ejecutar = st.sidebar.button("▶ Ejecutar AgenteClasificacion",
                              type="primary", use_container_width=True)

if "clasificacion_resultado" not in st.session_state:
    st.session_state["clasificacion_resultado"] = None

if ejecutar:
    with st.spinner("🤖 AgenteClasificacion ejecutándose… (puede tardar 1-2 min)"):
        clf = Clasificacion(df, variable_objetivo)
        resultado = clf.ejecutar(df, variable_objetivo,
                                 plan_eda=st.session_state.get("eda_plan", {}))
        st.session_state["clasificacion_resultado"] = resultado

resultado = st.session_state["clasificacion_resultado"]
if resultado is None:
    st.warning("Presiona **▶ Ejecutar AgenteClasificacion** para comenzar.")
    st.stop()

# ── Extraer datos ─────────────────────────────────────────────────────────────
msg    = resultado["mensaje"]["contenido"]
conoc  = resultado["conocimiento"]
bench  = resultado["resultados_df"]
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
  <div class="paso-titulo">📡 Percepción del Problema</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Filas",          conoc.get("filas", 0))
c2.metric("Columnas",       conoc.get("columnas", 0))
c3.metric("Variable Y",     conoc.get("variable_objetivo", ""))
c4.metric("Clases únicas",  conoc.get("cantidad_clases", 0))
c5.metric("Ratio desbalance", f"{conoc.get('ratio_desbalance', 1):.1f}:1")

# ═════════════════════════════════════════════════════════════════════════════
# VARIABLE OBJETIVO — distribución y balance
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">VARIABLE OBJETIVO</div>
  <div class="paso-titulo">🎯 Análisis de la Variable Y</div>
</div>
""", unsafe_allow_html=True)

col_y1, col_y2 = st.columns([1, 2])
with col_y1:
    subtipo = conoc.get("subtipo", "binaria")
    st.markdown(
        '<span class="badge badge-clf">🏷️ Clasificación</span>',
        unsafe_allow_html=True)
    sub_map = {"binaria": "badge-bin", "multiclase": "badge-multi"}
    sub_lbl = {"binaria": "Binaria (2 clases)", "multiclase": "Multiclase (>2 clases)"}
    st.markdown(
        f'<span class="badge {sub_map.get(subtipo,"badge-warn")}">'
        f'{sub_lbl.get(subtipo, subtipo)}</span>',
        unsafe_allow_html=True)

    st.metric("Valores únicos", conoc.get("cantidad_clases", 0))
    st.metric("Nulos en Y",
              int(conoc.get("nulos", {}).get(conoc.get("variable_objetivo",""), 0)))

    desbalance = conoc.get("desbalanceado", False)
    ratio      = conoc.get("ratio_desbalance", 1)
    if desbalance:
        st.warning(f"⚠️ Desbalance {ratio:.1f}:1 — SMOTE activado")
    else:
        st.success(f"✅ Balanceadas ({ratio:.1f}:1)")

with col_y2:
    # Tabla de distribución
    dist_abs = conoc.get("distribucion_abs", {})
    dist_pct = conoc.get("distribucion_pct", {})
    if dist_abs:
        dist_df = pd.DataFrame({
            "Clase":          [str(k) for k in dist_abs],
            "Conteo":         list(dist_abs.values()),
            "Porcentaje (%)": [dist_pct.get(k, 0) for k in dist_abs],
        })
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # Gráfico distribución de clases
    if "distribucion_clases" in figs:
        st.pyplot(figs["distribucion_clases"], use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PASOS 2-3 — ANÁLISIS Y RAZONAMIENTO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASOS 2 Y 3</div>
  <div class="paso-titulo">🧠 Análisis y Razonamiento</div>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**🚨 Alertas** ({len(alertas)})")
    if alertas:
        for a in alertas:
            st.markdown(f'<div class="alerta-card">{a}</div>',
                        unsafe_allow_html=True)
    else:
        st.success("Sin alertas críticas.")
with col_b:
    st.markdown(f"**💡 Recomendaciones** ({len(recs)})")
    for r in recs:
        st.markdown(f'<div class="rec-card">{r}</div>',
                    unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PASO 4 — DECISIÓN
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 4</div>
  <div class="paso-titulo">🔧 Configuración de Entrenamiento</div>
</div>
""", unsafe_allow_html=True)

d1, d2, d3 = st.columns(3)
d1.metric("Métrica principal",  msg.get("metrica_principal", ""))
d2.metric("SMOTE",
          "✅ Activo" if msg.get("apply_smote") else "❌ No necesario")
d3.metric("Modelos a entrenar", len(msg.get("modelos_evaluados", [])))

modelos_str = " ".join(
    f'<span class="badge badge-model">⚙️ {m}</span>'
    for m in msg.get("modelos_evaluados", [])
)
st.markdown(modelos_str, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# RESULTADO — BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">RESULTADOS</div>
  <div class="paso-titulo">🏆 Benchmark de Modelos</div>
</div>
""", unsafe_allow_html=True)

mejor  = msg.get("mejor_modelo", "N/A")
metrica = msg.get("metrica_principal", "Accuracy")
valor  = msg.get("valor_metrica", float("nan"))

bm1, bm2, bm3 = st.columns(3)
bm1.metric("Mejor modelo",       mejor)
bm2.metric(f"{metrica} (mejor)",
           f"{valor:.4f}" if valor == valor else "N/A")
bm3.metric("Modelos evaluados",  len(bench) if bench is not None else 0)

if bench is not None and not bench.empty:
    # Resaltar fila ganadora
    def resaltar_mejor(row):
        col_m = metrica if metrica in bench.columns else "Accuracy"
        try:
            max_val = bench[col_m].max()
            if row[col_m] == max_val:
                return ["background-color:#dcfce7; font-weight:700"] * len(row)
        except Exception:
            pass
        return [""] * len(row)

    st.dataframe(
        bench.style.apply(resaltar_mejor, axis=1).format(
            {c: "{:.4f}" for c in bench.columns if c != "Modelo"},
            na_rep="N/A"),
        use_container_width=True, hide_index=True
    )

# Observaciones del benchmark
obs_clf = observaciones_clasificacion(
    bench, msg.get("mejor_modelo",""), msg.get("desbalanceado", False),
    msg.get("metrica_principal","Accuracy"), msg.get("valor_metrica", 0))
if obs_clf:
    st.markdown("**Observaciones:**")
    for o in obs_clf:
        st.markdown(f"• {o}")

# Gráfico comparativo de barras
if "benchmark_barras" in figs:
    st.pyplot(figs["benchmark_barras"], use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# MATRIZ DE CONFUSIÓN
# ═════════════════════════════════════════════════════════════════════════════
if "matriz_confusion" in figs:
    st.markdown("---")
    st.markdown("""
    <div class="paso-card">
      <div class="paso-num">EVALUACIÓN</div>
      <div class="paso-titulo">🔲 Matriz de Confusión — Mejor Modelo</div>
    </div>
    """, unsafe_allow_html=True)

    col_mc, col_mc2 = st.columns([1, 2])
    with col_mc:
        st.markdown(
            f'<span class="badge badge-model">🏆 {mejor}</span>',
            unsafe_allow_html=True)
        st.caption(
            f"Evaluado con {metrica} = {valor:.4f}" if valor==valor else "")
    with col_mc2:
        st.pyplot(figs["matriz_confusion"], use_container_width=True)

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
    st.metric("Agente",            resultado["mensaje"].get("agente",""))
    st.metric("Timestamp",         ts)
    st.metric("Figuras generadas", len(figs))
    st.metric("Errores en etapas", len(errores))
    if errores:
        with st.expander("⚠️ Ver errores"):
            for e in errores:
                st.error(f"**{e.get('paso','')}**: {e.get('error','')}")
with col_m2:
    with st.expander("📋 Mensaje completo (JSON)", expanded=False):
        st.json(msg)

# Resumen ejecutivo
st.markdown("**📊 Resumen ejecutivo:**")
cs = st.columns(4)
for col_c, val, lbl in [
    (cs[0], mejor,                "Mejor modelo"),
    (cs[1], f"{valor:.4f}" if valor==valor else "N/A", f"{metrica} ganador"),
    (cs[2], conoc.get("subtipo",""), "Subtipo"),
    (cs[3], "Sí" if msg.get("apply_smote") else "No", "SMOTE aplicado"),
]:
    col_c.markdown(
        f'<div class="metric-mini"><div class="val">{val}</div>'
        f'<div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True)
