# =============================================================================
# pages/4_Regresion.py — AgenteRegresion
# =============================================================================

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.regresion import Regresion
from shared_utils import obtener_df, observaciones_regresion

try:
    import plotly  # noqa: F401  (solo para verificar disponibilidad)
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

st.set_page_config(page_title="AgenteRegresion",
                   page_icon="📈", layout="wide")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.paso-card { background:#f8fafc; border-left:4px solid #f59e0b;
             border-radius:0 8px 8px 0; padding:12px 18px; margin:8px 0; }
.paso-num  { font-size:11px; font-weight:700; color:#f59e0b;
             letter-spacing:1.5px; text-transform:uppercase; margin-bottom:2px; }
.paso-titulo { font-size:16px; font-weight:700; color:#1e293b; margin:0; }
.badge { display:inline-block; padding:4px 12px; border-radius:20px;
         font-size:13px; font-weight:600; margin:4px 4px 4px 0; }
.badge-reg   { background:#dcfce7; color:#166534; }
.badge-ok    { background:#dcfce7; color:#166534; }
.badge-warn  { background:#fef9c3; color:#854d0e; }
.badge-model { background:#fef3c7; color:#92400e; }
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
st.markdown("## 📈 Agente Regresión — Aprendizaje Supervisado")
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
st.sidebar.info(
    "🤖 La variable objetivo se infiere **automáticamente** del dataset "
    "usando la misma lógica del AgenteEDA."
)

# Inferencia automática de variable objetivo para regresión
# Prioridad: 1) lo que ya infirió el AgenteEDA, 2) inferir localmente
_var_eda  = st.session_state.get("eda_variable_objetivo")
_tipo_eda = st.session_state.get("eda_tipo_problema", "")

if _var_eda and _var_eda in df.columns:
    variable_objetivo = _var_eda
else:
    # Inferir localmente usando la misma heurística del AgenteEDA
    from backend.eda_base import analisisEDA as _AgenteEDA
    _eda_tmp = _AgenteEDA()
    _eda_tmp.cargar_dataframe(df)
    variable_objetivo, _just = _eda_tmp._inferir_variable_objetivo_automatica()
    if not variable_objetivo:
        # Fallback: última columna numérica continua
        _num_cands = [
            c for c in df.select_dtypes(include="number").columns
            if df[c].nunique() / max(len(df), 1) >= 0.05
        ]
        variable_objetivo = _num_cands[-1] if _num_cands else df.columns[-1]

ejecutar = st.sidebar.button("▶ Ejecutar AgenteRegresion",
                              type="primary", use_container_width=True)

if "regresion_resultado" not in st.session_state:
    st.session_state["regresion_resultado"] = None

if ejecutar:
    with st.spinner("🤖 AgenteRegresion ejecutándose… (puede tardar 1-2 min)"):
        reg = Regresion(df, variable_objetivo)
        resultado = reg.ejecutar(df, variable_objetivo,
                                 plan_eda=st.session_state.get("eda_plan", {}))
        st.session_state["regresion_resultado"] = resultado

resultado = st.session_state["regresion_resultado"]
if resultado is None:
    st.warning("Presiona **▶ Ejecutar AgenteRegresion** para comenzar.")
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
c1.metric("Filas",       conoc.get("filas", 0))
c2.metric("Columnas",    conoc.get("columnas", 0))
c3.metric("Variable Y",  conoc.get("variable_objetivo", ""))
c4.metric("Media Y",     conoc.get("media_y", 0))
c5.metric("Skewness Y",  conoc.get("skewness_y", 0))

# Pre-eliminadas
pre_elim = conoc.get("pre_eliminadas", [])
if pre_elim:
    st.markdown(
        f'<div class="alerta-card">🗑️ <b>Columnas eliminadas automáticamente: </b>'
        f'{", ".join(f"<code>{c}</code>" for c in pre_elim)}</div>',
        unsafe_allow_html=True
    )

# ═════════════════════════════════════════════════════════════════════════════
# VARIABLE OBJETIVO — distribución
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
    subtipo = conoc.get("subtipo", "regresion_normal")
    st.markdown('<span class="badge badge-reg">📈 Regresión</span>',
                unsafe_allow_html=True)
    badge_sub = ("badge-warn" if "sesgada" in subtipo else "badge-ok")
    label_sub = ("Distribución sesgada" if "sesgada" in subtipo
                 else "Distribución normal")
    st.markdown(
        f'<span class="badge {badge_sub}">{label_sub}</span>',
        unsafe_allow_html=True)

    stats_y = {
        "Media":    conoc.get("media_y", 0),
        "Mediana":  conoc.get("mediana_y", 0),
        "Std":      conoc.get("std_y", 0),
        "Mínimo":   conoc.get("min_y", 0),
        "Máximo":   conoc.get("max_y", 0),
        "Skewness": conoc.get("skewness_y", 0),
    }
    st.dataframe(pd.DataFrame(stats_y.items(),
                               columns=["Estadística", "Valor"]),
                 use_container_width=True, hide_index=True)

    if conoc.get("log_transform_aplicado"):
        st.warning("🔄 Se aplicó log(y+1) por distribución sesgada.")

with col_y2:
    if "distribucion_objetivo" in figs:
        st.pyplot(figs["distribucion_objetivo"], use_container_width=True)

    # Top correlaciones con objetivo
    corr_y = conoc.get("correlacion_con_objetivo", {})
    if corr_y:
        top5 = dict(list(corr_y.items())[:5])
        corr_df = pd.DataFrame({
            "Variable":    list(top5.keys()),
            "Correlación": [round(v, 4) for v in top5.values()],
        })
        st.markdown("**Top 5 variables más correlacionadas con Y:**")
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

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
d1.metric("Estrategia",      msg.get("estrategia_busqueda", "GridSearchCV"))
d2.metric("Búsqueda genética",
          "✅ Activa" if msg.get("usar_genetico") else "❌ Dataset pequeño")
d3.metric("Modelos a entrenar", len(msg.get("modelos_evaluados", [])))

modelos_str = " ".join(
    f'<span class="badge badge-model">⚙️ {m}</span>'
    for m in msg.get("modelos_evaluados", [])
)
st.markdown(modelos_str, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# RESULTADOS — BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">RESULTADOS</div>
  <div class="paso-titulo">🏆 Benchmark de Modelos</div>
</div>
""", unsafe_allow_html=True)

mejor    = msg.get("mejor_modelo", "N/A")
rmse_m   = msg.get("mejor_rmse", float("nan"))
r2_m     = msg.get("mejor_r2", float("nan"))
rmse_g   = msg.get("mejores_params_grid", {})
rmse_gen = conoc.get("rmse_gen")

bm1, bm2, bm3, bm4 = st.columns(4)
bm1.metric("Mejor modelo",     mejor)
bm2.metric("RMSE (mejor)",
           f"{rmse_m:.4f}" if rmse_m == rmse_m else "N/A")
bm3.metric("R² (mejor)",
           f"{r2_m:.4f}" if r2_m == r2_m else "N/A")
bm4.metric("RMSE Genético",
           f"{rmse_gen:.4f}" if rmse_gen else "N/A (no ejecutado)")

# Mostrar variable objetivo inferida
st.sidebar.success(f"🤖 Variable objetivo inferida: **{variable_objetivo}**")

if bench is not None and not bench.empty:
    # Observaciones del benchmark
    obs_reg = observaciones_regresion(
        bench, msg.get("mejor_modelo",""),
        msg.get("mejor_rmse", float("nan")),
        msg.get("mejor_r2", float("nan")))
    if obs_reg:
        st.markdown("**Observaciones:**")
        for o in obs_reg:
            st.markdown(f"• {o}")
        st.markdown("")

    def color_rmse(val):
        try:
            v  = float(val)
            mn = bench["RMSE"].min()
            if v == mn: return "background-color:#dcfce7;font-weight:700"
        except Exception:
            pass
        return ""

    def color_r2(val):
        try:
            v = float(val)
            if v >= 0.9: return "background-color:#dcfce7"
            if v >= 0.7: return "background-color:#fef9c3"
            return "background-color:#fee2e2"
        except Exception:
            return ""

    st.dataframe(
        bench.style
             .applymap(color_rmse, subset=["RMSE"])
             .applymap(color_r2,   subset=["R2"])
             .format({c: "{:.4f}" for c in bench.columns if c != "Modelo"},
                     na_rep="N/A"),
        use_container_width=True, hide_index=True
    )

    with st.expander("ℹ️ Interpretación de métricas"):
        st.markdown("""
        | Métrica | Mejor cuando | Referencia |
        |---------|-------------|-----------|
        | **RMSE** | Menor es mejor | Verde = mínimo del benchmark |
        | **R²** | Mayor (máx 1.0) | Verde ≥ 0.9 · Amarillo ≥ 0.7 · Rojo < 0.7 |
        | **MAE** | Menor es mejor | Error absoluto promedio |
        | **MAPE** | Menor es mejor | % de error relativo |
        """)

# ═════════════════════════════════════════════════════════════════════════════
# GRÁFICOS COMPARATIVOS (Plotly)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">VISUALIZACIONES</div>
  <div class="paso-titulo">📊 Gráficos Comparativos</div>
</div>
""", unsafe_allow_html=True)

tab_rmse, tab_r2, tab_pvr = st.tabs(
    ["📉 RMSE Comparativo", "📐 R² por Modelo", "🔵 Predicho vs Real"]
)

with tab_rmse:
    if "rmse_comparativo" in figs and _PLOTLY_OK:
        st.plotly_chart(figs["rmse_comparativo"],
                        use_container_width=True)
    elif not _PLOTLY_OK:
        st.error("⚠️ Plotly no instalado. Ejecuta: `pip install plotly`")
        st.caption(
            "🔴 GridSearch = mejor configuración por búsqueda exhaustiva. "
            "🔵 Genético = mejor configuración por algoritmo genético. "
            "⬛ Base = hiperparámetros por defecto."
        )
    else:
        st.info("Gráfico RMSE no disponible.")

    # Mejores parámetros
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        params_grid = msg.get("mejores_params_grid", {})
        if params_grid:
            st.markdown("**🔴 Mejores parámetros — GridSearchCV:**")
            st.json(params_grid)
        else:
            st.info("GridSearchCV: sin parámetros adicionales.")
    with col_p2:
        params_gen = msg.get("mejores_params_gen", {})
        if params_gen:
            st.markdown("**🔵 Mejores parámetros — GASearchCV (Genético):**")
            st.json(params_gen)
        elif msg.get("usar_genetico"):
            st.warning("Búsqueda genética activa pero sin resultados. "
                       "Instalar: `pip install sklearn-genetic-opt`")
        else:
            st.info("Búsqueda genética no activada (<500 filas).")

with tab_r2:
    if "r2_comparativo" in figs and _PLOTLY_OK:
        st.plotly_chart(figs["r2_comparativo"],
                        use_container_width=True)
    elif not _PLOTLY_OK:
        st.error("⚠️ Plotly no instalado. Ejecuta: `pip install plotly`")
    else:
        st.info("Gráfico R² no disponible.")

with tab_pvr:
    if "pred_vs_real" in figs:
        col_pvr1, col_pvr2 = st.columns([2, 1])
        with col_pvr1:
            st.pyplot(figs["pred_vs_real"], use_container_width=True)
        with col_pvr2:
            st.markdown(f"**Modelo:** {mejor}")
            st.metric("RMSE", f"{rmse_m:.4f}" if rmse_m == rmse_m else "N/A")
            st.metric("R²",   f"{r2_m:.4f}"   if r2_m   == r2_m   else "N/A")
            if conoc.get("log_transform_aplicado"):
                st.info("Los valores están en escala original "
                        "(se revirtió log(y+1)).")
    else:
        st.info("Gráfico predicho vs real no disponible.")

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
    (cs[0], mejor,                                     "Mejor modelo"),
    (cs[1], f"{rmse_m:.4f}" if rmse_m==rmse_m else "N/A", "RMSE ganador"),
    (cs[2], f"{r2_m:.4f}"   if r2_m  ==r2_m   else "N/A", "R² ganador"),
    (cs[3], msg.get("estrategia_busqueda",""),          "Estrategia"),
]:
    col_c.markdown(
        f'<div class="metric-mini"><div class="val">{val}</div>'
        f'<div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True)
