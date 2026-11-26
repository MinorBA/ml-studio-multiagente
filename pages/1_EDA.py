# =============================================================================
# pages/1_EDA.py — AgenteEDA: Análisis Exploratorio Inteligente
# =============================================================================

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.eda_base import analisisEDA
from shared_utils import (cargar_csv_en_eda,
                           observaciones_histogramas, observaciones_boxplots,
                           observaciones_correlacion)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AgenteEDA", page_icon="🔍", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Tarjeta de paso del agente ─────────────────────────────── */
.paso-card {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    margin: 8px 0;
}
.paso-num {
    font-size: 11px;
    font-weight: 700;
    color: #3b82f6;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.paso-titulo {
    font-size: 16px;
    font-weight: 700;
    color: #1e293b;
    margin: 0;
}

/* ── Badge de tipo de problema ──────────────────────────────── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 4px 4px 4px 0;
}
.badge-clf  { background:#dbeafe; color:#1d4ed8; }
.badge-reg  { background:#dcfce7; color:#166534; }
.badge-ns   { background:#f3e8ff; color:#7e22ce; }
.badge-bin  { background:#fef9c3; color:#854d0e; }
.badge-multi{ background:#ffedd5; color:#9a3412; }
.badge-ok   { background:#dcfce7; color:#166534; }
.badge-warn { background:#fef9c3; color:#854d0e; }
.badge-err  { background:#fee2e2; color:#991b1b; }

/* ── Alerta crítica ──────────────────────────────────────────── */
.alerta-card {
    background: #fff7ed;
    border: 1px solid #fb923c;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    color: #7c2d12;
}
/* ── Recomendación ───────────────────────────────────────────── */
.rec-card {
    background: #f0fdf4;
    border: 1px solid #4ade80;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    color: #14532d;
}
/* ── Item de plan ───────────────────────────────────────────── */
.plan-item {
    background: #f1f5f9;
    border-radius: 6px;
    padding: 6px 12px;
    margin: 4px 0;
    font-size: 13px;
    font-family: monospace;
    color: #334155;
}
.plan-item span {
    font-weight: 700;
    color: #0f172a;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENCABEZADO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🔍 Agente EDA — Análisis Exploratorio Inteligente")
st.caption("Ejecuta los 7 pasos del agente: percepción → análisis → razonamiento "
           "→ decisión → inferencia → comunicación → ejecución")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
df = cargar_csv_en_eda()

if df is None:
    st.info("📂 Suba un archivo CSV desde la barra lateral para activar el agente.")
    st.stop()

# ── Reset de sesión al cambiar de CSV ─────────────────────────────────────
_csv_actual = st.session_state.get("nombre_csv", "")
if st.session_state.get("eda_csv_previo", "") != _csv_actual:
    for _k in ["eda_resultado","clustering_resultado","clasificacion_resultado",
               "regresion_resultado","coordinador_resultado",
               "eda_plan","eda_tipo_problema","eda_variable_objetivo",
               "mas_conclusiones","mas_mejor_global"]:
        st.session_state.pop(_k, None)
    st.session_state["eda_csv_previo"] = _csv_actual

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONFIGURACIÓN DEL AGENTE
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuración del Agente")
st.sidebar.info(
    "🤖 La variable objetivo se infiere **automáticamente** "
    "del dataset. El agente analizará el nombre y tipo de cada "
    "columna para identificarla."
)

ejecutar = st.sidebar.button("▶ Ejecutar AgenteEDA", type="primary",
                              use_container_width=True)

# Cache del resultado en session_state
if "eda_resultado" not in st.session_state:
    st.session_state["eda_resultado"] = None

if ejecutar:
    with st.spinner("🤖 Agente EDA ejecutándose…"):
        agente = analisisEDA()
        # Sin variable_objetivo → el agente la infiere automáticamente
        resultado = agente.ejecutar(df)
        st.session_state["eda_resultado"] = resultado
        st.session_state["eda_plan"] = (
            resultado["mensaje"]["contenido"]["plan_preprocesamiento"]
        )
        st.session_state["eda_tipo_problema"] = (
            resultado["mensaje"]["contenido"]["tipo_problema"]
        )
        st.session_state["eda_variable_objetivo"] = (
            resultado["conocimiento"].get("variable_objetivo")
        )

resultado = st.session_state["eda_resultado"]

if resultado is None:
    st.warning("Presiona **▶ Ejecutar AgenteEDA** en la barra lateral para comenzar.")
    st.stop()

# ─── Extraer datos del resultado ──────────────────────────────────────────
msg        = resultado["mensaje"]["contenido"]
conoc      = resultado["conocimiento"]
figuras    = resultado["figuras"]
estadist   = conoc.get("estadisticas", {})
plan       = conoc.get("plan_preprocesamiento", {})
alertas    = conoc.get("alertas", [])
recs       = conoc.get("recomendaciones", [])
errores    = msg.get("errores_etapas", [])

# ═════════════════════════════════════════════════════════════════════════════
# PASO 1 — PERCEPCIÓN
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 1</div>
  <div class="paso-titulo">📡 Percepción del Dataset</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Filas",      conoc.get("filas", 0))
c2.metric("Columnas",   conoc.get("columnas", 0))
c3.metric("Duplicados", conoc.get("duplicados", 0))
c4.metric("Densidad",   f"{conoc.get('densidad_datos', 100)}%")
c5.metric("Nulos total",
          int(sum(conoc.get("nulos", {}).values())))

col_p1a, col_p1b = st.columns(2)

# ── Panel de limpieza automática aplicada ─────────────────────────────────
pre_elim = msg.get("pre_eliminadas", [])
pre_imp  = msg.get("pre_imputadas", [])
if pre_elim or pre_imp:
    with st.expander("🧹 Limpieza automática aplicada antes del análisis",
                     expanded=True):
        if pre_elim:
            elim_html = ", ".join(f"<code>{c}</code>" for c in pre_elim)
            st.markdown(
                f'''<div style="background:#fff7ed;border:1px solid #fb923c;
                border-radius:8px;padding:10px 14px;font-size:14px;color:#7c2d12;">
                🗑️ <b>Columnas eliminadas automáticamente ({len(pre_elim)}):</b>
                {elim_html}<br>
                <small>Razón: columnas 100% vacías o identificadores sin valor predictivo.</small>
                </div>''',
                unsafe_allow_html=True
            )
        if pre_imp:
            st.info(f"📊 **Imputaciones aplicadas:** " +
                    " | ".join(pre_imp[:5]) +
                    ("…" if len(pre_imp) > 5 else ""))

with col_p1a:
    with st.expander("📋 Tipos de variables", expanded=True):
        num_cols = conoc.get("columnas_numericas", [])
        cat_cols = conoc.get("columnas_categoricas", [])
        tipos_df = pd.DataFrame(
            [(c, "🔢 Numérica") for c in num_cols] +
            [(c, "🔤 Categórica") for c in cat_cols],
            columns=["Columna", "Tipo"]
        )
        st.dataframe(tipos_df, use_container_width=True, hide_index=True)

with col_p1b:
    with st.expander("🪪 Columnas detectadas como posibles IDs", expanded=True):
        ids = conoc.get("columnas_posibles_id", [])
        if ids:
            for col_id in ids:
                st.markdown(
                    f'<div class="alerta-card">🪪 <b>{col_id}</b> — '
                    f'más del 95% de valores únicos. '
                    f'Se recomienda excluir del modelo.</div>',
                    unsafe_allow_html=True
                )
        else:
            st.success("No se detectaron columnas identificadoras.")

# ═════════════════════════════════════════════════════════════════════════════
# PASO 2 — ANÁLISIS ESTADÍSTICO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 2</div>
  <div class="paso-titulo">📊 Análisis Estadístico</div>
</div>
""", unsafe_allow_html=True)

tab_desc, tab_nulos, tab_outl, tab_corr = st.tabs([
    "📈 Descriptivas", "🕳️ Nulos", "🎯 Outliers", "🔗 Correlaciones"
])

# ── Tab: Estadísticas descriptivas ────────────────────────────────────────
with tab_desc:
    describe_df = estadist.get("describe")
    if describe_df is not None:
        st.dataframe(describe_df.style.format(precision=3),
                     use_container_width=True)

    # Skewness con interpretación
    skew = estadist.get("skewness", {})
    if skew:
        st.markdown("**Asimetría (Skewness) por variable:**")
        skew_rows = []
        for col, val in skew.items():
            if not isinstance(val, (int, float)):
                continue
            if abs(val) < 0.5:
                interp = "Distribución simétrica ✅"
            elif abs(val) < 1:
                interp = "Asimetría moderada ⚠️"
            else:
                dir_ = "derecha (cola larga positiva)" if val > 0 else "izquierda (cola larga negativa)"
                interp = f"Sesgo pronunciado hacia la {dir_} 🚨"
            skew_rows.append({"Variable": col,
                               "Skewness": round(val, 4),
                               "Interpretación": interp})
        if skew_rows:
            st.dataframe(pd.DataFrame(skew_rows),
                         use_container_width=True, hide_index=True)

# ── Tab: Nulos ─────────────────────────────────────────────────────────────
with tab_nulos:
    nulos     = estadist.get("nulos", {})
    pct_nulos = estadist.get("pct_nulos", {})
    if nulos:
        nulos_df = pd.DataFrame([
            {"Columna": c,
             "Nulos": nulos.get(c, 0),
             "Porcentaje (%)": pct_nulos.get(c, 0),
             "Estado": ("🚨 Crítico (>40%)" if pct_nulos.get(c, 0) > 40
                        else "⚠️ Moderado" if pct_nulos.get(c, 0) > 0
                        else "✅ Sin nulos")}
            for c in nulos
        ])
        # Colorear según severidad
        def color_nulos(val):
            if "Crítico" in str(val):   return "background-color:#fee2e2"
            if "Moderado" in str(val):  return "background-color:#fef9c3"
            if "Sin nulos" in str(val): return "background-color:#dcfce7"
            return ""
        st.dataframe(
            nulos_df.style.applymap(color_nulos, subset=["Estado"]),
            use_container_width=True, hide_index=True
        )

# ── Tab: Outliers ──────────────────────────────────────────────────────────
with tab_outl:
    outliers_iqr = conoc.get("outliers_iqr", {})
    outliers_z   = conoc.get("outliers_zscore", {})
    if outliers_iqr:
        out_rows = []
        for col, info in outliers_iqr.items():
            if not isinstance(info, dict) or "error" in info:
                continue
            z_info   = outliers_z.get(col, {})
            n_z      = z_info.get("n_outliers", 0) if isinstance(z_info, dict) else 0
            pct      = info.get("pct_outliers", 0)
            gravedad = ("🚨 Alto (>10%)" if pct > 10
                        else "⚠️ Moderado (5-10%)" if pct > 5
                        else "✅ Bajo (<5%)")
            out_rows.append({
                "Variable":        col,
                "IQR — n outliers": info.get("n_outliers", 0),
                "IQR — %":         pct,
                "Z-Score — n":     n_z,
                "Lím. Inferior":   info.get("limite_inferior", ""),
                "Lím. Superior":   info.get("limite_superior", ""),
                "Gravedad":        gravedad,
            })
        if out_rows:
            st.dataframe(pd.DataFrame(out_rows),
                         use_container_width=True, hide_index=True)
    else:
        st.info("No se calcularon outliers (sin columnas numéricas).")

# ── Tab: Correlaciones ─────────────────────────────────────────────────────
with tab_corr:
    col_ca, col_cb = st.columns(2)
    with col_ca:
        st.markdown("**Pearson**")
        if "correlacion_pearson" in figuras:
            st.pyplot(figuras["correlacion_pearson"], use_container_width=True)
    with col_cb:
        st.markdown("**Spearman**")
        if "correlacion_spearman" in figuras:
            st.pyplot(figuras["correlacion_spearman"], use_container_width=True)
    var_obj = conoc.get("variable_objetivo")
    obs_corr = observaciones_correlacion(df, var_obj)
    if obs_corr:
        st.markdown("**Observaciones:**")
        for o in obs_corr:
            st.markdown(f"• {o}")

# ═════════════════════════════════════════════════════════════════════════════
# PASO 3 — RAZONAMIENTO: ALERTAS Y RECOMENDACIONES
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 3</div>
  <div class="paso-titulo">🧠 Razonamiento — Alertas y Recomendaciones</div>
</div>
""", unsafe_allow_html=True)

col_a3a, col_a3b = st.columns(2)

with col_a3a:
    st.markdown(f"**🚨 Alertas críticas** ({len(alertas)})")
    if alertas:
        for a in alertas:
            st.markdown(
                f'<div class="alerta-card">{a}</div>',
                unsafe_allow_html=True
            )
    else:
        st.success("Sin alertas críticas detectadas.")

with col_a3b:
    st.markdown(f"**💡 Recomendaciones** ({len(recs)})")
    if recs:
        for r in recs:
            st.markdown(
                f'<div class="rec-card">{r}</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("Sin recomendaciones adicionales.")

# ═════════════════════════════════════════════════════════════════════════════
# DISTRIBUCIÓN DE VARIABLES (histogramas y boxplots)
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">VISUALIZACIÓN</div>
  <div class="paso-titulo">📉 Distribución de Variables</div>
</div>
""", unsafe_allow_html=True)

col_v1, col_v2 = st.columns(2)
with col_v1:
    st.markdown("**Histogramas**")
    if "histogramas" in figuras:
        st.pyplot(figuras["histogramas"], use_container_width=True)
        obs_hist = observaciones_histogramas(df)
        if obs_hist:
            st.markdown("**Observaciones:**")
            for o in obs_hist:
                st.markdown(f"• {o}")

with col_v2:
    st.markdown("**Boxplots**")
    if "boxplots" in figuras:
        st.pyplot(figuras["boxplots"], use_container_width=True)
        obs_box = observaciones_boxplots(df)
        if obs_box:
            st.markdown("**Observaciones:**")
            for o in obs_box:
                st.markdown(f"• {o}")

# ═════════════════════════════════════════════════════════════════════════════
# VARIABLE OBJETIVO
# ═════════════════════════════════════════════════════════════════════════════
variable_objetivo = conoc.get('variable_objetivo')
if variable_objetivo:
    st.markdown("---")
    st.markdown("""
    <div class="paso-card">
      <div class="paso-num">VARIABLE OBJETIVO</div>
      <div class="paso-titulo">🎯 Análisis de la Variable Y</div>
    </div>
    """, unsafe_allow_html=True)

    y_serie = df[variable_objetivo]

    # Justificación de la inferencia automática
    justificacion = conoc.get("justificacion_objetivo", "")
    if justificacion:
        st.markdown(
            f'''<div style="background:#eff6ff;border-left:4px solid #3b82f6;
            border-radius:0 8px 8px 0;padding:10px 16px;margin-bottom:12px;
            font-size:14px;color:#1e40af;">
            🤖 <b>Inferencia automática:</b> {justificacion}
            </div>''',
            unsafe_allow_html=True
        )

    col_y1, col_y2 = st.columns([1, 2])

    with col_y1:
        tipo_prob = conoc.get("tipo_problema", "no_inferido")
        subtipo   = conoc.get("subtipo", "")

        tipo_map = {
            "clasificacion": ("badge-clf",  "🏷️ Clasificación"),
            "regresion":     ("badge-reg",  "📈 Regresión"),
            "no_supervisado":("badge-ns",   "🔵 No Supervisado"),
        }
        badge_cls, badge_label = tipo_map.get(tipo_prob,
                                               ("badge-warn", "⚠️ Desconocido"))
        st.markdown(
            f'<span class="badge {badge_cls}">{badge_label}</span>',
            unsafe_allow_html=True
        )
        sub_map = {
            "binaria":           ("badge-bin",   "Binaria (2 clases)"),
            "multiclase":        ("badge-multi", "Multiclase (>2 clases)"),
            "regresion_normal":  ("badge-ok",    "Distribución normal"),
            "regresion_sesgada": ("badge-warn",  "Distribución sesgada"),
            "clustering":        ("badge-ns",    "Sin objetivo"),
        }
        if subtipo in sub_map:
            s_cls, s_lbl = sub_map[subtipo]
            st.markdown(
                f'<span class="badge {s_cls}">{s_lbl}</span>',
                unsafe_allow_html=True
            )

        st.metric("Valores únicos", y_serie.nunique())
        st.metric("Nulos en Y",     int(y_serie.isnull().sum()))

    with col_y2:
        if tipo_prob == "clasificacion":
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            dist     = y_serie.value_counts(normalize=True).mul(100).round(1)
            dist_abs = y_serie.value_counts()

            # Detección de desbalance
            pct_max = dist.max()
            pct_min = dist.min()
            ratio   = pct_max / max(pct_min, 0.001)

            if ratio > 3:
                st.warning(
                    f"⚠️ **Desbalance detectado** — ratio {round(ratio,1)}:1. "
                    f"Clase mayoritaria: {round(pct_max,1)}%, "
                    f"minoritaria: {round(pct_min,1)}%. "
                    f"Se recomienda SMOTE."
                )
            else:
                st.success(
                    f"✅ Clases relativamente **balanceadas** "
                    f"(ratio {round(ratio,1)}:1)"
                )

            # Tabla de distribución
            dist_df = pd.DataFrame({
                "Clase":          dist_abs.index.astype(str),
                "Conteo":         dist_abs.values,
                "Porcentaje (%)": dist.values,
            })
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

            # Gráfico de barras de distribución de clases
            paleta = sns.color_palette("pastel", n_colors=len(dist_abs))
            fig_cls, ax_cls = plt.subplots(figsize=(5, 3), dpi=100)
            bars = ax_cls.bar(
                dist_abs.index.astype(str),
                dist_abs.values,
                color=paleta,
                edgecolor="white",
                linewidth=0.8,
            )
            ax_cls.set_title(
                f"Distribución de la Clase: {variable_objetivo}",
                fontsize=11, pad=8
            )
            ax_cls.set_xlabel(variable_objetivo, fontsize=9)
            ax_cls.set_ylabel("Frecuencia", fontsize=9)
            ax_cls.set_xticks(range(len(dist_abs)))
            ax_cls.set_xticklabels(dist_abs.index.astype(str), fontsize=9)
            ax_cls.grid(axis="y", linestyle="--", alpha=0.4)
            ax_cls.spines[["top","right"]].set_visible(False)
            for bar, pct in zip(bars, dist.values):
                ax_cls.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(dist_abs.values) * 0.01,
                    f"{pct}%",
                    ha="center", va="bottom", fontsize=8, color="#334155"
                )
            plt.tight_layout()
            st.pyplot(fig_cls, use_container_width=True)
            plt.close(fig_cls)

        elif tipo_prob == "regresion":
            stats_y = {
                "Media":    round(float(y_serie.mean()), 4),
                "Mediana":  round(float(y_serie.median()), 4),
                "Std":      round(float(y_serie.std()), 4),
                "Mínimo":   round(float(y_serie.min()), 4),
                "Máximo":   round(float(y_serie.max()), 4),
                "Skewness": round(float(y_serie.skew()), 4),
            }
            st.dataframe(
                pd.DataFrame(stats_y.items(),
                             columns=["Estadística", "Valor"]),
                use_container_width=True, hide_index=True
            )
            if abs(stats_y["Skewness"]) > 1:
                st.warning(
                    f"⚠️ Variable objetivo sesgada "
                    f"(skewness={stats_y['Skewness']}). "
                    f"Considerar transformación log(Y+1)."
                )

# ═════════════════════════════════════════════════════════════════════════════
# PASO 4 — PLAN DE PREPROCESAMIENTO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 4</div>
  <div class="paso-titulo">🔧 Plan de Preprocesamiento</div>
</div>
""", unsafe_allow_html=True)

iconos = {
    "eliminar_columnas":   ("🗑️", "Eliminar columnas"),
    "eliminar_duplicados": ("♻️", "Eliminar duplicados"),
    "imputar_mediana":     ("📊", "Imputar con mediana"),
    "imputar_moda":        ("📝", "Imputar con moda"),
    "escalar_standard":    ("📐", "StandardScaler"),
    "escalar_robust":      ("🛡️", "RobustScaler (outliers)"),
    "codificar_onehot":    ("🔠", "OneHotEncoder"),
    "codificar_label":     ("🔡", "LabelEncoder"),
    "transformar_log":     ("📉", "Transformación logarítmica"),
}

cols_plan = st.columns(3)
plan_items = [
    (k, v) for k, v in plan.items()
    if (v is True) or (isinstance(v, list) and len(v) > 0)
]

if plan_items:
    for i, (clave, valor) in enumerate(plan_items):
        icono, etiqueta = iconos.get(clave, ("⚙️", clave))
        cols_plan[i % 3].markdown(
            f'<div class="plan-item"><span>{icono} {etiqueta}:</span> '
            f'{"Sí" if valor is True else ", ".join(str(v) for v in valor)}'
            f'</div>',
            unsafe_allow_html=True
        )
else:
    st.info("El dataset no requiere preprocesamiento adicional.")

# ═════════════════════════════════════════════════════════════════════════════
# PASO 5 — INFERENCIA DEL TIPO DE PROBLEMA
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 5</div>
  <div class="paso-titulo">🧩 Inferencia del Tipo de Problema</div>
</div>
""", unsafe_allow_html=True)

tipo_prob = conoc.get("tipo_problema", "no_inferido")
subtipo   = conoc.get("subtipo", "")

descripciones = {
    ("clasificacion", "binaria"):
        "Clasificación Binaria — la variable objetivo tiene 2 clases. "
        "Se usarán modelos como Logistic Regression, Random Forest, XGBoost, SVM.",
    ("clasificacion", "multiclase"):
        "Clasificación Multiclase — la variable objetivo tiene más de 2 clases. "
        "Se usará AUC macro OvR y métricas F1 ponderadas.",
    ("regresion", "regresion_normal"):
        "Regresión — variable objetivo numérica continua con distribución normal. "
        "Se usarán modelos lineales, árboles y boosting.",
    ("regresion", "regresion_sesgada"):
        "Regresión con objetivo sesgado — se recomienda aplicar log(Y+1) "
        "antes del entrenamiento (como en predicción de defectos de software).",
    ("no_supervisado", "clustering"):
        "Problema no supervisado — sin variable objetivo. "
        "Se activará el AgenteClustering para descubrir grupos en los datos.",
}

desc_key = (tipo_prob, subtipo)
desc_txt  = descripciones.get(desc_key,
    f"Tipo: {tipo_prob} | Subtipo: {subtipo}")

if tipo_prob == "clasificacion":
    st.info(f"🏷️ {desc_txt}")
elif tipo_prob == "regresion":
    st.success(f"📈 {desc_txt}")
elif tipo_prob == "no_supervisado":
    st.markdown(
        f'<div style="background:#f3e8ff;border-radius:8px;padding:12px;'
        f'color:#5b21b6;border:1px solid #c4b5fd;">🔵 {desc_txt}</div>',
        unsafe_allow_html=True
    )
else:
    st.warning(f"⚠️ Tipo de problema no detectado. "
               f"Seleccione una variable objetivo en el panel lateral.")

# ═════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL — MENSAJE MAS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 6 — COMUNICACIÓN</div>
  <div class="paso-titulo">📨 Mensaje al AgenteCoordinador</div>
</div>
""", unsafe_allow_html=True)

col_r1, col_r2 = st.columns([1, 2])
with col_r1:
    ts = resultado["mensaje"].get("timestamp", "")[:19].replace("T", " ")
    st.metric("Agente",     resultado["mensaje"].get("agente", ""))
    st.metric("Timestamp",  ts)
    st.metric("Figuras generadas", len(figuras))
    st.metric("Errores en etapas", len(errores))

    if errores:
        with st.expander("⚠️ Ver errores de etapas"):
            for err in errores:
                st.error(f"**{err.get('paso','')}**: {err.get('error','')}")

with col_r2:
    with st.expander("📋 Ver mensaje completo (JSON)", expanded=False):
        # Versión serializable del mensaje (sin DataFrames)
        msg_serial = resultado["mensaje"].copy()
        st.json(msg_serial["contenido"])

# Resumen ejecutivo en cards
st.markdown("**📊 Resumen ejecutivo:**")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

n_alertas    = len(alertas)
n_recs       = len(recs)
n_plan       = sum(1 for v in plan.values()
                   if (v is True) or (isinstance(v, list) and v))
completitud  = conoc.get("densidad_datos", 100)

badge_alerta = "badge-err"  if n_alertas > 3 else \
               "badge-warn" if n_alertas > 0 else "badge-ok"

col_s1.markdown(
    f'<div style="text-align:center;padding:12px;background:#f8fafc;'
    f'border-radius:8px;border:1px solid #e2e8f0;">'
    f'<div style="font-size:24px;font-weight:700;color:#1e293b;">'
    f'{n_alertas}</div>'
    f'<div style="font-size:12px;color:#64748b;">Alertas críticas</div>'
    f'</div>', unsafe_allow_html=True
)
col_s2.markdown(
    f'<div style="text-align:center;padding:12px;background:#f8fafc;'
    f'border-radius:8px;border:1px solid #e2e8f0;">'
    f'<div style="font-size:24px;font-weight:700;color:#1e293b;">'
    f'{n_recs}</div>'
    f'<div style="font-size:12px;color:#64748b;">Recomendaciones</div>'
    f'</div>', unsafe_allow_html=True
)
col_s3.markdown(
    f'<div style="text-align:center;padding:12px;background:#f8fafc;'
    f'border-radius:8px;border:1px solid #e2e8f0;">'
    f'<div style="font-size:24px;font-weight:700;color:#1e293b;">'
    f'{n_plan}</div>'
    f'<div style="font-size:12px;color:#64748b;">Acciones en plan</div>'
    f'</div>', unsafe_allow_html=True
)
col_s4.markdown(
    f'<div style="text-align:center;padding:12px;background:#f8fafc;'
    f'border-radius:8px;border:1px solid #e2e8f0;">'
    f'<div style="font-size:24px;font-weight:700;color:#1e293b;">'
    f'{completitud}%</div>'
    f'<div style="font-size:12px;color:#64748b;">Completitud datos</div>'
    f'</div>', unsafe_allow_html=True
)
