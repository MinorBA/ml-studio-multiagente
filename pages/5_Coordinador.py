# =============================================================================
# pages/5_Coordinador.py — AgenteCoordinador: Superagente MAS
# =============================================================================

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.agente_coordinador import AgenteCoordinador
from shared_utils import obtener_df

st.set_page_config(page_title="AgenteCoordinador",
                   page_icon="🧠", layout="wide")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.paso-card { background:#f8fafc; border-left:4px solid #6366f1;
             border-radius:0 8px 8px 0; padding:12px 18px; margin:8px 0; }
.paso-num  { font-size:11px; font-weight:700; color:#6366f1;
             letter-spacing:1.5px; text-transform:uppercase; margin-bottom:2px; }
.paso-titulo { font-size:16px; font-weight:700; color:#1e293b; margin:0; }
.badge { display:inline-block; padding:4px 14px; border-radius:20px;
         font-size:13px; font-weight:600; margin:4px 4px 4px 0; }
.badge-eda  { background:#dbeafe; color:#1d4ed8; }
.badge-cl   { background:#ede9fe; color:#5b21b6; }
.badge-clf  { background:#dcfce7; color:#166534; }
.badge-reg  { background:#fef3c7; color:#92400e; }
.badge-ok   { background:#dcfce7; color:#166534; }
.badge-warn { background:#fef9c3; color:#854d0e; }
.badge-err  { background:#fee2e2; color:#991b1b; }
.alerta-card { background:#fff7ed; border:1px solid #fb923c; border-radius:8px;
               padding:10px 14px; margin:5px 0; font-size:14px; color:#7c2d12; }
.concl-card  { background:#f0f9ff; border:1px solid #7dd3fc; border-radius:8px;
               padding:10px 14px; margin:5px 0; font-size:14px; color:#0c4a6e; }
.winner-card { background:linear-gradient(135deg,#f0fdf4,#dcfce7);
               border:2px solid #22c55e; border-radius:12px; padding:20px;
               text-align:center; }
.winner-title{ font-size:13px; color:#166534; font-weight:600;
               text-transform:uppercase; letter-spacing:1px; }
.winner-val  { font-size:32px; font-weight:800; color:#15803d; margin:8px 0; }
.winner-sub  { font-size:13px; color:#4ade80; }
.metric-mini { text-align:center; padding:12px; background:#f8fafc;
               border-radius:8px; border:1px solid #e2e8f0; }
.metric-mini .val { font-size:22px; font-weight:700; color:#1e293b; }
.metric-mini .lbl { font-size:12px; color:#64748b; }
.agente-box { border:1px solid #e2e8f0; border-radius:10px; padding:14px;
              margin:6px 0; background:#fafafa; }
.agente-title{ font-size:14px; font-weight:700; color:#334155;
               margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ─── ENCABEZADO ───────────────────────────────────────────────────────────────
st.markdown("## 🧠 Agente Coordinador — Sistema Multiagente MAS")
st.caption(
    "Orquesta automáticamente EDA → Clustering → Clasificación / Regresión "
    "en un único flujo coordinado."
)
st.markdown("---")

# ─── ARQUITECTURA VISUAL ──────────────────────────────────────────────────────
with st.expander("📐 Arquitectura del Sistema MAS", expanded=False):
    st.markdown("""
    ```
    Usuario
        │
        ▼
    ┌─────────────────────────────────────────────────────┐
    │          AgenteCoordinador (Superagente MAS)         │
    │                                                     │
    │  ┌──────────┐  ┌────────────┐  ┌───────────────┐  │
    │  │ AgenteEDA│  │ Agente     │  │ AgenteSuperv. │  │
    │  │ (paso 1) │→ │ Clustering │→ │ Clasif/Regres │  │
    │  └──────────┘  └────────────┘  └───────────────┘  │
    │                                                     │
    │  Percibir → Analizar → Razonar → Decidir →         │
    │  Ejecutar → Consolidar → Comunicar                  │
    └─────────────────────────────────────────────────────┘
    ```
    """)

# ─── DATOS ────────────────────────────────────────────────────────────────────
df = obtener_df()
if df is None:
    st.info("📂 Primero cargue un CSV en la página **EDA**.")
    st.stop()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuración del Coordinador")
st.sidebar.info(
    "🤖 El coordinador infiere automáticamente el tipo de problema, "
    "decide qué agentes activar y orquesta el flujo completo."
)
ejecutar = st.sidebar.button("▶ Ejecutar Sistema MAS Completo",
                              type="primary", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**⏱️ Tiempo estimado:** 2-4 minutos")
st.sidebar.caption(
    "Se ejecutan: EDA + Clustering + Clasificación o Regresión."
)

if "coordinador_resultado" not in st.session_state:
    st.session_state["coordinador_resultado"] = None

if ejecutar:
    # ── Barra de progreso dinámica ─────────────────────────────────────────
    progress_bar  = st.progress(0)
    status_text   = st.empty()

    def actualizar_progreso(step, total, mensaje):
        pct = int(step / max(total, 1) * 100)
        progress_bar.progress(pct)
        status_text.markdown(f"⚙️ `{mensaje}`")

    coord    = AgenteCoordinador()
    resultado = coord.ejecutar(df, progress_callback=actualizar_progreso)
    progress_bar.progress(100)
    status_text.markdown("✅ **Sistema MAS completado.**")

    st.session_state["coordinador_resultado"] = resultado
    # Compartir con otras páginas si las necesitan
    st.session_state["mas_conclusiones"] = resultado.get("conclusiones", [])
    st.session_state["mas_mejor_global"] = resultado.get("mejor_global", {})

resultado = st.session_state["coordinador_resultado"]
if resultado is None:
    st.warning("Presiona **▶ Ejecutar Sistema MAS Completo** para comenzar.")
    st.stop()

# ── Extraer datos ──────────────────────────────────────────────────────────
msg          = resultado["mensaje"]
consolidado  = resultado["consolidado"]
conclusiones = resultado["conclusiones"]
mejor_global = resultado["mejor_global"]
planes       = resultado["planes"]
tipos        = resultado["tipos_problema"]
justif       = resultado["justificaciones"]
errores      = resultado["errores"]
resultados   = resultado["resultados"]

# ═════════════════════════════════════════════════════════════════════════════
# PASO 1 — PERCEPCIÓN
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 1</div>
  <div class="paso-titulo">📡 Percepción — Datasets Recibidos</div>
</div>
""", unsafe_allow_html=True)

datasets_proc = msg.get("datasets_procesados", [])
c1, c2, c3 = st.columns(3)
c1.metric("Datasets procesados", len(datasets_proc))
c2.metric("Agentes activados",
          sum(len(v) for v in planes.values()))
c3.metric("Errores totales", len(errores))

for nombre in datasets_proc:
    tipo_info = tipos.get(nombre, {})
    plan_ds   = planes.get(nombre, [])
    badges = {
        "EDA":          '<span class="badge badge-eda">🔍 EDA</span>',
        "Clustering":   '<span class="badge badge-cl">🔬 Clustering</span>',
        "Clasificacion":'<span class="badge badge-clf">🎯 Clasificación</span>',
        "Regresion":    '<span class="badge badge-reg">📈 Regresión</span>',
    }
    plan_html = " ".join(badges.get(a, a) for a in plan_ds)
    st.markdown(
        f'<div class="agente-box"><div class="agente-title">📦 {nombre}</div>'
        f'{plan_html}</div>',
        unsafe_allow_html=True
    )

# ═════════════════════════════════════════════════════════════════════════════
# PASO 2-3 — ANÁLISIS Y RAZONAMIENTO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASOS 2 Y 3</div>
  <div class="paso-titulo">🧠 Análisis y Razonamiento del Coordinador</div>
</div>
""", unsafe_allow_html=True)

for j in justif:
    st.markdown(
        f'<div class="concl-card">🤖 {j}</div>',
        unsafe_allow_html=True
    )

# ═════════════════════════════════════════════════════════════════════════════
# RESULTADOS POR AGENTE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">RESULTADOS</div>
  <div class="paso-titulo">📊 Métricas por Agente</div>
</div>
""", unsafe_allow_html=True)

for nombre, res_ds in resultados.items():
    with st.expander(f"📦 Dataset: {nombre}", expanded=True):
        tabs_labels = []
        if "EDA"          in res_ds: tabs_labels.append("🔍 EDA")
        if "Clustering"   in res_ds: tabs_labels.append("🔬 Clustering")
        if "Clasificacion" in res_ds: tabs_labels.append("🎯 Clasificación")
        if "Regresion"    in res_ds: tabs_labels.append("📈 Regresión")

        if not tabs_labels:
            st.warning("Sin resultados para este dataset.")
            continue

        tabs = st.tabs(tabs_labels)
        tab_idx = 0

        # ── EDA ──────────────────────────────────────────────────────────
        if "EDA" in res_ds:
            with tabs[tab_idx]:
                eda_c = res_ds["EDA"]["conocimiento"]
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Filas",       eda_c.get("filas", 0))
                e2.metric("Columnas",    eda_c.get("columnas", 0))
                e3.metric("Densidad",    f"{eda_c.get('densidad_datos',100)}%")
                e4.metric("Alertas EDA", len(eda_c.get("alertas", [])))

                pre_elim = eda_c.get("pre_eliminadas", [])
                if pre_elim:
                    st.markdown(
                        f'<div class="alerta-card">🗑️ Columnas eliminadas: '
                        f'{", ".join(pre_elim)}</div>',
                        unsafe_allow_html=True
                    )
                var_inf = eda_c.get("variable_objetivo", "N/A")
                tipo_p  = eda_c.get("tipo_problema", "")
                subtipo = eda_c.get("subtipo", "")
                just_i  = eda_c.get("justificacion_objetivo", "")
                st.markdown(
                    f'<div class="concl-card">🤖 Variable objetivo inferida: '
                    f'<b>{var_inf}</b> → {tipo_p} / {subtipo}<br>'
                    f'<small>{just_i}</small></div>',
                    unsafe_allow_html=True
                )

                # Figuras EDA
                figs_eda = res_ds["EDA"].get("figuras", {})
                col_e1, col_e2 = st.columns(2)
                if "correlacion_pearson" in figs_eda:
                    with col_e1:
                        st.markdown("**Correlación Pearson**")
                        st.pyplot(figs_eda["correlacion_pearson"],
                                  use_container_width=True)
                if "histogramas" in figs_eda:
                    with col_e2:
                        st.markdown("**Histogramas**")
                        st.pyplot(figs_eda["histogramas"],
                                  use_container_width=True)
            tab_idx += 1

        # ── Clustering ───────────────────────────────────────────────────
        if "Clustering" in res_ds:
            with tabs[tab_idx]:
                cl_msg = res_ds["Clustering"]["mensaje"]["contenido"]
                cl_bench = res_ds["Clustering"]["benchmark"]
                cl_figs  = res_ds["Clustering"]["figuras"]

                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Mejor algoritmo", cl_msg.get("mejor_algoritmo"))
                cc2.metric("Silhouette",
                           f"{cl_msg.get('mejor_silhouette',0):.4f}")
                cc3.metric("K óptimo",        cl_msg.get("k_optimo"))

                if not cl_bench.empty:
                    st.dataframe(cl_bench.style.format(precision=4),
                                 use_container_width=True, hide_index=True)

                if "kmedias_scatter" in cl_figs:
                    st.pyplot(cl_figs["kmedias_scatter"],
                              use_container_width=True)
            tab_idx += 1

        # ── Clasificación ────────────────────────────────────────────────
        if "Clasificacion" in res_ds:
            with tabs[tab_idx]:
                clf_msg   = res_ds["Clasificacion"]["mensaje"]["contenido"]
                clf_bench = res_ds["Clasificacion"]["resultados_df"]
                clf_figs  = res_ds["Clasificacion"]["figuras"]

                cb1, cb2, cb3 = st.columns(3)
                cb1.metric("Mejor modelo",    clf_msg.get("mejor_modelo"))
                cb2.metric(clf_msg.get("metrica_principal","Accuracy"),
                           f"{clf_msg.get('valor_metrica',0):.4f}")
                cb3.metric("SMOTE",
                           "✅" if clf_msg.get("apply_smote") else "❌")

                if clf_bench is not None and not clf_bench.empty:
                    st.dataframe(
                        clf_bench.style.format(
                            {c: "{:.4f}" for c in clf_bench.columns
                             if c != "Modelo"}, na_rep="N/A"),
                        use_container_width=True, hide_index=True)

                col_clf1, col_clf2 = st.columns(2)
                if "distribucion_clases" in clf_figs:
                    with col_clf1:
                        st.pyplot(clf_figs["distribucion_clases"],
                                  use_container_width=True)
                if "matriz_confusion" in clf_figs:
                    with col_clf2:
                        st.pyplot(clf_figs["matriz_confusion"],
                                  use_container_width=True)
            tab_idx += 1

        # ── Regresión ────────────────────────────────────────────────────
        if "Regresion" in res_ds:
            with tabs[tab_idx]:
                reg_msg   = res_ds["Regresion"]["mensaje"]["contenido"]
                reg_bench = res_ds["Regresion"]["resultados_df"]
                reg_figs  = res_ds["Regresion"]["figuras"]

                rb1, rb2, rb3 = st.columns(3)
                rb1.metric("Mejor modelo", reg_msg.get("mejor_modelo"))
                rb2.metric("RMSE",
                           f"{reg_msg.get('mejor_rmse',0):.4f}"
                           if reg_msg.get('mejor_rmse') else "N/A")
                rb3.metric("R²",
                           f"{reg_msg.get('mejor_r2',0):.4f}"
                           if reg_msg.get('mejor_r2') else "N/A")

                if reg_bench is not None and not reg_bench.empty:
                    st.dataframe(
                        reg_bench.style.format(
                            {c: "{:.4f}" for c in reg_bench.columns
                             if c != "Modelo"}, na_rep="N/A"),
                        use_container_width=True, hide_index=True)

                try:
                    import plotly  # noqa: F401  (solo para verificar disponibilidad)
                    if "rmse_comparativo" in reg_figs:
                        st.plotly_chart(reg_figs["rmse_comparativo"],
                                        use_container_width=True)
                except ImportError:
                    if "pred_vs_real" in reg_figs:
                        st.pyplot(reg_figs["pred_vs_real"],
                                  use_container_width=True)
            tab_idx += 1

# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD EJECUTIVO — MEJOR MODELO GLOBAL
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">DASHBOARD EJECUTIVO</div>
  <div class="paso-titulo">🏆 Mejor Modelo Global</div>
</div>
""", unsafe_allow_html=True)

if mejor_global.get("modelo"):
    col_w, col_c = st.columns([1, 2])
    with col_w:
        valor_fmt = (f"{mejor_global['valor']:.4f}"
                     if mejor_global.get("valor") else "N/A")
        st.markdown(
            f'''<div class="winner-card">
            <div class="winner-title">🏆 Mejor modelo global</div>
            <div class="winner-val">{mejor_global["modelo"]}</div>
            <div class="winner-sub">
                {mejor_global.get("agente","")} ·
                {mejor_global.get("metrica","")} = {valor_fmt}
            </div>
            <div class="winner-sub" style="margin-top:6px">
                Dataset: {mejor_global.get("dataset","")}
            </div>
            </div>''',
            unsafe_allow_html=True
        )
    with col_c:
        # Tabla resumen de todos los modelos ganadores por dataset
        rows = []
        for nombre, res in consolidado.items():
            row = {"Dataset": nombre}
            if "eda" in res:
                row["Densidad"] = f"{res['eda'].get('densidad',100)}%"
                row["Variable Y"] = res["eda"].get("variable_obj", "")
            if "clustering" in res:
                row["Mejor Cluster"] = res["clustering"].get(
                    "mejor_algoritmo", "")
                row["Silhouette"] = f"{res['clustering'].get('mejor_silhouette',0):.4f}"
            if "clasificacion" in res:
                row["Mejor Modelo"] = res["clasificacion"].get(
                    "mejor_modelo", "")
                met = res["clasificacion"].get("metrica", "")
                val = res["clasificacion"].get("valor_metrica", 0)
                row[f"{met}"] = f"{val:.4f}"
            if "regresion" in res:
                row["Mejor Modelo"] = res["regresion"].get("mejor_modelo","")
                row["R²"] = f"{res['regresion'].get('mejor_r2',0):.4f}"
            rows.append(row)

        if rows:
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONCLUSIONES AUTOMÁTICAS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">CONCLUSIONES</div>
  <div class="paso-titulo">💬 Conclusiones Automáticas del Sistema</div>
</div>
""", unsafe_allow_html=True)

for c in conclusiones:
    st.markdown(
        f'<div class="concl-card">{c}</div>',
        unsafe_allow_html=True
    )

# ═════════════════════════════════════════════════════════════════════════════
# MENSAJE MAS FINAL
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="paso-card">
  <div class="paso-num">PASO 7 — COMUNICACIÓN</div>
  <div class="paso-titulo">📨 Mensaje Final del Sistema MAS</div>
</div>
""", unsafe_allow_html=True)

col_m1, col_m2 = st.columns([1, 2])
with col_m1:
    ts = msg.get("timestamp","")[:19].replace("T"," ")
    st.metric("Agente",            msg.get("agente",""))
    st.metric("Timestamp",         ts)
    st.metric("Datasets procesados", len(msg.get("datasets_procesados",[])))
    st.metric("Errores totales",   len(errores))
    if errores:
        with st.expander("⚠️ Ver errores del sistema"):
            for e in errores:
                st.error(f"**{e.get('paso','')}**: {e.get('error','')}")
with col_m2:
    with st.expander("📋 Mensaje completo del coordinador (JSON)",
                     expanded=False):
        # Versión serializable (sin DataFrames ni Figures)
        msg_serial = {
            "agente":              msg.get("agente"),
            "tipo":               msg.get("tipo"),
            "timestamp":          msg.get("timestamp"),
            "datasets_procesados": msg.get("datasets_procesados"),
            "plan_ejecutado":     msg.get("plan_ejecutado"),
            "mejor_modelo_global": msg.get("mejor_modelo_global"),
            "conclusiones":       msg.get("conclusiones"),
            "errores":            msg.get("errores"),
        }
        st.json(msg_serial)

    # Botón de descarga del reporte como JSON
    import json
    reporte_json = json.dumps(msg_serial, ensure_ascii=False, indent=2)
    st.download_button(
        label="⬇️ Descargar reporte JSON",
        data=reporte_json,
        file_name=f"reporte_mas_{msg.get('timestamp','')[:10]}.json",
        mime="application/json",
        use_container_width=True
    )
