import streamlit as st

st.set_page_config(
    page_title="ML Studio MAS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.badge { display:inline-block; padding:4px 14px; border-radius:20px;
         font-size:13px; font-weight:600; margin:4px 4px 4px 0; }
.badge-inf { background:#dbeafe; color:#1d4ed8; }
.integrante-card {
    background: linear-gradient(135deg, #ede9fe, #dbeafe);
    border: 2px solid #6366f1; border-radius: 12px;
    padding: 20px 24px; text-align: center;
}
.integrante-nombre { font-size:22px; font-weight:800; color:#3730a3; }
.integrante-rol    { font-size:14px; color:#6366f1; margin-top:4px; }
.obj-card { background:#f0f9ff; border:1px solid #7dd3fc; border-radius:8px;
            padding:8px 14px; margin:4px 0; font-size:14px; color:#0c4a6e; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 ML Studio MAS")
st.sidebar.caption("Sistema Multiagente para Machine Learning")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Páginas del sistema:**

1. 🔍 **EDA** — Análisis exploratorio
2. 🔬 **Clustering** — No supervisado
3. 🎯 **Clasificación** — Supervisado
4. 📈 **Regresión** — Supervisado
5. 🧠 **Coordinador MAS** — Dashboard ejecutivo
""")
st.sidebar.markdown("---")
st.sidebar.caption("EIF420O — UNA Costa Rica — I Ciclo 2026")

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1e1b4b,#312e81,#4338ca);
     border-radius:16px;padding:32px 40px;margin-bottom:24px;color:white;">
  <h1 style="margin:0;font-size:2rem;">
      🧠 Sistema Multiagente Inteligente para ML
  </h1>
  <p style="margin:8px 0 4px;font-size:1.05rem;opacity:0.9;">
      Automatización del ciclo completo de Machine Learning
  </p>
  <p style="margin:0;font-size:0.9rem;opacity:0.75;">
      EIF420O – Inteligencia Artificial &nbsp;|&nbsp;
      Universidad Nacional de Costa Rica &nbsp;|&nbsp;
      I Ciclo 2026 &nbsp;|&nbsp;
      👨‍🏫 Dr. Juan De Dios Murillo Morera
  </p>
</div>
""", unsafe_allow_html=True)

col_int, col_info = st.columns([1, 2])

with col_int:
    st.markdown("""
    <div class="integrante-card">
      <div class="integrante-nombre">Minor Brenes Aguilar</div>
      <div class="integrante-rol">Estudiante — EIF420O</div>
      <div class="integrante-rol" style="margin-top:8px;font-size:13px;color:#4338ca;">
          Carné: 116730106
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_info:
    st.markdown("""
    <div style="background:#f8fafc;border-radius:10px;padding:16px 20px;
    border:1px solid #e2e8f0;height:100%;">
    <b>Laboratorio:</b> #04 — Sistema Multiagente para ML<br>
    <b>Curso:</b> EIF420O – Inteligencia Artificial<br>
    <b>Universidad:</b> Universidad Nacional de Costa Rica<br>
    <b>Escuela:</b> Escuela de Informática<br>
    <b>Período:</b> I Ciclo 2026
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── ARQUITECTURA MAS ─────────────────────────────────────────────────────────

st.markdown(
    "<h3 style='text-align:center;'>🏗️ Arquitectura del Sistema MAS</h3>",
    unsafe_allow_html=True
)

col_izq, col_centro, col_der = st.columns([1, 2, 1])
with col_centro:
    st.code("""
          Usuario
             │
             ▼
  ┌──────────────────────────┐
  │  AgenteCoordinador MAS   │
  └──────────────────────────┘
    │      │       │      │
    ▼      ▼       ▼      ▼
   EDA  Cluster  Clasif  Regres
    """, language="text")

    st.markdown(
        "<p style='text-align:center;'><b>Jerarquía de clases:</b></p>",
        unsafe_allow_html=True
    )
    st.code("""
analisisEDA        (AgenteEDA)
├── NoSupervisado  (AgenteClustering)
└── Supervisado
    ├── Clasificacion
    └── Regresion

AgenteCoordinador  (composición)
    ├── agente_eda
    ├── agente_clustering
    ├── agente_clasificacion
    └── agente_regresion
    """, language="python")

st.markdown("---")


# ─── INSTRUCCIONES ────────────────────────────────────────────────────────────
st.markdown("### 🚀 Cómo usar el sistema")
c1, c2 = st.columns(2)
with c1:
    st.info("**Paso 1 — EDA:** Carga tu CSV y ejecuta el AgenteEDA. "
            "Infiere la variable objetivo, detecta outliers y genera el plan de preprocesamiento.")
    st.info("**Paso 2 — Clustering:** Ejecuta el AgenteClustering para descubrir "
            "patrones no supervisados en los datos.")
with c2:
    st.info("**Paso 3 — Clasificación o Regresión:** Entrena y compara múltiples modelos "
            "supervisados con benchmarking automático.")
    st.info("**Paso 4 — Coordinador MAS:** Ejecuta el flujo completo de forma automática "
            "y descarga el reporte JSON consolidado.")