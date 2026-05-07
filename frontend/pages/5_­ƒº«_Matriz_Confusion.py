# =============================================================================
# frontend/pages/5_🧮_Matriz_Confusion.py
# Página para evaluación manual de MC 2×2
# =============================================================================

import streamlit as st
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.matriz_confusion import MatrizDeConfusion

st.set_page_config(page_title="Matriz de Confusión", page_icon="🧮",
                    layout="wide")
st.title("🧮 Evaluación — Matriz de Confusión 2×2")
st.markdown("---")

st.markdown(
    """
    Ingrese los valores de su Matriz de Confusión para calcular
    automáticamente todas las métricas de evaluación.
    """
)

st.markdown("#### Estructura de la MC:")
st.markdown("""
|             | Predicho + | Predicho − |
|-------------|:----------:|:----------:|
| **Real +**  |     VP     |     FP     |
| **Real −**  |     FN     |     VN     |
""")

st.markdown("---")

c1, c2 = st.columns(2)
with c1:
    vp = st.number_input("VP (Verdaderos Positivos)", min_value=0,
                          value=150, step=1)
    fn = st.number_input("FN (Falsos Negativos)", min_value=0,
                          value=5, step=1)
with c2:
    fp = st.number_input("FP (Falsos Positivos)", min_value=0,
                          value=10, step=1)
    vn = st.number_input("VN (Verdaderos Negativos)", min_value=0,
                          value=135, step=1)

if st.button("▶️ Calcular métricas"):
    mc = MatrizDeConfusion([[vp, fp], [fn, vn]])
    resultado = mc.calcular()

    if "error" in resultado:
        st.error(resultado["error"])
    else:
        st.success("Métricas calculadas correctamente")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Precisión Global",
                      f"{resultado['Precisión Global']:.4f}")
        col_b.metric("Error Global",
                      f"{resultado['Error Global']:.4f}")
        col_c.metric("Precisión Positiva",
                      f"{resultado['Precisión Positiva']:.4f}")
        col_d.metric("Precisión Negativa",
                      f"{resultado['Precisión Negativa']:.4f}")

        st.markdown("---")
        st.subheader("📋 Detalle completo")
        for nombre, valor in resultado.items():
            st.write(f"**{nombre}:** {valor:.4f}")

        # Heatmap de la MC
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        mc_array = np.array([[vp, fp], [fn, vn]])
        fig, ax = plt.subplots(figsize=(5, 4), dpi=130)
        sns.heatmap(mc_array, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Positivo", "Negativo"],
                    yticklabels=["Positivo", "Negativo"])
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusión")
        plt.tight_layout()
        st.pyplot(fig)
