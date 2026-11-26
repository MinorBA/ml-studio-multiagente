# =============================================================================
# backend/__init__.py
#
# Paquete backend de ML Studio. Se deja deliberadamente vacío de
# importaciones a nivel de paquete: cada página de Streamlit y cada
# módulo importan directamente lo que necesitan
# (ej. `from backend.clasificacion import Clasificacion`).
#
# Esto evita que una dependencia opcional faltante (p. ej. `prince` o
# `umap-learn`, usadas solo por NoSupervisado/CACP) rompa la carga de
# módulos que no la necesitan, como Clasificacion o Regresion.
# =============================================================================
