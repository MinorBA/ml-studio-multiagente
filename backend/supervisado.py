# =============================================================================
# backend/supervisado.py
# CLASE: Supervisado  (hereda de analisisEDA)
# ─────────────────────────────────────────────────────────────────────────────
# Funcionalidades comunes a Clasificación y Regresión:
#   • Preparación de datos (split + estandarización)
#   • Predicción
#   • Cálculo de indices_general (Matriz de Confusión)
#   • Evaluación estándar
#
# Las subclases Clasificacion y Regresion heredan de esta clase.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import confusion_matrix

from backend.eda_base import analisisEDA


class Supervisado(analisisEDA):
    """
    Clase padre con funciones compartidas entre Clasificación y Regresión.
    Hereda de analisisEDA → tiene acceso a todos los gráficos EDA,
    limpieza, helpers estáticos, etc.

    Parámetros
    ----------
    df             : pd.DataFrame — datos con features + target
    columna_target : str          — nombre de la columna a predecir
    test_size      : float        — proporción de test (default 0.25)
    """

    def __init__(self, df: pd.DataFrame,
                 columna_target: str = "target",
                 test_size: float = 0.25):
        super().__init__()                       # inicializa analisisEDA
        self._df             = df                # sobreescribe el df vacío
        self._columna_target = columna_target
        self._test_size      = test_size

    def get_nombre(self) -> str:
        """Retorna el nombre de la clase. Usado por el sistema MAS."""
        return self.__class__.__name__

    # ────────────────────── PROPERTY ────────────────────────────────────────
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, v):
        self._df = v

    @property
    def columna_target(self) -> str:
        return self._columna_target

    @columna_target.setter
    def columna_target(self, v: str):
        self._columna_target = v

    # ────────────────────── PREPARAR DATOS ──────────────────────────────────
    def preparar_datos(self):
        """
        Split + estandarización.

        Retorna
        -------
        X_train, X_test, y_train, y_test, y_completo
        """
        X = self._df.drop(columns=[self._columna_target])
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        y = self._df[self._columna_target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self._test_size)
        return X_train, X_test, y_train, y_test, y

    # ────────────────────── PREDICCIÓN Y EVALUACIÓN ────────────────────────
    def predecir(self, modelo, X_test) -> np.ndarray:
        return modelo.predict(X_test)

    def indices_general(self, MC: np.ndarray,
                         nombres: list = None) -> dict:
        """
        Calcula métricas desde una Matriz de Confusión.

        Retorna
        -------
        dict con: Matriz de Confusión | Precisión Global |
                  Error Global | Precisión por categoría
        """
        precision_global    = np.sum(MC.diagonal()) / np.sum(MC)
        error_global        = 1 - precision_global
        precision_categoria = pd.DataFrame(
            MC.diagonal() / np.sum(MC, axis=1)).T
        if nombres is not None:
            precision_categoria.columns = nombres
        return {
            "Matriz de Confusión":     MC,
            "Precisión Global":        precision_global,
            "Error Global":            error_global,
            "Precisión por categoría": precision_categoria,
        }

    def evaluar(self, y_test, y_pred, y) -> dict:
        """Genera la MC y devuelve indices_general como dict."""
        MC = confusion_matrix(y_test, y_pred)
        return self.indices_general(MC, list(np.unique(y)))
