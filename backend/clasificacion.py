# =============================================================================
# backend/clasificacion.py
# CLASE: Clasificacion  (hereda de Supervisado)
# ─────────────────────────────────────────────────────────────────────────────
# Algoritmos: KNN, Árbol de Decisión, Random Forest, GradientBoosting,
#             AdaBoost
# Funcionalidades: entrenamiento, barrido de hiperparámetros, benchmark
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neighbors  import KNeighborsClassifier
from sklearn.tree       import DecisionTreeClassifier
from sklearn.ensemble   import (RandomForestClassifier,
                                 AdaBoostClassifier,
                                 GradientBoostingClassifier)
from sklearn.metrics    import confusion_matrix
from sklearn.model_selection import GridSearchCV

from backend.supervisado import Supervisado


class Clasificacion(Supervisado):
    """
    Clasificación supervisada — hereda de Supervisado.

    Uso típico
    ----------
    clf = Clasificacion(df, columna_target="target")
    res = clf.ejecutar_modelo("KNN", n_neighbors=5, algorithm="auto")
    tabla = clf.benchmark()
    """

    def __init__(self, df: pd.DataFrame,
                 columna_target: str = "target",
                 test_size: float = 0.25):
        super().__init__(df, columna_target, test_size)

    # ────────────────────── BUILDERS DE MODELOS ─────────────────────────────
    @staticmethod
    def _build_knn(n_neighbors=5, algorithm="auto"):
        return KNeighborsClassifier(n_neighbors=n_neighbors,
                                     algorithm=algorithm)

    @staticmethod
    def _build_dt(min_samples_split=2, max_depth=4):
        return DecisionTreeClassifier(min_samples_split=min_samples_split,
                                       max_depth=max_depth)

    @staticmethod
    def _build_rf(n_estimators=100, min_samples_split=2, max_depth=4):
        return RandomForestClassifier(n_estimators=n_estimators,
                                       min_samples_split=min_samples_split,
                                       max_depth=max_depth)

    @staticmethod
    def _build_xg(n_estimators=100, min_samples_split=2, max_depth=4):
        return GradientBoostingClassifier(n_estimators=n_estimators,
                                           min_samples_split=min_samples_split,
                                           max_depth=max_depth)

    @staticmethod
    def _build_ada(estimator=None, n_estimators=100):
        if estimator is None:
            estimator = DecisionTreeClassifier(min_samples_split=2,
                                                max_depth=4)
        return AdaBoostClassifier(estimator=estimator,
                                   n_estimators=n_estimators)

    # ────────────────────── MÉTODO GENÉRICO DE EJECUCIÓN ───────────────────
    def ejecutar_modelo(self, nombre: str, **kwargs) -> dict:
        """
        Entrena un modelo por nombre y devuelve métricas.

        Parámetros
        ----------
        nombre : str — 'KNN' | 'DT' | 'RF' | 'XG' | 'ADA'
        **kwargs : hiperparámetros del modelo

        Retorna
        -------
        dict con: indices + 'y_pred' + 'y_test' + 'modelo'
        """
        builders = {
            "KNN": self._build_knn,
            "DT":  self._build_dt,
            "RF":  self._build_rf,
            "XG":  self._build_xg,
            "ADA": self._build_ada,
        }
        if nombre not in builders:
            raise ValueError(f"Modelo '{nombre}' no soportado. "
                             f"Opciones: {list(builders.keys())}")
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        modelo = builders[nombre](**kwargs)
        modelo.fit(X_train, y_train)
        y_pred  = self.predecir(modelo, X_test)
        indices = self.evaluar(y_test, y_pred, y)
        indices["y_pred"] = y_pred
        indices["y_test"] = y_test
        indices["modelo"] = modelo
        return indices

    # ────────────────────── BARRIDOS DE HIPERPARÁMETROS ─────────────────────
    def barrido_knn(self, rango_k=range(1, 11),
                    algorithms=("auto", "ball_tree",
                                "kd_tree", "brute")) -> pd.DataFrame:
        """Barre K y algoritmos de KNN, devuelve tabla de resultados."""
        rows = []
        for k in rango_k:
            for alg in algorithms:
                r = self.ejecutar_modelo("KNN",
                                          n_neighbors=k, algorithm=alg)
                rows.append({
                    "k": k, "algorithm": alg,
                    "PG": r["Precisión Global"],
                    "EG": r["Error Global"],
                })
        return pd.DataFrame(rows)

    def barrido_dt(self, rango_depth=range(1, 11)) -> pd.DataFrame:
        rows = []
        for d in rango_depth:
            r = self.ejecutar_modelo("DT", min_samples_split=2, max_depth=d)
            rows.append({"max_depth": d,
                         "PG": r["Precisión Global"],
                         "EG": r["Error Global"]})
        return pd.DataFrame(rows)

    def barrido_rf(self, rango_depth=range(1, 11)) -> pd.DataFrame:
        rows = []
        for d in rango_depth:
            r = self.ejecutar_modelo("RF", n_estimators=100,
                                      min_samples_split=2, max_depth=d)
            rows.append({"max_depth": d,
                         "PG": r["Precisión Global"],
                         "EG": r["Error Global"]})
        return pd.DataFrame(rows)

    def barrido_xg(self, rango_depth=range(1, 11)) -> pd.DataFrame:
        rows = []
        for d in rango_depth:
            r = self.ejecutar_modelo("XG", n_estimators=100,
                                      min_samples_split=2, max_depth=d)
            rows.append({"max_depth": d,
                         "PG": r["Precisión Global"],
                         "EG": r["Error Global"]})
        return pd.DataFrame(rows)

    def barrido_ada(self, rango_n=range(10, 101, 10)) -> pd.DataFrame:
        rows = []
        est = DecisionTreeClassifier(min_samples_split=2, max_depth=4)
        for n in rango_n:
            r = self.ejecutar_modelo("ADA", estimator=est, n_estimators=n)
            rows.append({"n_estimators": n,
                         "PG": r["Precisión Global"],
                         "EG": r["Error Global"]})
        return pd.DataFrame(rows)

    # ────────────────────── GRID SEARCH ────────────────────────────────────
    def grid_search_ada(self, param_grid: dict) -> dict:
        """GridSearchCV sobre AdaBoost."""
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        gs = GridSearchCV(
            AdaBoostClassifier(estimator=DecisionTreeClassifier()),
            param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        indices = self.evaluar(y_test, gs.predict(X_test), y)
        indices["best_params"] = gs.best_params_
        indices["best_score_cv"] = gs.best_score_
        return indices

    # ────────────────────── BENCHMARK ──────────────────────────────────────
    def benchmark(self, clase_positiva=None,
                  clase_negativa=None) -> pd.DataFrame:
        """
        Ejecuta todos los algoritmos con hiperparámetros por defecto
        y devuelve tabla comparativa.

        Parámetros
        ----------
        clase_positiva : nombre de la clase + en el target (para PP)
        clase_negativa : nombre de la clase - en el target (para PN)
        """
        configs = {
            "KNN":      dict(n_neighbors=3, algorithm="ball_tree"),
            "DT":       dict(min_samples_split=8, max_depth=1),
            "RF":       dict(n_estimators=200, min_samples_split=9,
                             max_depth=8),
            "XGBoost":  dict(n_estimators=100, min_samples_split=5,
                             max_depth=1),
            "AdaBoost": dict(
                estimator=GradientBoostingClassifier(
                    n_estimators=100, min_samples_split=5, max_depth=1),
                n_estimators=10),
        }
        # mapeo nombre → nombre de builder
        mapeo_builder = {"KNN": "KNN", "DT": "DT", "RF": "RF",
                         "XGBoost": "XG", "AdaBoost": "ADA"}
        resultados = {}
        for nombre, params in configs.items():
            r  = self.ejecutar_modelo(mapeo_builder[nombre], **params)
            pc = r["Precisión por categoría"]
            clases = list(pc.columns)
            cp = clase_positiva if clase_positiva and clase_positiva in clases \
                else clases[-1]
            cn = clase_negativa if clase_negativa and clase_negativa in clases \
                else clases[0]
            resultados[nombre] = {
                "PG": r["Precisión Global"],
                "EG": r["Error Global"],
                "PP": pc[cp].values[0] if cp in pc.columns else np.nan,
                "PN": pc[cn].values[0] if cn in pc.columns else np.nan,
            }
        return pd.DataFrame(resultados).T

    # ────────────────────── GRÁFICOS ───────────────────────────────────────
    @staticmethod
    def fig_matriz_confusion(y_test, y_pred, clases=None) -> plt.Figure:
        """Heatmap de la Matriz de Confusión."""
        import seaborn as sns
        MC  = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
        sns.heatmap(MC, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=clases, yticklabels=clases)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusión")
        plt.tight_layout()
        return fig
