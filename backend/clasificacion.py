# =============================================================================
# backend/clasificacion.py
# CLASE: Clasificacion  (hereda de Supervisado)
# ─────────────────────────────────────────────────────────────────────────────
# Algoritmos: KNN, Árbol de Decisión, Random Forest, GradientBoosting,
#             AdaBoost (con Decision Tree base), XGBoost (GradientBoosting).
# =============================================================================

import numpy as np
from datetime import datetime
try:
    from shared_utils import limpiar_df_para_agente as _limpiar
except ImportError:
    _limpiar = None
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import (RandomForestClassifier,
                                      AdaBoostClassifier,
                                      GradientBoostingClassifier)
from sklearn.metrics         import (confusion_matrix,
                                      f1_score,
                                      precision_score,
                                      roc_auc_score,
                                      ConfusionMatrixDisplay)
from sklearn.preprocessing   import label_binarize
from imblearn.over_sampling  import SMOTE

from backend.supervisado import Supervisado


class Clasificacion(Supervisado):
    """
    Clasificación supervisada — hereda de Supervisado.

    Uso típico (sin SMOTE)
    ----------------------
    clf = Clasificacion(df, columna_target="target")
    res = clf.ejecutar_modelo("KNN", n_neighbors=5, algorithm="auto")
    tabla = clf.benchmark()

    Uso típico (con SMOTE y métricas de desbalance)
    -------------------------------------------------------
    clf = Clasificacion(df, columna_target="target")
    res = clf.ejecutar_modelo("RF", n_estimators=100, max_depth=4,
                               apply_smote=True)
    print(res["F1-macro"], res["AUC-ROC"], res["Precisión"])
    tabla = clf.benchmark_desbalance()
    """

    def __init__(self, df: pd.DataFrame,
                 columna_target: str = "target",
                 test_size: float = 0.25):
        super().__init__(df, columna_target, test_size)
        # ── Atributos del agente inteligente ─────────────────────────── #
        self.conocimiento: dict = {}
        self.mensajes:     list = []
        self._errores:     list = []
        # ─────────────────────────────────────────────────────────────── #

    def get_nombre(self) -> str:
        """Retorna el nombre de la clase. Usado por el sistema MAS."""
        return self.__class__.__name__

    def benchmark_completo(self, apply_smote: bool = False) -> pd.DataFrame:
        """
        Tabla comparativa con TODOS los algoritmos del enunciado, con las
        columnas estandarizadas requeridas por el AgenteCoordinador.

        Columnas: Modelo, Accuracy, Precision, Recall, F1, AUC.

        Incluye: KNN, Decision Tree, Random Forest, AdaBoost, XGBoost,
                 Naive Bayes, Logistic Regression, SVM.
        """
        from sklearn.naive_bayes      import GaussianNB
        from sklearn.linear_model     import LogisticRegression
        from sklearn.svm              import SVC
        from sklearn.neighbors        import KNeighborsClassifier
        from sklearn.metrics          import (accuracy_score, precision_score,
                                              recall_score, f1_score,
                                              roc_auc_score)
        from sklearn.preprocessing    import LabelEncoder, label_binarize
        from sklearn.pipeline         import Pipeline
        from sklearn.compose          import ColumnTransformer
        from sklearn.preprocessing    import StandardScaler, OneHotEncoder
        import numpy as _np

        df_clean = self._df.dropna().copy()

        # ── Validación: dataset vacío tras eliminar nulos ────────────────
        if df_clean.empty:
            self._errores.append({
                "paso": "benchmark_completo",
                "error": "El dataset quedó vacío después de eliminar filas "
                         "con valores nulos. No es posible entrenar modelos.",
            })
            return pd.DataFrame(columns=[
                "Modelo", "Accuracy", "Precision", "Recall", "F1", "AUC"])

        X = df_clean.drop(columns=[self._columna_target])
        y = df_clean[self._columna_target]

        # ── Validación: se necesitan al menos 2 clases para clasificar ───
        if y.nunique() < 2:
            self._errores.append({
                "paso": "benchmark_completo",
                "error": f"La variable objetivo '{self._columna_target}' "
                         f"tiene una sola clase tras la limpieza de nulos. "
                         f"Se requieren al menos 2 clases para entrenar "
                         f"modelos de clasificación.",
            })
            return pd.DataFrame(columns=[
                "Modelo", "Accuracy", "Precision", "Recall", "F1", "AUC"])

        # Codificar target si es categórico
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        clases, conteos = _np.unique(y_enc, return_counts=True)
        es_binaria = len(clases) == 2

        # ── Validación de tamaño mínimo por clase ────────────────────────
        # train_test_split(..., stratify=y) requiere al menos 2 miembros
        # por clase (1 para train y 1 para test). Si alguna clase tiene
        # solo 1 registro, el split estratificado falla con:
        # "The least populated class in y has only 1 member..."
        clases_insuficientes = [
            (le.inverse_transform([c])[0], int(n))
            for c, n in zip(clases, conteos) if n < 2
        ]
        if clases_insuficientes:
            detalle = ", ".join(
                f"'{c}' (n={n})" for c, n in clases_insuficientes)
            mensaje_aviso = (
                f"No se pudo aplicar split estratificado: la(s) clase(s) "
                f"{detalle} tiene(n) menos de 2 registros, el mínimo "
                f"requerido para dividir en train/test. Se usará un split "
                f"sin estratificar; los resultados pueden ser menos "
                f"representativos para esas clases."
            )
            self._errores.append({
                "paso": "benchmark_completo_split",
                "error": mensaje_aviso,
            })
            self.conocimiento.setdefault("alertas", []).append(
                f"⚠ {mensaje_aviso}")
            usar_stratify = None
        else:
            usar_stratify = y_enc

        # Split (estratificado si es posible, sin estratificar si no)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=self._test_size,
            random_state=42, stratify=usar_stratify)

        # SMOTE opcional
        if apply_smote:
            try:
                from imblearn.over_sampling import SMOTE
                # SMOTE necesita k_neighbors+1 muestras en la clase minoritaria
                # (por defecto k_neighbors=5 → mínimo 6 muestras). Si hay menos,
                # se reduce k_neighbors automáticamente para evitar el error.
                _, conteos_train = _np.unique(y_train, return_counts=True)
                min_clase_train  = int(conteos_train.min())
                k_neighbors = max(1, min(5, min_clase_train - 1))
                sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_arr = X_train.select_dtypes(include="number").values
                X_train_arr, y_train = sm.fit_resample(X_train_arr, y_train)
                X_train = pd.DataFrame(X_train_arr,
                                       columns=X_train.select_dtypes(
                                           include="number").columns)
                X_test = X_test.select_dtypes(include="number")
            except Exception as e:
                self._errores.append({
                    "paso": "benchmark_completo_smote",
                    "error": f"SMOTE no se pudo aplicar ({e}). "
                             f"Se continúa sin sobremuestreo.",
                })

        # Preprocesador
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()
        transformers = []
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
        preprocesador = ColumnTransformer(transformers) if transformers else "passthrough"

        modelos_dict = {
            "KNN":                KNeighborsClassifier(n_neighbors=5),
            "Decision Tree":      self._build_dt(),
            "Random Forest":      self._build_rf(),
            "AdaBoost":           self._build_ada(),
            "XGBoost":            self._build_xg(),
            "Naive Bayes":        GaussianNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000,
                                                       random_state=42),
            "SVM":                SVC(probability=True, random_state=42),
        }

        filas = []
        for nombre, modelo in modelos_dict.items():
            try:
                pipe = Pipeline([("prep", preprocesador), ("clf", modelo)])
                pipe.fit(X_train, y_train)
                y_pred  = pipe.predict(X_test)
                acc     = round(accuracy_score(y_test, y_pred), 4)
                prec    = round(precision_score(y_test, y_pred,
                                                average="weighted",
                                                zero_division=0), 4)
                rec     = round(recall_score(y_test, y_pred,
                                             average="weighted",
                                             zero_division=0), 4)
                f1      = round(f1_score(y_test, y_pred,
                                         average="weighted",
                                         zero_division=0), 4)
                try:
                    if hasattr(pipe.named_steps["clf"], "predict_proba"):
                        y_prob = pipe.predict_proba(X_test)
                        if es_binaria:
                            auc = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
                        else:
                            y_bin = label_binarize(y_test, classes=clases)
                            auc   = round(roc_auc_score(y_bin, y_prob,
                                                         multi_class="ovr",
                                                         average="macro"), 4)
                    else:
                        auc = float("nan")
                except Exception:
                    auc = float("nan")
                filas.append({"Modelo": nombre, "Accuracy": acc,
                               "Precision": prec, "Recall": rec,
                               "F1": f1, "AUC": auc})
            except Exception as e:
                self._errores.append({
                    "paso": f"benchmark_completo_{nombre}",
                    "error": str(e),
                })
                filas.append({"Modelo": nombre, "Accuracy": float("nan"),
                               "Precision": float("nan"), "Recall": float("nan"),
                               "F1": float("nan"), "AUC": float("nan")})

        if not filas:
            self._errores.append({
                "paso": "benchmark_completo",
                "error": "No se generó ninguna fila de resultados: el "
                         "dataset no tiene modelos configurados o quedó "
                         "vacío tras la limpieza de nulos.",
            })

        return pd.DataFrame(filas)

    # =========================================================================
    # PREPARACIÓN DE DATOS — con opción SMOTE
    # =========================================================================
    def preparar_datos_smote(self):
        """
        Igual que preparar_datos() (heredado de Supervisado) pero aplica
        SMOTE al conjunto de entrenamiento para corregir el desbalance
        de clases.

        Retorna
        -------
        X_train_res, X_test, y_train_res, y_test, y_completo
        """
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        return X_train_res, X_test, y_train_res, y_test, y

    # =========================================================================
    # BUILDERS DE MODELOS 
    # =========================================================================
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
                                       max_depth=max_depth,
                                       random_state=42)

    @staticmethod
    def _build_xg(n_estimators=100, min_samples_split=2, max_depth=4):
        return GradientBoostingClassifier(n_estimators=n_estimators,
                                           min_samples_split=min_samples_split,
                                           max_depth=max_depth,
                                           random_state=42)

    @staticmethod
    def _build_ada(estimator=None, n_estimators=100):
        if estimator is None:
            estimator = DecisionTreeClassifier(min_samples_split=2,
                                                max_depth=4)
        return AdaBoostClassifier(estimator=estimator,
                                   n_estimators=n_estimators)

    # =========================================================================
    # MÉTRICAS DE DESBALANCE
    # =========================================================================
    @staticmethod
    def metricas_desbalance(modelo, X_test, y_test, y_pred,
                            clases: list) -> dict:
        """
         Calcula tres métricas clave para evaluar modelos sobre datasets
        con desbalance de clases:
          • F1-macro   — promedio macro del F1 por clase
          • AUC-ROC    — macro One-vs-Rest; usa predict_proba cuando está
                         disponible, NaN si el modelo no la soporta
          • Precisión  — promedio macro de la precisión por clase

        Parámetros
        ----------
        modelo   : modelo ya entrenado (con predict_proba si es posible)
        X_test   : features de test
        y_test   : etiquetas reales
        y_pred   : etiquetas predichas
        clases   : lista ordenada de clases únicas

        Retorna
        -------
        dict con keys 'F1-macro', 'AUC-ROC', 'Precisión'
        """
        f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_test, y_pred, average="macro",
                                zero_division=0)
        try:
            y_prob = modelo.predict_proba(X_test)
            if len(clases) == 2:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                y_bin = label_binarize(y_test, classes=clases)
                auc   = roc_auc_score(y_bin, y_prob,
                                       average="macro", multi_class="ovr")
        except AttributeError:
            auc = float("nan")

        return {
            "F1-macro":  round(f1, 4),
            "AUC-ROC":   round(auc, 4) if not np.isnan(auc) else float("nan"),
            "Precisión": round(prec, 4),
        }

    # =========================================================================
    # MÉTODO GENÉRICO DE EJECUCIÓN 
    # =========================================================================
    def ejecutar_modelo(self, nombre: str,
                         apply_smote: bool = False,
                         **kwargs) -> dict:
        """
        Entrena un modelo por nombre y devuelve métricas.

        Parámetros
        ----------
        nombre      : str  — 'KNN' | 'DT' | 'RF' | 'XG' | 'ADA'
        apply_smote : bool — si True aplica SMOTE antes de entrenar
                             (recomendado para datasets desbalanceados)
        **kwargs    : hiperparámetros del modelo

        Retorna
        -------
        dict con:
            • métricas clásicas de Supervisado
              (Matriz de Confusión, Precisión Global, Error Global,
               Precisión por categoría)
            • 'y_pred', 'y_test', 'X_test', 'modelo'
            • 'smote_aplicado' : bool
            • métricas del desbalance: 'F1-macro', 'AUC-ROC', 'Precisión'
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

        # Preparar datos con o sin SMOTE
        if apply_smote:
            X_train, X_test, y_train, y_test, y = self.preparar_datos_smote()
        else:
            X_train, X_test, y_train, y_test, y = self.preparar_datos()

        # Entrenar y predecir
        modelo  = builders[nombre](**kwargs)
        modelo.fit(X_train, y_train)
        y_pred  = self.predecir(modelo, X_test)
        indices = self.evaluar(y_test, y_pred, y)

         # Métricas de desbalance (F1-macro, AUC-ROC, Precisión macro)
        clases = sorted(list(np.unique(y)))
        mp = self.metricas_desbalance(modelo, X_test, y_test, y_pred, clases)

        # Componer resultado final
        indices.update({
            "y_pred":         y_pred,
            "y_test":         y_test,
            "X_test":         X_test,
            "modelo":         modelo,
            "smote_aplicado": apply_smote,
            **mp,             # F1-macro, AUC-ROC, Precisión
        })
        return indices

    # =========================================================================
    # BENCHMARK CLÁSICO — comparación de modelos con métricas estándar
    # =========================================================================
    def benchmark(self, clase_positiva=None,
                  clase_negativa=None) -> pd.DataFrame:
        """
        Ejecuta todos los algoritmos con hiperparámetros por defecto
        y devuelve tabla comparativa (PG, EG, PP, PN).
        Sin SMOTE — para compatibilidad con el flujo original de ml_studio.
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

    # =========================================================================
    # BENCHMARK DE DESBALANCE — comparación con/sin SMOTE
    # =========================================================================
    def benchmark_desbalance(self, apply_smote: bool = True) -> pd.DataFrame:
        """
        Benchmark enfocado en desbalance de clases: compara KNN, RF y
        XGBoost usando F1-macro, AUC-ROC y Precisión macro.

        Parámetros
        ----------
        apply_smote : bool — si True entrena con SMOTE (default: True)

        Retorna
        -------
        pd.DataFrame con filas = algoritmos,
        columnas = [F1-macro, AUC-ROC, Precisión, PG, EG, SMOTE]
        """
        configs = {
            "KNN":     dict(n_neighbors=3, algorithm="ball_tree"),
            "RF":      dict(n_estimators=200, min_samples_split=9,
                            max_depth=8),
            "XGBoost": dict(n_estimators=100, min_samples_split=5,
                            max_depth=1),
        }
        mapeo_builder = {"KNN": "KNN", "RF": "RF", "XGBoost": "XG"}

        filas = []
        for nombre, params in configs.items():
            r = self.ejecutar_modelo(mapeo_builder[nombre],
                                      apply_smote=apply_smote, **params)
            filas.append({
                "Algoritmo": nombre,
                "F1-macro":  r["F1-macro"],
                "AUC-ROC":   r["AUC-ROC"],
                "Precisión": r["Precisión"],
                "PG":        round(r["Precisión Global"], 4),
                "EG":        round(r["Error Global"], 4),
                "SMOTE":     apply_smote,
            })

        return pd.DataFrame(filas).set_index("Algoritmo")

    # =========================================================================
    # GRÁFICOS
    # =========================================================================
    @staticmethod
    def fig_matriz_confusion(y_test, y_pred, clases=None) -> plt.Figure:
        """
        Matriz de Confusión.

        Si se proveen `clases` (todas las clases originales del dataset),
        se fuerza esa lista como `labels` en confusion_matrix para evitar
        un desajuste de tamaño cuando alguna clase no aparece en y_test
        (común en datasets pequeños o muy desbalanceados, donde el split
        no garantiza presencia de todas las clases en el conjunto de test).
        """
        if clases is not None:
            etiquetas_idx = list(range(len(clases)))
            MC = confusion_matrix(y_test, y_pred, labels=etiquetas_idx)
            display_labels = clases
        else:
            MC = confusion_matrix(y_test, y_pred)
            display_labels = None

        n     = MC.shape[0]
        ancho = max(4, n * 1.4)
        alto  = max(3, n * 1.1)

        fig, ax = plt.subplots(figsize=(ancho, alto), dpi=100)       
        disp = ConfusionMatrixDisplay(confusion_matrix=MC,
                                       display_labels=display_labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Matriz de Confusión", fontsize=11)
        plt.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════════════
    # AGENTE CLASIFICACIÓN — 7 PASOS DEL AGENTE INTELIGENTE
    # ═══════════════════════════════════════════════════════════════════════

    def percibir(self, df: pd.DataFrame, variable_objetivo: str) -> dict:
        """
        PASO 1 — PERCEPCIÓN.

        Carga el dataset y registra las características del problema de
        clasificación: clases, distribución, balance y tipos de variables.
        """
        if variable_objetivo not in df.columns:
            raise ValueError(f"Variable objetivo '{variable_objetivo}' "
                             f"no encontrada en el DataFrame.")

        self._df             = df.copy().reset_index(drop=True)
        self._columna_target = variable_objetivo

        y            = df[variable_objetivo]
        dist_abs     = y.value_counts()
        dist_pct     = y.value_counts(normalize=True).mul(100).round(2)
        pct_max      = float(dist_pct.max())
        pct_min      = float(dist_pct.min())
        ratio        = pct_max / max(pct_min, 0.001)

        X            = df.drop(columns=[variable_objetivo])
        num_cols     = X.select_dtypes(include="number").columns.tolist()
        cat_cols     = X.select_dtypes(exclude="number").columns.tolist()

        self.conocimiento.update({
            "variable_objetivo":   variable_objetivo,
            "filas":               len(df),
            "columnas":            df.shape[1],
            "clases":              dist_abs.index.astype(str).tolist(),
            "cantidad_clases":     int(y.nunique()),
            "distribucion_abs":    dist_abs.to_dict(),
            "distribucion_pct":    dist_pct.to_dict(),
            "ratio_desbalance":    round(ratio, 2),
            "desbalanceado":       ratio > 3,
            "columnas_numericas":  num_cols,
            "columnas_categoricas": cat_cols,
            "nulos":               df.isnull().sum().to_dict(),
        })
        return self.conocimiento

    def analizar(self) -> dict:
        """
        PASO 2 — ANÁLISIS.

        Determina el subtipo del problema (binario/multiclase), calcula
        el porcentaje de nulos en predictoras y evalúa el desbalance.
        """
        n_cls   = self.conocimiento["cantidad_clases"]
        subtipo = "binaria" if n_cls == 2 else "multiclase"

        nulos   = self.conocimiento["nulos"]
        pct_nulos = {
            col: round(v / max(self.conocimiento["filas"], 1) * 100, 2)
            for col, v in nulos.items()
        }

        self.conocimiento.update({
            "subtipo":    subtipo,
            "pct_nulos":  pct_nulos,
        })
        return self.conocimiento

    def razonar(self) -> list:
        """
        PASO 3 — RAZONAMIENTO.

        Genera alertas y recomendaciones basadas en las características
        del problema detectadas en los pasos anteriores.
        """
        alertas         = []
        recomendaciones = []

        if self.conocimiento.get("desbalanceado"):
            ratio = self.conocimiento["ratio_desbalance"]
            alertas.append(
                f"⚠ ALERTA: Clases desbalanceadas (ratio {ratio}:1). "
                f"Se recomienda SMOTE y usar F1 / AUC como métricas principales."
            )
        else:
            recomendaciones.append(
                "Clases relativamente balanceadas. "
                "Accuracy puede ser una métrica útil."
            )

        if self.conocimiento["filas"] < 500:
            recomendaciones.append(
                f"Dataset pequeño ({self.conocimiento['filas']} filas). "
                "Se recomienda validación cruzada k-fold."
            )

        if self.conocimiento["columnas_numericas"]:
            recomendaciones.append(
                "Variables numéricas detectadas. Aplicar StandardScaler "
                "para KNN, SVM y Regresión Logística."
            )

        if self.conocimiento["columnas_categoricas"]:
            recomendaciones.append(
                "Variables categóricas detectadas. Aplicar OneHotEncoder."
            )

        if self.conocimiento["subtipo"] == "multiclase":
            recomendaciones.append(
                "Problema multiclase. Se usará AUC macro OvR y F1 ponderado."
            )

        for col, pct in self.conocimiento.get("pct_nulos", {}).items():
            if col == self.conocimiento["variable_objetivo"]:
                continue
            if pct > 20:
                alertas.append(
                    f"⚠ '{col}' tiene {pct}% de nulos en las predictoras."
                )

        self.conocimiento["alertas"]         = alertas
        self.conocimiento["recomendaciones"] = recomendaciones
        return alertas + recomendaciones

    def decidir_modelos(self) -> list:
        """
        PASO 4 — DECISIÓN.

        Selecciona configuración de entrenamiento según las características
        del problema (SMOTE, métrica principal, modelos a usar).
        """
        apply_smote      = self.conocimiento.get("desbalanceado", False)
        metrica_principal = "AUC" if self.conocimiento.get("desbalanceado") else "Accuracy"

        modelos = [
            "KNN", "Decision Tree", "Random Forest",
            "AdaBoost", "XGBoost", "Naive Bayes",
            "Logistic Regression", "SVM",
        ]

        self.conocimiento.update({
            "apply_smote":        apply_smote,
            "metrica_principal":  metrica_principal,
            "modelos_seleccionados": modelos,
        })
        return modelos

    def inferir_tipo_problema(self) -> str:
        """
        PASO 5 — INFERENCIA.

        Confirma el tipo y subtipo de problema de clasificación.
        """
        subtipo = self.conocimiento.get("subtipo", "binaria")
        tipo    = f"clasificacion_{subtipo}"
        self.conocimiento["tipo_confirmado"] = tipo
        return tipo

    def comunicar(self) -> dict:
        """
        PASO 6 — COMUNICACIÓN CON EL AGENTE COORDINADOR.

        Construye el mensaje estandarizado MAS con los resultados
        del análisis de clasificación.
        """
        mensaje = {
            "agente":    "AgenteClasificacion",
            "tipo":      "REPORTE_CLASIFICACION",
            "timestamp": datetime.now().isoformat(),
            "contenido": {
                "variable_objetivo":   self.conocimiento.get("variable_objetivo"),
                "subtipo":             self.conocimiento.get("subtipo"),
                "cantidad_clases":     self.conocimiento.get("cantidad_clases"),
                "desbalanceado":       self.conocimiento.get("desbalanceado"),
                "ratio_desbalance":    self.conocimiento.get("ratio_desbalance"),
                "apply_smote":         self.conocimiento.get("apply_smote"),
                "metrica_principal":   self.conocimiento.get("metrica_principal"),
                "mejor_modelo":        self.conocimiento.get("mejor_modelo", "N/A"),
                "valor_metrica":       self.conocimiento.get("valor_metrica", float("nan")),
                "modelos_evaluados":   self.conocimiento.get("modelos_seleccionados", []),
                "alertas":             self.conocimiento.get("alertas", []),
                "recomendaciones":     self.conocimiento.get("recomendaciones", []),
                "errores_etapas":      self._errores,
            },
        }
        self.mensajes.append(mensaje)
        return mensaje

    def ejecutar(self, df: pd.DataFrame,
                 variable_objetivo: str,
                 plan_eda: dict = None) -> dict:
        """
        PASO 7 — EJECUCIÓN COMPLETA DEL AGENTE.

        Orquesta los 6 pasos con try/except individual por etapa.
        Entrena todos los modelos y genera las figuras del mejor modelo.

        Retorna dict con: mensaje, resultados_df, figuras, conocimiento.
        """
        # ── Pre-limpieza antes de pasar al agente ────────────────────────
        try:
            if _limpiar is not None:
                df, _elim, _imp = _limpiar(df, plan_eda=plan_eda,
                                            variable_objetivo=variable_objetivo)
                if _elim:
                    self.conocimiento["pre_eliminadas"] = _elim
                    self.conocimiento["pre_imputadas"]  = _imp
        except Exception as e:
            self._errores.append({"paso": "pre_limpieza", "error": str(e)})

        for nombre_paso, fn, args in [
            ("percibir",            self.percibir,            (df, variable_objetivo)),
            ("analizar",            self.analizar,            ()),
            ("razonar",             self.razonar,             ()),
            ("decidir_modelos",     self.decidir_modelos,     ()),
            ("inferir_tipo_problema", self.inferir_tipo_problema, ()),
        ]:
            try:
                fn(*args)
            except Exception as e:
                self._errores.append({"paso": nombre_paso, "error": str(e)})

        # ── Entrenamiento ─────────────────────────────────────────────────
        resultados_df = None
        figuras       = {}
        try:
            apply_smote   = self.conocimiento.get("apply_smote", False)
            resultados_df = self.benchmark_completo(apply_smote=apply_smote)

            # Seleccionar mejor modelo por métrica principal
            metrica = self.conocimiento.get("metrica_principal", "Accuracy")
            col_ord  = metrica if metrica in resultados_df.columns else "Accuracy"

            df_validos = resultados_df.dropna(subset=[col_ord])

            # Fallback: si la métrica principal (ej. AUC) quedó NaN para
            # todos los modelos —por ejemplo porque una clase no tiene
            # suficientes muestras para calcular AUC— se reintenta con
            # Accuracy, que casi siempre puede calcularse.
            if df_validos.empty and col_ord != "Accuracy" \
                    and "Accuracy" in resultados_df.columns:
                self._errores.append({
                    "paso": "benchmark_completo_metrica_fallback",
                    "error": f"La métrica principal '{col_ord}' no pudo "
                             f"calcularse para ningún modelo (todos NaN). "
                             f"Se usa Accuracy como métrica de respaldo "
                             f"para seleccionar el mejor modelo.",
                })
                col_ord    = "Accuracy"
                df_validos = resultados_df.dropna(subset=[col_ord])

            if not df_validos.empty:
                mejor_fila   = df_validos.loc[df_validos[col_ord].idxmax()]
                mejor_modelo = mejor_fila["Modelo"]
                valor_metrica = round(float(mejor_fila[col_ord]), 4)
            else:
                mejor_modelo  = "N/A"
                valor_metrica = float("nan")

            self.conocimiento["mejor_modelo"]      = mejor_modelo
            self.conocimiento["valor_metrica"]     = valor_metrica
            self.conocimiento["metrica_principal"] = col_ord

        except Exception as e:
            self._errores.append({"paso": "benchmark_completo", "error": str(e)})

        # ── Figura: distribución de clases ────────────────────────────────
        try:

            y       = df[variable_objetivo]
            dist    = y.value_counts()
            paleta  = sns.color_palette("pastel", n_colors=len(dist))
            fig_cls, ax = plt.subplots(figsize=(5, 3), dpi=100)
            bars = ax.bar(dist.index.astype(str), dist.values,
                          color=paleta, edgecolor="white", linewidth=0.8)
            pcts = y.value_counts(normalize=True).mul(100).round(1)
            for bar, pct in zip(bars, pcts.values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(dist.values)*0.01,
                        f"{pct}%", ha="center", va="bottom",
                        fontsize=8, color="#334155")
            ax.set_title(f"Distribución: {variable_objetivo}", fontsize=11)
            ax.set_xlabel(variable_objetivo, fontsize=9)
            ax.set_ylabel("Frecuencia", fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            figuras["distribucion_clases"] = fig_cls
            plt.close(fig_cls)
        except Exception as e:
            self._errores.append({"paso": "fig_distribucion", "error": str(e)})

        # ── Figura: benchmark comparativo ─────────────────────────────────
        try:
            if resultados_df is not None and not resultados_df.empty:

                metricas_plot = [c for c in ["Accuracy","F1","AUC"]
                                 if c in resultados_df.columns]

                if not metricas_plot:
                    self._errores.append({
                        "paso": "fig_benchmark",
                        "error": "El benchmark no contiene ninguna de las "
                                 "métricas esperadas (Accuracy, F1, AUC). "
                                 "No se puede generar el gráfico comparativo.",
                    })
                else:
                    df_plot = (resultados_df.set_index("Modelo")
                               [metricas_plot].dropna(how="all"))

                    if df_plot.empty:
                        self._errores.append({
                            "paso": "fig_benchmark",
                            "error": "Ningún modelo produjo métricas válidas "
                                     "(todas las filas son NaN), posiblemente "
                                     "porque todos los entrenamientos fallaron "
                                     "en benchmark_completo. No se puede "
                                     "generar el gráfico comparativo. Revise "
                                     "'errores_etapas' para más detalle.",
                        })
                    else:
                        fig_bm, ax_bm = plt.subplots(figsize=(7, 4), dpi=100)
                        df_plot.plot(kind="bar", ax=ax_bm, colormap="Set2",
                                     width=0.7, edgecolor="white")
                        ax_bm.set_title("Comparativa de Modelos", fontsize=11)
                        ax_bm.set_ylabel("Score", fontsize=9)
                        ax_bm.set_ylim(0, 1.1)
                        ax_bm.legend(fontsize=8)
                        ax_bm.tick_params(axis="x", rotation=30)
                        ax_bm.grid(axis="y", linestyle="--", alpha=0.4)
                        ax_bm.spines[["top","right"]].set_visible(False)
                        plt.tight_layout()
                        figuras["benchmark_barras"] = fig_bm
                        plt.close(fig_bm)
        except Exception as e:
            self._errores.append({"paso": "fig_benchmark", "error": str(e)})

        # ── Figura: matriz de confusión del mejor modelo ──────────────────
        try:
            if (resultados_df is not None and
                    self.conocimiento.get("mejor_modelo", "N/A") != "N/A"):

                import numpy as _np
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing   import LabelEncoder, StandardScaler
                from sklearn.pipeline        import Pipeline
                from sklearn.compose         import ColumnTransformer
                from sklearn.preprocessing   import OneHotEncoder

                df_clean = df.dropna().copy().reset_index(drop=True)
                X = df_clean.drop(columns=[variable_objetivo])
                y = df_clean[variable_objetivo]
                le = LabelEncoder(); y_enc = le.fit_transform(y)

                # Validar tamaño mínimo por clase antes de estratificar
                # (mismo criterio que benchmark_completo).
                _, conteos_cm = _np.unique(y_enc, return_counts=True)
                usar_stratify_cm = y_enc if conteos_cm.min() >= 2 else None
                if usar_stratify_cm is None:
                    self._errores.append({
                        "paso": "fig_matriz_confusion_split",
                        "error": "Alguna clase tiene menos de 2 registros; "
                                 "se usa split sin estratificar para la "
                                 "matriz de confusión.",
                    })

                num_c = X.select_dtypes(include="number").columns.tolist()
                cat_c = X.select_dtypes(exclude="number").columns.tolist()
                transformers = []
                if num_c: transformers.append(("num", StandardScaler(), num_c))
                if cat_c: transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_c))
                prep = ColumnTransformer(transformers) if transformers else "passthrough"

                nombre_mejor = self.conocimiento["mejor_modelo"]
                modelos_map  = {
                    "KNN": self._build_knn(),
                    "Decision Tree": self._build_dt(),
                    "Random Forest": self._build_rf(),
                    "XGBoost": self._build_xg(),
                    "AdaBoost": self._build_ada(),
                }
                from sklearn.naive_bayes  import GaussianNB
                from sklearn.linear_model import LogisticRegression
                from sklearn.svm          import SVC
                modelos_map.update({
                    "Naive Bayes":        GaussianNB(),
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "SVM":                SVC(probability=True, random_state=42),
                })

                modelo_obj = modelos_map.get(nombre_mejor)
                if modelo_obj is not None:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y_enc, test_size=self._test_size,
                        random_state=42, stratify=usar_stratify_cm)
                    pipe = Pipeline([("prep", prep), ("clf", modelo_obj)])
                    pipe.fit(X_tr, y_tr)
                    y_pred = pipe.predict(X_te)
                    fig_cm = self.fig_matriz_confusion(
                        y_te, y_pred, clases=le.classes_.astype(str))
                    figuras["matriz_confusion"] = fig_cm

        except Exception as e:
            self._errores.append({"paso": "fig_matriz_confusion", "error": str(e)})

        try:
            mensaje = self.comunicar()
        except Exception as e:
            self._errores.append({"paso": "comunicar", "error": str(e)})
            mensaje = {}

        return {
            "mensaje":       mensaje,
            "resultados_df": resultados_df,
            "figuras":       figuras,
            "conocimiento":  self.conocimiento,
        }