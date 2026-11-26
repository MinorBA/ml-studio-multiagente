# =============================================================================
# backend/regresion.py
# CLASE: Regresion  (hereda de Supervisado)
# ─────────────────────────────────────────────────────────────────────────────
# Modelos: Lineal Simple/Múltiple, Lasso(CV), Ridge(CV), SVM, Árbol,
#          Random Forest, Gradient Boosting.
# =============================================================================

import math
import numpy as np
import pandas as pd
from datetime import datetime
try:
    from shared_utils import limpiar_df_para_agente as _limpiar
except ImportError:
    _limpiar = None

from sklearn.linear_model   import (LinearRegression, Lasso, LassoCV,
                                     Ridge, RidgeCV)
from sklearn              import linear_model
from sklearn.svm          import SVR
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline      import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from backend.supervisado import Supervisado

try:
    from xgboost import XGBRegressor as _XGBRegressor
    _XGBOOST_OK = True
except ImportError:
    _XGBOOST_OK = False

# Visualización — imports al nivel de módulo para evitar errores en runtime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False


class Regresion(Supervisado):
    """
    Regresión supervisada — hereda de Supervisado.

    Uso típico
    ----------
    reg = Regresion(df, columna_target="precio")
    X_train, X_test, y_train, y_test, _ = reg.preparar_datos()
    tabla = reg.comparacion_final(X_train, X_test, y_train, y_test)
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

    # ────────────────────── MÉTRICAS ───────────────────────────────────────
    @staticmethod
    def errores(y_true, y_pred) -> pd.DataFrame:
        """RMSE, MAE, MSE, Error Relativo (ER), R², MAPE."""        
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        rmse   = math.sqrt(np.mean((y_true - y_pred) ** 2))
        mse    = np.mean((y_true - y_pred) ** 2)
        mae    = np.mean(np.abs(y_true - y_pred))
        er     = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-15)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2     = 1 - ss_res / (ss_tot + 1e-15)
        # MAPE: ignorar filas donde y_true == 0 para evitar div/0
        mask   = y_true != 0
        mape   = (np.mean(np.abs((y_true[mask] - y_pred[mask])
                                  / y_true[mask])) * 100
                  if mask.sum() > 0 else float("nan"))
        return pd.DataFrame(
            {"Tipo":  ["RMSE",  "MSE",  "MAE", "ER", "R2",   "MAPE"],
             "Error": [rmse,    mse,    mae,   er,   r2,    mape]}
        )

    def comparacion_completa(self, Xtr, Xte, ytr, yte) -> pd.DataFrame:
        """
        Tabla con TODOS los modelos, columnas estandarizadas para el MAS:
        Modelo, RMSE, MSE, MAE, R², MAPE.

        Reemplaza a comparacion_final() que usaba columnas distintas.
        """
        modelos_run = [
            ("Múltiple",       self.RegLinMult(Xtr, Xte, ytr, yte)),
            ("Lasso CV",       self.RegLassoCV(Xtr, Xte, ytr, yte)),
            ("Ridge CV",       self.RegRidgeCV(Xtr, Xte, ytr, yte)),
            ("Árbol",          self.DecisionTreeReg(Xtr, Xte, ytr, yte)),
            ("Bosque",         self.RandomForestReg(Xtr, Xte, ytr, yte)),
            ("XGBoost" if _XGBOOST_OK else "Gradient Boost",
             self.XGBoostingReg(Xtr, Xte, ytr, yte)),
        ]
        svm_rbf, svm_lin, svm_poly = self.SVM(Xtr, Xte, ytr, yte)
        modelos_run += [
            ("SVM rbf",    svm_rbf),
            ("SVM linear", svm_lin),
            ("SVM poly",   svm_poly),
        ]

        filas = []
        for nombre, df_err in modelos_run:
            err_dict = dict(zip(df_err["Tipo"], df_err["Error"]))
            filas.append({
                "Modelo": nombre,
                "RMSE":   round(err_dict.get("RMSE",  float("nan")), 4),
                "MSE":    round(err_dict.get("MSE",   float("nan")), 4),
                "MAE":    round(err_dict.get("MAE",   float("nan")), 4),
                "R2":     round(err_dict.get("R2",    float("nan")), 4),
                "MAPE":   round(err_dict.get("MAPE",  float("nan")), 4),
            })
        return pd.DataFrame(filas)

    # ────────────────────── MODELOS INDIVIDUALES ───────────────────────────

    def RegLinMult(self, X_train, X_test, y_train, y_test) -> pd.DataFrame:
        m = LinearRegression().fit(X_train, y_train)
        return self.errores(y_test, m.predict(X_test))

    def RegLasso(self, X_train, X_test, y_train, y_test,
                 scale=True, alpha=0.1) -> pd.DataFrame:
        X_train, X_test = self._escalar(X_train, X_test, scale)
        m = linear_model.Lasso(alpha=alpha).fit(X_train, y_train)
        return self.errores(y_test, m.predict(X_test))

    def RegLassoCV(self, X_train, X_test, y_train, y_test,
                   scale=True) -> pd.DataFrame:
        X_train, X_test = self._escalar(X_train, X_test, scale)
        cv = LassoCV(alphas=np.logspace(-6, 6, 200), cv=10).fit(X_train, y_train)
        m  = Lasso(alpha=cv.alpha_).fit(X_train, y_train)
        return self.errores(y_test, m.predict(X_test))

    def RegRidge(self, X_train, X_test, y_train, y_test,
                 scale=True, alpha=1.0) -> pd.DataFrame:
        X_train, X_test = self._escalar(X_train, X_test, scale)
        m = Ridge(alpha=alpha).fit(X_train, y_train)
        return self.errores(y_test, m.predict(X_test))

    def RegRidgeCV(self, X_train, X_test, y_train, y_test,
                   scale=True) -> pd.DataFrame:
        X_train, X_test = self._escalar(X_train, X_test, scale)
        cv = RidgeCV(alphas=np.logspace(-10, 2, 200),
                     fit_intercept=True).fit(X_train, y_train)
        m  = Ridge(alpha=cv.alpha_).fit(X_train, y_train)
        return self.errores(y_test, m.predict(X_test))

    def SVM(self, X_train, X_test, y_train, y_test):
        """Retorna tupla (err_rbf, err_linear, err_poly)."""
        resultados = {}
        for nombre, svr in [
            ("rbf",    SVR(kernel="rbf", C=100, epsilon=0.1)),
            ("linear", SVR(kernel="linear", C=100, epsilon=0.1)),
            ("poly",   SVR(kernel="poly", C=100, degree=3, epsilon=0.1)),
        ]:
            pipe = make_pipeline(StandardScaler(), svr)
            pipe.fit(X_train, y_train)
            resultados[nombre] = self.errores(y_test, pipe.predict(X_test))
        return resultados["rbf"], resultados["linear"], resultados["poly"]

    def DecisionTreeReg(self, X_train, X_test, y_train, y_test,
                         max_depth=3) -> pd.DataFrame:
        m = DecisionTreeRegressor(max_depth=max_depth,
                                   random_state=123).fit(X_train, y_train)
        return self.errores(y_test, m.predict(X_test))

    def RandomForestReg(self, X_train, X_test, y_train, y_test,
                         max_depth=2) -> pd.DataFrame:
        m = RandomForestRegressor(max_depth=max_depth,
                                   random_state=0).fit(X_train, y_train)
        return self.errores(y_test, m.predict(X_test))

    def XGBoostingReg(self, X_train, X_test, y_train, y_test,  
                       n_estimators=100, max_depth=4,
                       min_samples_split=5) -> pd.DataFrame:
        """Usa XGBRegressor si xgboost instalado, fallback a GradientBoosting."""
        if _XGBOOST_OK:
            m = _XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                              learning_rate=0.1, random_state=42, verbosity=0)
        else:
            m = GradientBoostingRegressor(n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          min_samples_split=min_samples_split)
        m.fit(X_train, y_train)
        return self.errores(y_test, m.predict(X_test))

    # ────────────────────── TABLAS COMPARATIVAS ────────────────────────────

    def comparacion_final(self, Xtr, Xte, ytr, yte) -> pd.DataFrame:
        """Tabla con TODOS los modelos de regresión."""
        e1 = self.RegLinMult(Xtr, Xte, ytr, yte)
        e2 = self.RegLassoCV(Xtr, Xte, ytr, yte)
        e3 = self.RegRidgeCV(Xtr, Xte, ytr, yte)
        e4, e5, e6 = self.SVM(Xtr, Xte, ytr, yte)
        e7 = self.DecisionTreeReg(Xtr, Xte, ytr, yte)
        e8 = self.RandomForestReg(Xtr, Xte, ytr, yte)
        e9 = self.XGBoostingReg(Xtr, Xte, ytr, yte)
        return self._tabla(
            [e1, e2, e3, e4, e5, e6, e7, e8, e9],
            ["Múltiple", "Lasso CV", "Ridge CV",
             "SVM rbf", "SVM linear", "SVM poly",
             "Árbol", "Bosque", "Gradient Boosting"])

    # ────────────────────── HELPERS ────────────────────────────────────────
    @staticmethod
    def _escalar(X_train, X_test, scale):
        if scale:
            # Si la matriz es sparse (OneHotEncoder produce sparse)
            # StandardScaler no puede centrar (with_mean=False)
            import scipy.sparse as _sp
            is_sparse = _sp.issparse(X_train)
            sc = StandardScaler(with_mean=not is_sparse)
            X_train = sc.fit_transform(X_train)
            X_test  = sc.transform(X_test)
        return X_train, X_test

    @staticmethod
    def _tabla(lista_err, indices) -> pd.DataFrame:
        return pd.DataFrame(
            [e["Error"].values for e in lista_err],
            columns=["RMSE", "MAE", "ER", "R2"],
            index=indices)


    # ═══════════════════════════════════════════════════════════════════════
    # AGENTE REGRESIÓN — 7 PASOS DEL AGENTE INTELIGENTE
    # ═══════════════════════════════════════════════════════════════════════

    def percibir(self, df: pd.DataFrame, variable_objetivo: str) -> dict:
        """
        PASO 1 — PERCEPCIÓN.

        Carga el dataset y registra las características del problema de
        regresión: estadísticas del objetivo, tipos de variables y nulos.
        """
        if variable_objetivo not in df.columns:
            raise ValueError(f"Variable objetivo '{variable_objetivo}' "
                             f"no encontrada en el DataFrame.")

        self._df             = df.copy().reset_index(drop=True)
        self._columna_target = variable_objetivo

        y        = df[variable_objetivo].dropna()
        X        = df.drop(columns=[variable_objetivo])
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()

        self.conocimiento.update({
            "variable_objetivo":   variable_objetivo,
            "filas":               len(df),
            "columnas":            df.shape[1],
            "dtype_objetivo":      str(df[variable_objetivo].dtype),
            "media_y":             round(float(y.mean()), 4),
            "mediana_y":           round(float(y.median()), 4),
            "std_y":               round(float(y.std()), 4),
            "min_y":               round(float(y.min()), 4),
            "max_y":               round(float(y.max()), 4),
            "skewness_y":          round(float(y.skew()), 4),
            "columnas_numericas":  num_cols,
            "columnas_categoricas": cat_cols,
            "nulos":               df.isnull().sum().to_dict(),
        })
        return self.conocimiento

    def analizar(self) -> dict:
        """
        PASO 2 — ANÁLISIS.

        Calcula la correlación de las predictoras con el objetivo,
        detecta multicolinealidad y analiza la distribución del target.
        """
        df     = self._df.copy()
        target = self._columna_target
        num_c  = self.conocimiento["columnas_numericas"]

        # Correlación predictoras con objetivo
        corr_con_y = {}
        if num_c:
            df_num = df[num_c + [target]].dropna()
            corr_con_y = (df_num.corr()[target]
                          .drop(target, errors="ignore")
                          .abs().sort_values(ascending=False)
                          .round(4).to_dict())

        # Correlación media entre predictoras (multicolinealidad)
        corr_media = 0.0
        if len(num_c) >= 2:
            corr_arr = df[num_c].dropna().corr().abs().to_numpy(copy=True)
            np.fill_diagonal(corr_arr, 0)
            corr_media = round(float(corr_arr.mean()), 4)

        self.conocimiento.update({
            "correlacion_con_objetivo": corr_con_y,
            "correlacion_media_predictoras": corr_media,
            "objetivo_sesgado": abs(self.conocimiento["skewness_y"]) > 1,
        })
        return self.conocimiento

    def razonar(self) -> list:
        """
        PASO 3 — RAZONAMIENTO.

        Genera alertas y recomendaciones basadas en las características
        del problema y las variables detectadas.
        """
        alertas         = []
        recomendaciones = []
        n               = self.conocimiento["filas"]
        skew            = self.conocimiento["skewness_y"]
        corr_m          = self.conocimiento["correlacion_media_predictoras"]

        if self.conocimiento["objetivo_sesgado"]:
            alertas.append(
                f"⚠ Variable objetivo muy sesgada (skewness={skew}). "
                f"Se recomienda transformación log(y+1) antes del entrenamiento."
            )

        if corr_m > 0.7:
            alertas.append(
                f"⚠ Alta multicolinealidad entre predictoras "
                f"(correlación media={corr_m}). "
                f"Se recomienda Lasso o Ridge sobre Regresión Lineal simple."
            )

        if n < 500:
            recomendaciones.append(
                f"Dataset pequeño ({n} filas). "
                f"Se recomienda validación cruzada k-fold."
            )
            estrategia = "GridSearchCV"
        else:
            recomendaciones.append(
                f"Dataset con {n} filas, suficiente para búsqueda genética. "
                f"Se usarán GridSearchCV y GASearchCV para comparar estrategias."
            )
            estrategia = "GridSearchCV + GASearchCV"

        if self.conocimiento["columnas_categoricas"]:
            recomendaciones.append(
                "Variables categóricas detectadas. "
                "Se codificarán con OneHotEncoder antes del entrenamiento."
            )

        top3 = list(self.conocimiento["correlacion_con_objetivo"].items())[:3]
        if top3:
            recomendaciones.append(
                "Variables más correlacionadas con el objetivo: " +
                ", ".join(f"{c} ({v:.2f})" for c, v in top3) + "."
            )

        self.conocimiento["alertas"]              = alertas
        self.conocimiento["recomendaciones"]      = recomendaciones
        self.conocimiento["estrategia_busqueda"]  = estrategia
        return alertas + recomendaciones

    def decidir_modelos(self) -> list:
        """
        PASO 4 — DECISIÓN.

        Selecciona la estrategia de búsqueda de hiperparámetros y los
        modelos a entrenar según las características del dataset.
        """
        n = self.conocimiento["filas"]
        usar_genetico = n >= 500

        modelos = [
            "Múltiple", "Lasso CV", "Ridge CV",
            "SVR rbf", "SVR linear", "SVR poly",
            "Árbol", "Bosque", "Gradient Boost",
        ]

        self.conocimiento.update({
            "usar_genetico":      usar_genetico,
            "modelos_seleccionados": modelos,
        })
        return modelos

    def inferir_tipo_problema(self) -> str:
        """
        PASO 5 — INFERENCIA.

        Confirma el tipo de problema y detecta si es regresión normal
        o con objetivo sesgado.
        """
        subtipo = ("regresion_sesgada"
                   if self.conocimiento.get("objetivo_sesgado")
                   else "regresion_normal")
        self.conocimiento["subtipo"] = subtipo
        return f"regresion_{subtipo}"

    def comunicar(self) -> dict:
        """
        PASO 6 — COMUNICACIÓN CON EL AGENTE COORDINADOR.

        Construye el mensaje estandarizado MAS con los resultados
        del análisis de regresión.
        """
        mensaje = {
            "agente":    "AgenteRegresion",
            "tipo":      "REPORTE_REGRESION",
            "timestamp": datetime.now().isoformat(),
            "contenido": {
                "variable_objetivo":      self.conocimiento.get("variable_objetivo"),
                "subtipo":                self.conocimiento.get("subtipo"),
                "skewness_y":             self.conocimiento.get("skewness_y"),
                "objetivo_sesgado":       self.conocimiento.get("objetivo_sesgado"),
                "estrategia_busqueda":    self.conocimiento.get("estrategia_busqueda"),
                "usar_genetico":          self.conocimiento.get("usar_genetico"),
                "mejor_modelo":           self.conocimiento.get("mejor_modelo", "N/A"),
                "mejor_rmse":             self.conocimiento.get("mejor_rmse",
                                                                float("nan")),
                "mejor_r2":               self.conocimiento.get("mejor_r2",
                                                                float("nan")),
                "modelos_evaluados":      self.conocimiento.get(
                                              "modelos_seleccionados", []),
                "alertas":                self.conocimiento.get("alertas", []),
                "recomendaciones":        self.conocimiento.get(
                                              "recomendaciones", []),
                "mejores_params_grid":    self.conocimiento.get(
                                              "mejores_params_grid", {}),
                "mejores_params_gen":     self.conocimiento.get(
                                              "mejores_params_gen", {}),
                "errores_etapas":         self._errores,
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
        Entrena todos los modelos, busca hiperparámetros con GridSearchCV
        (y GASearchCV si el dataset es suficientemente grande) y genera
        las figuras comparativas.

        Retorna dict con: mensaje, resultados_df, figuras, conocimiento.
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing   import StandardScaler, OneHotEncoder
        from sklearn.compose         import ColumnTransformer

        # ── Pre-limpieza ──────────────────────────────────────────────────
        try:
            if _limpiar is not None:
                df, _elim, _imp = _limpiar(df, plan_eda=plan_eda,
                                            variable_objetivo=variable_objetivo)
                if _elim:
                    self.conocimiento["pre_eliminadas"] = _elim
                    self.conocimiento["pre_imputadas"]  = _imp
        except Exception as e:
            self._errores.append({"paso": "pre_limpieza", "error": str(e)})

        # ── Pasos 1-5 ─────────────────────────────────────────────────────
        for nombre_paso, fn, args in [
            ("percibir",              self.percibir,
             (df, variable_objetivo)),
            ("analizar",              self.analizar,              ()),
            ("razonar",               self.razonar,               ()),
            ("decidir_modelos",       self.decidir_modelos,       ()),
            ("inferir_tipo_problema", self.inferir_tipo_problema, ()),
        ]:
            try:
                fn(*args)
            except Exception as e:
                self._errores.append({"paso": nombre_paso, "error": str(e)})

        # ── Preparar datos para entrenamiento ─────────────────────────────
        resultados_df = None
        figuras       = {}

        try:
            df_clean = df.dropna().copy().reset_index(drop=True)
            X = df_clean.drop(columns=[variable_objetivo])
            y = df_clean[variable_objetivo].astype(float)

            # Transformación log si objetivo muy sesgado
            y_train_final = y
            if self.conocimiento.get("objetivo_sesgado"):
                y_train_final = np.log1p(y)
                self.conocimiento["log_transform_aplicado"] = True
            else:
                self.conocimiento["log_transform_aplicado"] = False

            num_c = X.select_dtypes(include="number").columns.tolist()
            cat_c = X.select_dtypes(exclude="number").columns.tolist()
            transformers = []
            if num_c:
                transformers.append(("num", StandardScaler(), num_c))
            if cat_c:
                transformers.append(
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_c))
            prep = (ColumnTransformer(transformers)
                    if transformers else "passthrough")

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_train_final, test_size=self._test_size, random_state=42)

            # Preprocesar
            if prep != "passthrough":
                X_tr_p = prep.fit_transform(X_tr)
                X_te_p = prep.transform(X_te)
            else:
                X_tr_p, X_te_p = X_tr.values, X_te.values

            # Convertir sparse a denso si hay columnas categóricas
            # (algunos modelos como SVM y comparacion_completa no aceptan sparse)
            import scipy.sparse as _sp
            if _sp.issparse(X_tr_p):
                X_tr_p = X_tr_p.toarray()
                X_te_p = X_te_p.toarray()

        except Exception as e:
            self._errores.append({"paso": "preparar_train_test", "error": str(e)})
            try:
                mensaje = self.comunicar()
            except Exception:
                mensaje = {}
            return {"mensaje": mensaje, "resultados_df": None,
                    "figuras": figuras, "conocimiento": self.conocimiento}

        # ── Entrenamiento con comparacion_completa (usa prep interno) ─────
        try:
            resultados_df = self.comparacion_completa(
                X_tr_p, X_te_p, y_tr.values, y_te.values)
        except Exception as e:
            self._errores.append({"paso": "comparacion_completa", "error": str(e)})

        # ── Seleccionar mejor modelo por RMSE ─────────────────────────────
        if resultados_df is not None and not resultados_df.empty:
            df_val = resultados_df.dropna(subset=["RMSE"])
            if not df_val.empty:
                mejor_fila  = df_val.loc[df_val["RMSE"].idxmin()]
                mejor_modelo = mejor_fila["Modelo"]
                mejor_rmse   = round(float(mejor_fila["RMSE"]), 4)
                mejor_r2     = round(float(mejor_fila["R2"]), 4)
                self.conocimiento["mejor_modelo"] = mejor_modelo
                self.conocimiento["mejor_rmse"]   = mejor_rmse
                self.conocimiento["mejor_r2"]     = mejor_r2

        # ── Búsqueda de hiperparámetros GridSearchCV en mejor modelo ──────
        try:
            mejor_nom = self.conocimiento.get("mejor_modelo", "Bosque")
            modelos_cv_map = {
                "Bosque":         RandomForestRegressor(random_state=42),
                ("XGBoost" if _XGBOOST_OK else "Gradient Boost"):
                    (_XGBRegressor(n_estimators=100, max_depth=3,
                                   learning_rate=0.1, random_state=42,
                                   verbosity=0)
                     if _XGBOOST_OK else GradientBoostingRegressor()),
                "Árbol":          DecisionTreeRegressor(random_state=42),
                "Múltiple":       LinearRegression(),
                "Lasso CV":       Lasso(),
                "Ridge CV":       Ridge(),
            }
            param_grids = {
                "Bosque":         {"n_estimators":[50,100],"max_depth":[3,5]},
                "Gradient Boost": {"n_estimators":[50,100],"max_depth":[3,5]},
                "Árbol":          {"max_depth":[3,5,7]},
                "Múltiple":       {},
                "Lasso CV":       {"alpha":[0.01,0.1,1.0]},
                "Ridge CV":       {"alpha":[0.1,1.0,10.0]},
            }
            modelo_cv = modelos_cv_map.get(mejor_nom,
                            RandomForestRegressor(random_state=42))
            param_grid = param_grids.get(mejor_nom, {})

            if param_grid:
                gs = GridSearchCV(modelo_cv, param_grid, cv=3,
                                  scoring="neg_root_mean_squared_error",
                                  n_jobs=-1)
                gs.fit(X_tr_p, y_tr.values)
                self.conocimiento["mejores_params_grid"] = gs.best_params_
                y_pred_gs = gs.predict(X_te_p)
                rmse_grid = float(np.sqrt(np.mean((y_te.values - y_pred_gs)**2)))
                self.conocimiento["rmse_grid"] = round(rmse_grid, 4)
            else:
                self.conocimiento["mejores_params_grid"] = {}
                self.conocimiento["rmse_grid"] = mejor_rmse

        except Exception as e:
            self._errores.append({"paso": "GridSearchCV", "error": str(e)})

        # ── Búsqueda genética GASearchCV (si dataset >= 500 filas) ────────
        self.conocimiento["rmse_gen"] = None
        if self.conocimiento.get("usar_genetico", False):
            try:
                from sklearn_genetic import GASearchCV
                from sklearn_genetic.space import Integer

                modelo_gen = RandomForestRegressor(random_state=42)
                param_gen  = {
                    "n_estimators": Integer(50, 150),
                    "max_depth":    Integer(2, 8),
                }
                ga = GASearchCV(
                    estimator=modelo_gen,
                    cv=3,
                    scoring="neg_root_mean_squared_error",
                    param_grid=param_gen,
                    population_size=10,
                    generations=5,
                    tournament_size=3,
                    elitism=True,
                    crossover_probability=0.8,
                    mutation_probability=0.1,
                    verbose=False,
                )
                ga.fit(X_tr_p, y_tr.values)
                self.conocimiento["mejores_params_gen"] = ga.best_params_
                y_pred_gen = ga.predict(X_te_p)
                rmse_gen   = float(np.sqrt(np.mean(
                    (y_te.values - y_pred_gen)**2)))
                self.conocimiento["rmse_gen"] = round(rmse_gen, 4)
            except ImportError:
                self.conocimiento["mejores_params_gen"] = {}
                recomendaciones = self.conocimiento.get("recomendaciones", [])
                recomendaciones.append(
                    "sklearn-genetic-opt no instalado. "
                    "Instalar con: pip install sklearn-genetic-opt")
                self.conocimiento["recomendaciones"] = recomendaciones
            except Exception as e:
                self._errores.append({"paso": "GASearchCV", "error": str(e)})

        # ── FIGURA 1: Gráfico RMSE comparativo Plotly ────────────────────
        try:
            if resultados_df is not None and not resultados_df.empty and _PLOTLY_OK:
                df_plot = resultados_df.dropna(subset=["RMSE"]).copy()

                if df_plot.empty:
                    self._errores.append({
                        "paso": "fig_rmse_plotly",
                        "error": "Ningún modelo produjo un RMSE válido "
                                 "(todas las filas son NaN). No se puede "
                                 "generar el gráfico comparativo de RMSE.",
                    })
                else:
                    modelos_names = df_plot["Modelo"].tolist()
                    rmse_base     = df_plot["RMSE"].tolist()

                    fig_rmse = go.Figure()
                    fig_rmse.add_trace(go.Bar(
                        name="Base", x=modelos_names, y=rmse_base,
                        marker_color="#94a3b8",
                    ))

                    rmse_grid_val = self.conocimiento.get("rmse_grid")
                    rmse_gen_val  = self.conocimiento.get("rmse_gen")
                    mejor_nom     = self.conocimiento.get("mejor_modelo", "")

                    if rmse_grid_val is not None:
                        y_grid = [rmse_grid_val if m == mejor_nom else None
                                  for m in modelos_names]
                        fig_rmse.add_trace(go.Bar(
                            name="GridSearch", x=modelos_names, y=y_grid,
                            marker_color="#ef4444",
                        ))
                    if rmse_gen_val is not None:
                        y_gen = [rmse_gen_val if m == mejor_nom else None
                                 for m in modelos_names]
                        fig_rmse.add_trace(go.Bar(
                            name="Genético", x=modelos_names, y=y_gen,
                            marker_color="#3b82f6",
                        ))

                    fig_rmse.update_layout(
                        title="Comparativa RMSE — Todos los Modelos",
                        xaxis_title="Modelo",
                        yaxis_title="RMSE",
                        barmode="group",
                        height=380,
                        margin=dict(l=40, r=20, t=50, b=100),
                        xaxis_tickangle=-35,
                        legend=dict(orientation="h", yanchor="bottom",
                                    y=1.02, xanchor="right", x=1),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                    )
                    figuras["rmse_comparativo"] = fig_rmse

        except Exception as e:
            self._errores.append({"paso": "fig_rmse_plotly", "error": str(e)})

        # ── FIGURA 2: R² comparativo ──────────────────────────────────────
        try:
            if resultados_df is not None and not resultados_df.empty and _PLOTLY_OK:
                df_r2 = resultados_df.dropna(subset=["R2"]).copy()

                if df_r2.empty:
                    self._errores.append({
                        "paso": "fig_r2_plotly",
                        "error": "Ningún modelo produjo un R² válido "
                                 "(todas las filas son NaN). No se puede "
                                 "generar el gráfico comparativo de R².",
                    })
                else:
                    fig_r2 = go.Figure(go.Bar(
                        x=df_r2["Modelo"],
                        y=df_r2["R2"],
                        marker_color=[
                            "#10b981" if v >= 0.7 else
                            "#f59e0b" if v >= 0.4 else "#ef4444"
                            for v in df_r2["R2"]
                        ],
                    ))
                    fig_r2.update_layout(
                        title="R² por Modelo",
                        xaxis_title="Modelo",
                        yaxis_title="R²",
                        height=350,
                        margin=dict(l=40, r=20, t=50, b=100),
                        xaxis_tickangle=-35,
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                    )
                    figuras["r2_comparativo"] = fig_r2
        except Exception as e:
            self._errores.append({"paso": "fig_r2_plotly", "error": str(e)})

        # ── FIGURA 3: Predicho vs Real del mejor modelo ───────────────────
        try:
            mejor_nom = self.conocimiento.get("mejor_modelo", "Bosque")
            modelos_pred = {
                "Múltiple":       LinearRegression(),
                "Lasso CV":       Lasso(),
                "Ridge CV":       Ridge(),
                "Árbol":          DecisionTreeRegressor(random_state=42),
                "Bosque":         RandomForestRegressor(
                                      n_estimators=100, random_state=42),
                ("XGBoost" if _XGBOOST_OK else "Gradient Boost"):
                    (_XGBRegressor(n_estimators=100, max_depth=3,
                                   learning_rate=0.1, random_state=42,
                                   verbosity=0)
                     if _XGBOOST_OK else GradientBoostingRegressor()),
            }
            modelo_pred = modelos_pred.get(mejor_nom, LinearRegression())
            modelo_pred.fit(X_tr_p, y_tr.values)
            y_pred_final = modelo_pred.predict(X_te_p)

            # Revertir log si se aplicó
            if self.conocimiento.get("log_transform_aplicado"):
                y_real_plot = np.expm1(y_te.values)
                y_pred_plot = np.expm1(y_pred_final)
            else:
                y_real_plot = y_te.values
                y_pred_plot = y_pred_final

            fig_pv, ax_pv = plt.subplots(figsize=(5, 4), dpi=100)
            ax_pv.scatter(y_real_plot, y_pred_plot,
                         alpha=0.5, color="#3b82f6", s=20)
            lims = [min(y_real_plot.min(), y_pred_plot.min()),
                    max(y_real_plot.max(), y_pred_plot.max())]
            ax_pv.plot(lims, lims, "r--", linewidth=1, label="Ideal")
            ax_pv.set_xlabel("Valor Real", fontsize=9)
            ax_pv.set_ylabel("Valor Predicho", fontsize=9)
            ax_pv.set_title(f"Predicho vs Real — {mejor_nom}", fontsize=11)
            ax_pv.legend(fontsize=8)
            ax_pv.grid(alpha=0.3)
            ax_pv.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            figuras["pred_vs_real"] = fig_pv
            plt.close(fig_pv)

        except Exception as e:
            self._errores.append({"paso": "fig_pred_vs_real", "error": str(e)})

        # ── FIGURA 4: Distribución del objetivo ──────────────────────────
        try:
            y_vals = df[variable_objetivo].dropna()
            fig_dist, ax_dist = plt.subplots(figsize=(5, 3), dpi=100)
            ax_dist.hist(y_vals, bins=30, color="#6366f1",
                        edgecolor="white", linewidth=0.5, alpha=0.85)
            ax_dist.set_title(f"Distribución: {variable_objetivo}", fontsize=11)
            ax_dist.set_xlabel(variable_objetivo, fontsize=9)
            ax_dist.set_ylabel("Frecuencia", fontsize=9)
            ax_dist.axvline(float(y_vals.mean()), color="#ef4444",
                           linestyle="--", linewidth=1, label="Media")
            ax_dist.axvline(float(y_vals.median()), color="#10b981",
                           linestyle="--", linewidth=1, label="Mediana")
            ax_dist.legend(fontsize=8)
            ax_dist.grid(alpha=0.3)
            ax_dist.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            figuras["distribucion_objetivo"] = fig_dist
            plt.close(fig_dist)
        except Exception as e:
            self._errores.append({"paso": "fig_distribucion", "error": str(e)})

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
