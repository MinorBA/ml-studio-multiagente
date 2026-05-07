# =============================================================================
# backend/regresion.py
# CLASE: Regresion  (hereda de Supervisado)
# ─────────────────────────────────────────────────────────────────────────────
# Modelos: Lineal Simple/Múltiple, Lasso(CV), Ridge(CV), SVM, Árbol,
#          Random Forest, Gradient Boosting
# =============================================================================

import math
import statistics
import numpy as np
import pandas as pd

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

    # ────────────────────── MÉTRICAS ───────────────────────────────────────
    @staticmethod
    def errores(y_true, y_pred) -> pd.DataFrame:
        """RMSE, MAE, Error Relativo (ER), R²."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
        mae  = np.mean(np.abs(y_true - y_pred))
        er   = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-15)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-15)
        return pd.DataFrame({"Tipo": ["RMSE", "MAE", "ER", "R2"],
                              "Error": [rmse, mae, er, r2]})

    @staticmethod
    def res_var_pred(y) -> dict:
        return {"Máximo": float(np.max(y)),
                "Cuartiles": statistics.quantiles(y),
                "Mínimo": float(np.min(y))}

    # ────────────────────── MODELOS INDIVIDUALES ───────────────────────────
    def RegLinSimp(self, X_train, X_test, y_train, y_test) -> pd.DataFrame:
        modelo = LinearRegression().fit(
            np.asarray(X_train).reshape(-1, 1), y_train)
        return self.errores(y_test, modelo.predict(np.asarray(X_test)))

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
                       n_estimators=500, max_depth=4,
                       min_samples_split=5) -> pd.DataFrame:
        m = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split).fit(X_train, y_train)
        return self.errores(y_test, m.predict(X_test))

    # ────────────────────── TABLAS COMPARATIVAS ────────────────────────────
    def comparacion_lasso(self, Xtr, Xte, ytr, yte, scale=True) -> pd.DataFrame:
        e1 = self.RegLasso(Xtr, Xte, ytr, yte, scale)
        e2 = self.RegLassoCV(Xtr, Xte, ytr, yte, scale)
        return self._tabla([e1, e2], ["Lasso", "Lasso CV"])

    def comparacion_ridge(self, Xtr, Xte, ytr, yte, scale=True) -> pd.DataFrame:
        e1 = self.RegRidge(Xtr, Xte, ytr, yte, scale)
        e2 = self.RegRidgeCV(Xtr, Xte, ytr, yte, scale)
        return self._tabla([e1, e2], ["Ridge", "Ridge CV"])

    def comparacion_svm(self, Xtr, Xte, ytr, yte) -> pd.DataFrame:
        e1, e2, e3 = self.SVM(Xtr, Xte, ytr, yte)
        return self._tabla([e1, e2, e3],
                            ["SVM rbf", "SVM linear", "SVM poly"])

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
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test  = sc.transform(X_test)
        return X_train, X_test

    @staticmethod
    def _tabla(lista_err, indices) -> pd.DataFrame:
        return pd.DataFrame(
            [e["Error"].values for e in lista_err],
            columns=["RMSE", "MAE", "ER", "R2"],
            index=indices)
