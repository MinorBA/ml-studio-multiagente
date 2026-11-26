# =============================================================================
# backend/eda_base.py
# CLASE RAÍZ: analisisEDA
# ─────────────────────────────────────────────────────────────────────────────
# Responsabilidad: carga, limpieza, inspección y visualización exploratoria.
# Todas las clases especializadas (NoSupervisado, Supervisado, Clasificacion,
# Regresion) heredan de aquí.
#
# NOTA IMPORTANTE: cada método de gráfico devuelve un objeto matplotlib.Figure
# para que Streamlit pueda renderizarlo con st.pyplot(fig).
# =============================================================================

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")                       # backend sin pantalla
import matplotlib.pyplot as plt
from math import ceil, floor, pi
from seaborn import color_palette
from datetime import datetime
try:
    from shared_utils import limpiar_df_para_agente as _limpiar
except ImportError:
    _limpiar = None


class analisisEDA:
    """
    Clase base de Análisis Exploratorio de Datos.

    Parámetros del constructor
    -------------------------
    path : str | None  — ruta al archivo CSV  (None si se carga después)
    sep  : int         — 1 → coma/punto  |  2 → punto-y-coma/punto
    """

    # ────────────────────────────────── CONSTRUCTOR ──────────────────────────
    def __init__(self, path: str = None, sep: int = 1):
        self._df: pd.DataFrame = pd.DataFrame()
        # ── Atributos del agente inteligente ─────────────────────────────── #
        self.conocimiento: dict = {}   # memoria acumulada del agente
        self.mensajes:     list = []   # historial de mensajes MAS
        self._errores:     list = []   # errores registrados por etapa
        # ─────────────────────────────────────────────────────────────────── #
        if path:
            self._df = self._cargar_csv(path, sep)

    @staticmethod
    def _cargar_csv(path: str, sep: int) -> pd.DataFrame:
        opts = {
            1: dict(sep=",",  decimal="."),
            2: dict(sep=";",  decimal="."),
        }
        if sep not in opts:
            raise ValueError("sep debe ser 1 (coma) o 2 (punto y coma).")
        return pd.read_csv(path, **opts[sep])

    def cargar_dataframe(self, df: pd.DataFrame):
        """Carga un DataFrame ya construido (ej. desde st.file_uploader)."""
        self._df = df.copy()

    # ────────────────────────────────── PROPERTY ────────────────────────────
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, valor: pd.DataFrame):
        self._df = valor

    # ────────────────────────────────── INSPECCIÓN ──────────────────────────

    # ────────────────────────────────── Requerido por AgenteEDA ───────────

    def get_nombre(self) -> str:
        """Retorna el nombre de la clase. Usado por el sistema MAS."""
        return self.__class__.__name__

    def resumen_completo(self) -> dict:
        """
        Resumen extendido con todas las estadísticas descriptivas.
        Requerido por AgenteEDA para la etapa de análisis.

        Retorna
        -------
        dict con: shape, dtypes, describe, nulos, pct_nulos, duplicados,
                  media, mediana, moda, std, varianza, max, min, skewness,
                  kurtosis, columnas_numericas, columnas_categoricas.
        """
        df  = self._df
        num = df.select_dtypes(include="number")
        cat = df.select_dtypes(exclude="number")

        pct_nulos = (df.isnull().sum() / max(len(df), 1) * 100).round(2)

        return {
            "shape":                df.shape,
            "dtypes":               df.dtypes.astype(str).to_dict(),
            "describe":             df.describe(include="all"),
            "nulos":                df.isnull().sum().to_dict(),
            "pct_nulos":            pct_nulos.to_dict(),
            "duplicados":           int(df.duplicated().sum()),
            "media":                num.mean().to_dict(),
            "mediana":              num.median().to_dict(),
            "moda":                 df.mode().iloc[0].to_dict() if len(df) > 0 else {},
            "std":                  num.std().to_dict(),
            "varianza":             num.var().to_dict(),
            "max":                  num.max().to_dict(),
            "min":                  num.min().to_dict(),
            "skewness":             num.skew().to_dict(),
            "kurtosis":             num.kurtosis().to_dict(),
            "columnas_numericas":   num.columns.tolist(),
            "columnas_categoricas": cat.columns.tolist(),
        }

    def correlacion_spearman(self) -> pd.DataFrame:
        """
        Matriz de correlación de Spearman para variables numéricas.
        Complementa fig_correlacion() que usa Pearson.
        """
        num = self._df.select_dtypes(include="number")
        return num.corr(method="spearman")

    def detectar_outliers_iqr(self, columna: str) -> dict:
        """
        Detecta outliers en una columna numérica usando el método IQR.

        Un punto es outlier si está fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

        Retorna
        -------
        dict con: q1, q3, iqr, limite_inferior, limite_superior,
                  n_outliers, pct_outliers, indices_outliers.
        """
        if columna not in self._df.columns:
            raise ValueError(f"Columna '{columna}' no existe en el DataFrame.")
        serie = self._df[columna].dropna()
        q1  = float(serie.quantile(0.25))
        q3  = float(serie.quantile(0.75))
        iqr = q3 - q1
        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr
        mascara  = (serie < lim_inf) | (serie > lim_sup)
        n_out    = int(mascara.sum())
        pct_out  = round(n_out / max(len(serie), 1) * 100, 2)
        return {
            "columna":          columna,
            "q1":               round(q1, 4),
            "q3":               round(q3, 4),
            "iqr":              round(iqr, 4),
            "limite_inferior":  round(lim_inf, 4),
            "limite_superior":  round(lim_sup, 4),
            "n_outliers":       n_out,
            "pct_outliers":     pct_out,
            "indices_outliers": serie[mascara].index.tolist(),
        }

    def detectar_outliers_zscore(self, columna: str,
                                  umbral: float = 3.0) -> dict:
        """
        Detecta outliers usando Z-Score. Un punto es outlier si |z| > umbral.

        Parámetros
        ----------
        columna : str   — nombre de la columna numérica.
        umbral  : float — valor de corte (por defecto 3.0).

        Retorna
        -------
        dict con: media, std, umbral, n_outliers, pct_outliers,
                  indices_outliers.
        """
        if columna not in self._df.columns:
            raise ValueError(f"Columna '{columna}' no existe en el DataFrame.")
        serie   = self._df[columna].dropna()
        media   = float(serie.mean())
        std     = float(serie.std())
        if std == 0:
            return {
                "columna": columna, "media": media, "std": 0,
                "umbral": umbral, "n_outliers": 0,
                "pct_outliers": 0.0, "indices_outliers": [],
            }
        z_scores = np.abs((serie - media) / std)
        mascara  = z_scores > umbral
        n_out    = int(mascara.sum())
        pct_out  = round(n_out / max(len(serie), 1) * 100, 2)
        return {
            "columna":          columna,
            "media":            round(media, 4),
            "std":              round(std, 4),
            "umbral":           umbral,
            "n_outliers":       n_out,
            "pct_outliers":     pct_out,
            "indices_outliers": serie[mascara].index.tolist(),
        }

    def fig_correlacion_spearman(self) -> plt.Figure:
        """
        Heatmap de correlación de Spearman.
        Mismo estilo que fig_correlacion() (Pearson) para consistencia visual.
        """
        corr = self.correlacion_spearman()
        if corr.empty:
            return self._fig_vacia("Sin columnas numéricas para Spearman")
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)              
        cmap = sns.diverging_palette(240, 10, as_cmap=True).reversed()
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=cmap, annot=True,
                    fmt=".2f", linewidths=0.5, linecolor="white",
                    square=True, ax=ax,
                    cbar_kws={"shrink": 0.8, "label": "Correlación"},
                    annot_kws={"size": 8, "color": "black"})
        ax.set_title("Mapa de Calor — Correlación Spearman", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    # ────────────────────────────────── LIMPIEZA ────────────────────────────

    # ────────────────────────────────── GRÁFICOS ────────────────────────────
    # Todos devuelven plt.Figure para uso con st.pyplot(fig)

    def fig_boxplot(self) -> plt.Figure:
        return self._fig_grid(
            fn=lambda col, ax: sns.boxplot(y=self._df[col], ax=ax),
            titulo=lambda c: f"Boxplot — {c}",
            ylabel=lambda c: c,
            palette="Set3",
        )

    def fig_histogramas(self) -> plt.Figure:
        cols = self._df.select_dtypes(include="number").columns
        n = len(cols)
        if n == 0:
            return self._fig_vacia("Sin columnas numéricas")
        ncols = 3
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(4 * ncols, 3 * nrows), dpi=100)  
        axes = np.array(axes).flatten()
        colores = sns.color_palette("Set2", n)
        for i, col in enumerate(cols):
            axes[i].hist(self._df[col], bins=30,
                         color=colores[i], edgecolor="black", alpha=0.7)
            axes[i].set_title(f"Histograma — {col}", fontsize=9)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].grid(True, linestyle="--", alpha=0.4)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        return fig

    def fig_correlacion(self) -> plt.Figure:
        corr = self._df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)               
        cmap = sns.diverging_palette(240, 10, as_cmap=True).reversed()
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=cmap, annot=True,
                    fmt=".2f", linewidths=0.5, linecolor="white",
                    square=True, ax=ax,
                    cbar_kws={"shrink": 0.8, "label": "Correlación"},
                    annot_kws={"size": 8, "color": "black"})
        ax.set_title("Mapa de Calor — Correlaciones", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def fig_histograma_clase(self, columna: str) -> plt.Figure:
        if columna not in self._df.columns:
            return self._fig_vacia(f"Columna '{columna}' no encontrada")
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)              
        self._df[columna].value_counts().plot(
            kind="bar", color=sns.color_palette("pastel"), ax=ax)
        ax.set_title(f"Distribución de la clase: {columna}")
        ax.set_xlabel(columna)
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig

    def fig_pairplot(self) -> plt.Figure:
        cols = self._df.select_dtypes(include="number").columns
        if len(cols) < 2:
            return self._fig_vacia("Se necesitan ≥ 2 variables numéricas")
        g = sns.pairplot(self._df[cols])
        g.fig.suptitle("Dispersión por Pares", y=1.02)
        return g.fig

    # ────────────────────────────────── HELPERS ESTÁTICOS ───────────────────
    @staticmethod
    def centroide(num_cluster: int, datos: pd.DataFrame,
                  clusters: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(datos[clusters == num_cluster].mean()).T

    @staticmethod
    def fig_bar_clusters(centros: np.ndarray, labels,
                         scale: bool = False) -> plt.Figure:
        centros = np.copy(centros)
        if scale:
            for c in range(centros.shape[1]):
                mx = max(centros[:, c])
                if mx != 0:
                    centros[:, c] /= mx
        k = centros.shape[0]
        colores = color_palette("Set2")
        minimo  = floor(centros.min()) if floor(centros.min()) < 0 else 0
        fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), dpi=100)   
        if k == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.barh(range(len(centros[i])), centros[i],
                    color=[colores[j % len(colores)]
                           for j in range(len(centros[i]))],
                    height=0.6)
            ax.set_xlim(minimo, ceil(centros.max()))
            ax.set_title(f"Cluster {i}")
            if i == 0:
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=8)
            else:
                ax.set_yticks([])
        plt.tight_layout()
        return fig

    @staticmethod
    def fig_radar(centros: np.ndarray, labels) -> plt.Figure:
        centros_n = np.array([
            ((n - min(n)) / (max(n) - min(n)) * 100)
            if max(n) != min(n) else (n / n * 50)
            for n in centros.T
        ])
        angulos = [n / float(len(labels)) * 2 * pi
                   for n in range(len(labels))] + [0]
        fig = plt.figure(figsize=(5, 5), dpi=100)                      
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(
            [a * 180 / pi for a in angulos[:-1]], labels, size=8)
        ax.set_rlabel_position(0)
        ax.set_yticks(range(10, 110, 10))
        ax.set_yticklabels(
            [f"{p}%" for p in range(10, 110, 10)], color="grey", size=7)
        ax.set_ylim(-10, 100)
        paleta = color_palette("tab10")
        for i in range(centros_n.shape[1]):
            vals = centros_n[:, i].tolist() + centros_n[:, i][:1].tolist()
            ax.plot(angulos, vals, linewidth=1.5,
                    linestyle="solid", label=f"Cluster {i}",
                    color=paleta[i % len(paleta)])
            ax.fill(angulos, vals, alpha=0.2,
                    color=paleta[i % len(paleta)])
        ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        return fig

    # ────────────────────────────────── HELPERS PRIVADOS ────────────────────
    def _fig_grid(self, fn, titulo, ylabel,
                  ncols: int = 3, palette: str = "Set3") -> plt.Figure:
        cols = self._df.select_dtypes(include="number").columns
        n = len(cols)
        if n == 0:
            return self._fig_vacia("Sin columnas numéricas")
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(4 * ncols, 3 * nrows), dpi=100)  
        axes = np.array(axes).flatten()
        for i, col in enumerate(cols):
            fn(col, axes[i])
            axes[i].set_title(titulo(col), fontsize=9)
            axes[i].set_ylabel(ylabel(col))
            axes[i].grid(True, linestyle="--", alpha=0.4)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        return fig

    @staticmethod
    def _fig_vacia(mensaje: str) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, mensaje, ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # ═══════════════════════════════════════════════════════════════════════
    # AGENTE EDA — 7 PASOS DEL AGENTE INTELIGENTE
    # ═══════════════════════════════════════════════════════════════════════

    def percibir(self, df: pd.DataFrame) -> dict:
        """
        PASO 1 — PERCEPCIÓN.

        Carga el DataFrame y registra en self.conocimiento la estructura
        básica del dataset: dimensiones, tipos, nulos, duplicados y
        columnas que parecen identificadores.

        Parámetros
        ----------
        df : pd.DataFrame — dataset a analizar.

        Retorna
        -------
        dict con el conocimiento acumulado hasta este paso.
        """
        self.cargar_dataframe(df)

        num = self._df.select_dtypes(include="number")
        cat = self._df.select_dtypes(exclude="number")

        # Columnas posibles ID: nunique > 95% de filas Y tipo entero/object (no float continuo)
        n_filas = max(len(self._df), 1)
        posibles_id = [
            col for col in self._df.columns
            if self._df[col].nunique() / n_filas > 0.95
            and not pd.api.types.is_float_dtype(self._df[col])
        ]

        # Densidad de datos: % promedio de celdas no nulas
        densidad = round(
            (1 - self._df.isnull().mean().mean()) * 100, 2
        )

        self.conocimiento.update({
            "filas":               n_filas,
            "columnas":            self._df.shape[1],
            "tipos":               self._df.dtypes.astype(str).to_dict(),
            "columnas_numericas":  num.columns.tolist(),
            "columnas_categoricas": cat.columns.tolist(),
            "nulos":               self._df.isnull().sum().to_dict(),
            "duplicados":          int(self._df.duplicated().sum()),
            "columnas_posibles_id": posibles_id,
            "densidad_datos":      densidad,
        })
        return self.conocimiento

    def analizar(self) -> dict:
        """
        PASO 2 — ANÁLISIS.

        Calcula estadísticas descriptivas completas, correlaciones Pearson
        y Spearman, y detecta outliers por IQR y Z-Score en cada columna
        numérica.

        Retorna
        -------
        dict con el conocimiento acumulado (incluye resultados del análisis).
        """
        # Estadísticas descriptivas completas
        self.conocimiento["estadisticas"] = self.resumen_completo()

        # Correlación Pearson
        num_cols = self.conocimiento.get("columnas_numericas", [])
        if len(num_cols) >= 2:
            self.conocimiento["correlacion_pearson"] = (
                self._df[num_cols].corr().to_dict()
            )
            self.conocimiento["correlacion_spearman"] = (
                self.correlacion_spearman().to_dict()
            )
        else:
            self.conocimiento["correlacion_pearson"]  = {}
            self.conocimiento["correlacion_spearman"] = {}

        # Outliers por columna numérica
        outliers_iqr    = {}
        outliers_zscore = {}
        for col in num_cols:
            try:
                outliers_iqr[col]    = self.detectar_outliers_iqr(col)
                outliers_zscore[col] = self.detectar_outliers_zscore(col)
            except Exception as e:
                outliers_iqr[col]    = {"error": str(e)}
                outliers_zscore[col] = {"error": str(e)}

        self.conocimiento["outliers_iqr"]    = outliers_iqr
        self.conocimiento["outliers_zscore"] = outliers_zscore
        return self.conocimiento

    def razonar(self) -> list:
        """
        PASO 3 — RAZONAMIENTO.

        Aplica reglas sobre el conocimiento acumulado y genera dos listas:
        - alertas       : problemas críticos que necesitan atención inmediata.
        - recomendaciones: sugerencias opcionales de preprocesamiento.

        Retorna
        -------
        list combinada de alertas y recomendaciones (alertas primero).
        """
        alertas         = []
        recomendaciones = []

        pct_nulos  = self.conocimiento.get("estadisticas", {}).get(
                         "pct_nulos", {})
        skewness   = self.conocimiento.get("estadisticas", {}).get(
                         "skewness", {})
        outliers   = self.conocimiento.get("outliers_iqr", {})
        num_cols   = self.conocimiento.get("columnas_numericas",   [])
        cat_cols   = self.conocimiento.get("columnas_categoricas", [])
        posibles_id = self.conocimiento.get("columnas_posibles_id", [])

        # ── Reglas sobre nulos ────────────────────────────────────────────
        for col, pct in pct_nulos.items():
            if pct > 40:
                alertas.append(
                    f"⚠ ALERTA: '{col}' tiene {pct}% de nulos. "
                    f"Se recomienda eliminar la columna."
                )
            elif pct > 0:
                tipo = "mediana" if col in num_cols else "moda"
                recomendaciones.append(
                    f"'{col}' tiene {pct}% de nulos. "
                    f"Se recomienda imputar con {tipo}."
                )

        # ── Reglas sobre outliers IQR ─────────────────────────────────────
        for col, info in outliers.items():
            if isinstance(info, dict) and info.get("pct_outliers", 0) > 5:
                recomendaciones.append(
                    f"'{col}' tiene {info['pct_outliers']}% de outliers (IQR). "
                    f"Se recomienda RobustScaler en lugar de StandardScaler."
                )

        # ── Reglas sobre skewness ─────────────────────────────────────────
        for col, skew in skewness.items():
            if isinstance(skew, (int, float)) and abs(skew) > 1:
                recomendaciones.append(
                    f"'{col}' tiene asimetría alta (skewness={round(skew,2)}). "
                    f"Se recomienda transformación logarítmica o BoxCox."
                )

        # ── Reglas sobre duplicados ───────────────────────────────────────
        dups = self.conocimiento.get("duplicados", 0)
        if dups > 0:
            alertas.append(
                f"⚠ ALERTA: Se detectaron {dups} registros duplicados. "
                f"Se recomienda eliminarlos."
            )

        # ── Reglas sobre columnas posibles ID ─────────────────────────────
        for col in posibles_id:
            alertas.append(
                f"⚠ ALERTA: '{col}' parece un identificador "
                f"(>95% valores únicos). Considerar eliminarla del modelo."
            )

        # ── Reglas sobre variables categóricas ───────────────────────────
        if cat_cols:
            recomendaciones.append(
                f"Existen {len(cat_cols)} variable(s) categórica(s): "
                f"{cat_cols}. Aplicar OneHotEncoder (pocas categorías) "
                f"o LabelEncoder (muchas categorías)."
            )

        # ── Reglas sobre variables numéricas ─────────────────────────────
        if num_cols:
            recomendaciones.append(
                f"Existen {len(num_cols)} variable(s) numérica(s). "
                f"Evaluar StandardScaler, MinMaxScaler o RobustScaler "
                f"según el algoritmo a usar."
            )

        # ── Regla sobre tamaño del dataset ────────────────────────────────
        if self.conocimiento.get("filas", 0) < 100:
            alertas.append(
                f"⚠ ALERTA: Dataset muy pequeño "
                f"({self.conocimiento['filas']} filas). "
                f"Los resultados pueden no ser confiables."
            )

        self.conocimiento["alertas"]         = alertas
        self.conocimiento["recomendaciones"] = recomendaciones
        return alertas + recomendaciones

    def decidir_preprocesamiento(self) -> dict:
        """
        PASO 4 — DECISIÓN.

        Construye un plan de preprocesamiento estructurado como dict,
        directamente ejecutable por los agentes AgenteClustering y
        AgenteSupervisado mediante ColumnTransformer / Pipeline.

        Retorna
        -------
        dict con claves: eliminar_columnas, eliminar_duplicados,
        imputar_mediana, imputar_moda, escalar_standard, escalar_robust,
        codificar_onehot, codificar_label, transformar_log.
        """
        pct_nulos  = self.conocimiento.get("estadisticas", {}).get(
                         "pct_nulos", {})
        skewness   = self.conocimiento.get("estadisticas", {}).get(
                         "skewness", {})
        outliers   = self.conocimiento.get("outliers_iqr", {})
        num_cols   = self.conocimiento.get("columnas_numericas",   [])
        cat_cols   = self.conocimiento.get("columnas_categoricas", [])
        posibles_id = self.conocimiento.get("columnas_posibles_id", [])

        eliminar   = list(posibles_id)   # IDs siempre se eliminan
        imp_med    = []   # imputar mediana (numéricas)
        imp_moda   = []   # imputar moda (categóricas)
        esc_std    = []   # StandardScaler
        esc_rob    = []   # RobustScaler
        cod_ohe    = []   # OneHotEncoder (≤ 10 categorías únicas)
        cod_lbl    = []   # LabelEncoder  (> 10 categorías únicas)
        transf_log = []   # transformación logarítmica

        # Nulos → eliminar o imputar
        for col, pct in pct_nulos.items():
            if pct > 40 and col not in eliminar:
                eliminar.append(col)
            elif pct > 0:
                if col in num_cols:
                    imp_med.append(col)
                elif col in cat_cols:
                    imp_moda.append(col)

        # Escalado: RobustScaler si outliers > 5 %, StandardScaler si no
        for col in num_cols:
            if col in eliminar:
                continue
            info = outliers.get(col, {})
            if isinstance(info, dict) and info.get("pct_outliers", 0) > 5:
                esc_rob.append(col)
            else:
                esc_std.append(col)

        # Skewness → log transform
        for col, skew in skewness.items():
            if col in eliminar:
                continue
            if isinstance(skew, (int, float)) and abs(skew) > 1:
                transf_log.append(col)

        # Codificación categóricas
        for col in cat_cols:
            if col in eliminar:
                continue
            n_cats = self._df[col].nunique()
            if n_cats <= 10:
                cod_ohe.append(col)
            else:
                cod_lbl.append(col)

        plan = {
            "eliminar_columnas":   eliminar,
            "eliminar_duplicados": self.conocimiento.get("duplicados", 0) > 0,
            "imputar_mediana":     imp_med,
            "imputar_moda":        imp_moda,
            "escalar_standard":    esc_std,
            "escalar_robust":      esc_rob,
            "codificar_onehot":    cod_ohe,
            "codificar_label":     cod_lbl,
            "transformar_log":     transf_log,
        }
        self.conocimiento["plan_preprocesamiento"] = plan
        return plan

    def _inferir_variable_objetivo_automatica(self) -> tuple:
        """
        MÉTODO AUXILIAR — Infiere automáticamente la variable objetivo.

        Estrategia de selección (en orden de prioridad):
        1. Columna con nombre canónico conocido (target, label, defects…).
        2. Columna categórica (object/bool) con 2–20 clases únicas.
        3. Columna numérica entera con ratio nunique/n < 0.05.
        4. Última columna numérica del DataFrame (convención habitual).

        Retorna
        -------
        (nombre_columna: str | None, justificacion: str)
        """
        df = self._df
        n  = max(len(df), 1)

        # Nombres con alta certeza (específicos de ML)
        nombres_alta = {
            "target", "label", "clase", "class", "defects", "potability",
            "survived", "diagnosis", "outcome", "output", "result",
            "churn", "default", "fraud", "etiqueta", "respuesta",
        }
        # Nombres con certeza media (también pueden ser predictores)
        nombres_media = {
            "y", "price", "salary", "revenue", "ventas", "precio", "ingreso",
        }

        # Prioridad 1a: nombre de alta certeza
        for col in df.columns:
            if col.lower().strip() in nombres_alta:
                y = df[col]
                if pd.api.types.is_numeric_dtype(y):
                    ratio = y.nunique() / n
                    tipo  = "clasificacion" if ratio < 0.05 else "regresion"
                else:
                    tipo  = "clasificacion"
                just = (f"La columna \'{col}\' tiene un nombre típico de "
                        f"variable objetivo. Tipo inferido: {tipo}.")
                return col, just

        # Prioridad 1b: nombre de certeza media
        for col in df.columns:
            if col.lower().strip() in nombres_media:
                y = df[col]
                if pd.api.types.is_numeric_dtype(y):
                    ratio = y.nunique() / n
                    tipo  = "clasificacion" if ratio < 0.05 else "regresion"
                else:
                    tipo  = "clasificacion"
                just = (f"La columna \'{col}\' tiene un nombre asociado a "
                        f"variables objetivo. Tipo inferido: {tipo}.")
                return col, just

        # Prioridad 2: categórica con pocas clases
        for col in df.columns:
            y = df[col]
            if (not pd.api.types.is_numeric_dtype(y) and 2 <= y.nunique() <= 20):
                just = (f"La columna \'{col}\' es categórica con "
                        f"{y.nunique()} valores únicos, candidata ideal "
                        f"para clasificación.")
                return col, just

        # Prioridad 3: numérica entera con pocas clases
        for col in df.columns:
            y = df[col]
            if pd.api.types.is_integer_dtype(y):
                ratio = y.nunique() / n
                if ratio < 0.05:
                    just = (f"La columna \'{col}\' es numérica entera con "
                            f"solo {y.nunique()} valores únicos "
                            f"({round(ratio*100,1)}% del total), lo que indica "
                            f"una variable categórica codificada numéricamente.")
                    return col, just

        # Prioridad 4: última columna numérica
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            col  = num_cols[-1]
            just = (f"No se encontró una variable objetivo clara. Se seleccionó "
                    f"\'{col}\' por ser la última columna numérica "
                    f"(convención habitual en datasets ML).")
            return col, just

        return None, "No se pudo inferir una variable objetivo automáticamente."

    def inferir_tipo_problema(self, variable_objetivo: str = None) -> str:
        """
        PASO 5 — INFERENCIA DEL TIPO DE PROBLEMA.

        Si variable_objetivo es None, infiere automáticamente la columna
        objetivo usando _inferir_variable_objetivo_automatica().

        Retorna
        -------
        str — "no_supervisado" | "clasificacion" | "regresion" | "error"
        """
        # Inferencia automática
        if variable_objetivo is None:
            col_inf, justificacion = self._inferir_variable_objetivo_automatica()
            if col_inf is None:
                self.conocimiento["tipo_problema"]          = "no_supervisado"
                self.conocimiento["subtipo"]                = "clustering"
                self.conocimiento["variable_objetivo"]      = None
                self.conocimiento["justificacion_objetivo"] = justificacion
                return "no_supervisado"
            variable_objetivo = col_inf
            self.conocimiento["justificacion_objetivo"] = justificacion
        else:
            self.conocimiento["justificacion_objetivo"] = (
                f"Variable objetivo \'{variable_objetivo}\' "
                f"seleccionada manualmente por el usuario."
            )

        if variable_objetivo not in self._df.columns:
            self.conocimiento["tipo_problema"] = "error"
            self.conocimiento["subtipo"]       = "columna_no_encontrada"
            return "error"

        y     = self._df[variable_objetivo]
        dtype = str(y.dtype)

        # Clasificación: object, bool, category, o string de pandas
        is_cat = (dtype in ("object", "bool", "string", "boolean", "str") or
                  isinstance(y.dtype, pd.CategoricalDtype) or
                  pd.api.types.is_object_dtype(y) or
                  hasattr(y.dtype, "name") and "str" in str(y.dtype))

        if is_cat:
            tipo    = "clasificacion"
            n_cls   = y.nunique()
            subtipo = "binaria" if n_cls == 2 else "multiclase"

        elif pd.api.types.is_numeric_dtype(y):
            ratio = y.nunique() / max(len(y), 1)
            if ratio < 0.05:
                tipo    = "clasificacion"
                n_cls   = y.nunique()
                subtipo = "binaria" if n_cls == 2 else "multiclase"
            else:
                tipo    = "regresion"
                skew    = float(y.skew())
                subtipo = "regresion_sesgada" if abs(skew) > 1 else "regresion_normal"
                if abs(skew) > 1:
                    self.conocimiento.setdefault("alertas", []).append(
                        f"La variable objetivo \'{variable_objetivo}\' "
                        f"tiene skewness={round(skew,2)}. "
                        f"Considerar transformación log(y+1)."
                    )
        else:
            tipo    = "error"
            subtipo = "tipo_no_reconocido"

        self.conocimiento["tipo_problema"]     = tipo
        self.conocimiento["subtipo"]           = subtipo
        self.conocimiento["variable_objetivo"] = variable_objetivo
        return tipo

    def comunicar(self) -> dict:
        """
        PASO 6 — COMUNICACIÓN CON EL AGENTE COORDINADOR.

        Construye el mensaje estandarizado MAS con todos los hallazgos
        del AgenteEDA y lo agrega al historial de mensajes.

        Retorna
        -------
        dict con estructura: agente, tipo, timestamp, contenido.
        """
        mensaje = {
            "agente":    "AgenteEDA",
            "tipo":      "REPORTE_EDA",
            "timestamp": datetime.now().isoformat(),
            "contenido": {
                "shape":                  (self.conocimiento.get("filas", 0),
                                           self.conocimiento.get("columnas", 0)),
                "columnas_numericas":     self.conocimiento.get(
                                              "columnas_numericas", []),
                "columnas_categoricas":   self.conocimiento.get(
                                              "columnas_categoricas", []),
                "columnas_posibles_id":   self.conocimiento.get(
                                              "columnas_posibles_id", []),
                "duplicados":             self.conocimiento.get("duplicados", 0),
                "densidad_datos":         self.conocimiento.get(
                                              "densidad_datos", 100.0),
                "alertas":                self.conocimiento.get("alertas", []),
                "recomendaciones":        self.conocimiento.get(
                                              "recomendaciones", []),
                "plan_preprocesamiento":  self.conocimiento.get(
                                              "plan_preprocesamiento", {}),
                "tipo_problema":          self.conocimiento.get(
                                              "tipo_problema", "no_inferido"),
                "subtipo":                self.conocimiento.get(
                                              "subtipo", ""),
                "variable_objetivo":      self.conocimiento.get(
                                              "variable_objetivo", None),
                "pre_eliminadas":         self.conocimiento.get(
                                              "pre_eliminadas", []),
                "pre_imputadas":          self.conocimiento.get(
                                              "pre_imputadas", []),
                "errores_etapas":         self._errores,
            },
        }
        self.mensajes.append(mensaje)
        return mensaje

    def ejecutar(self, df: pd.DataFrame,
                 variable_objetivo: str = None) -> dict:
        """
        PASO 7 — EJECUCIÓN COMPLETA DEL AGENTE.

        Orquesta los 6 pasos anteriores en orden. Cada paso se ejecuta
        dentro de un try/except individual: si falla, el error se registra
        en self._errores y la ejecución continúa con el siguiente paso.

        Parámetros
        ----------
        df               : pd.DataFrame — dataset a analizar.
        variable_objetivo: str | None   — columna objetivo (opcional).

        Retorna
        -------
        dict con claves: mensaje, figuras, conocimiento.
        """
        # ── Pre-limpieza: eliminar columnas 100% vacías e IDs ─────────────
        try:
            if _limpiar is not None:
                df, _elim, _imp = _limpiar(df, variable_objetivo=variable_objetivo)
                if _elim:
                    self._errores  # no es error, es info
                    # guardar para mostrar en EDA
                    self.conocimiento["pre_eliminadas"] = _elim
                    self.conocimiento["pre_imputadas"]  = _imp
        except Exception as e:
            self._errores.append({"paso": "pre_limpieza", "error": str(e)})

        # ── Paso 1: Percepción ────────────────────────────────────────────
        try:
            self.percibir(df)
        except Exception as e:
            self._errores.append({"paso": "percibir", "error": str(e)})

        # ── Paso 2: Análisis ──────────────────────────────────────────────
        try:
            self.analizar()
        except Exception as e:
            self._errores.append({"paso": "analizar", "error": str(e)})

        # ── Paso 3: Razonamiento ──────────────────────────────────────────
        try:
            self.razonar()
        except Exception as e:
            self._errores.append({"paso": "razonar", "error": str(e)})

        # ── Paso 4: Decisión ──────────────────────────────────────────────
        try:
            self.decidir_preprocesamiento()
        except Exception as e:
            self._errores.append({"paso": "decidir_preprocesamiento",
                                   "error": str(e)})

        # ── Paso 5: Inferencia ────────────────────────────────────────────
        try:
            self.inferir_tipo_problema(variable_objetivo)
        except Exception as e:
            self._errores.append({"paso": "inferir_tipo_problema",
                                   "error": str(e)})

        # ── Figuras (independientes del flujo principal) ───────────────────
        # reset_index para evitar errores con índices duplicados en seaborn
        self._df = self._df.reset_index(drop=True)
        figuras = {}
        for nombre, metodo in [
            ("histogramas",          self.fig_histogramas),
            ("boxplots",             self.fig_boxplot),
            ("correlacion_pearson",  self.fig_correlacion),
            ("correlacion_spearman", self.fig_correlacion_spearman),
        ]:
            try:
                figuras[nombre] = metodo()
            except Exception as e:
                self._errores.append({"paso": f"fig_{nombre}",
                                       "error": str(e)})

        # ── Paso 6: Comunicación ──────────────────────────────────────────
        try:
            mensaje = self.comunicar()
        except Exception as e:
            self._errores.append({"paso": "comunicar", "error": str(e)})
            mensaje = {}

        return {
            "mensaje":      mensaje,
            "figuras":      figuras,
            "conocimiento": self.conocimiento,
        }
