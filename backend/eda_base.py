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
    def tipoDatos(self) -> pd.Series:
        return self._df.dtypes

    def resumen(self) -> dict:
        """Dict con métricas descriptivas (útil para Streamlit)."""
        num = self._df.select_dtypes(include="number")
        return {
            "shape":      self._df.shape,
            "dtypes":     self._df.dtypes,
            "describe":   self._df.describe(include="all"),
            "nulos":      self._df.isnull().sum(),
            "duplicados": int(self._df.duplicated().sum()),
            "media":      num.mean(),
            "mediana":    num.median(),
            "std":        num.std(),
            "max":        num.max(),
            "min":        num.min(),
            "quantiles":  num.quantile([0, .25, .5, .75, 1]),
        }

    def valores_unicos(self, columna: str) -> pd.Series:
        return self._df[columna].value_counts()

    def valores_faltantes(self) -> pd.Series:
        return self._df.isnull().sum()

    # ────────────────────────────────── LIMPIEZA ────────────────────────────
    def analisisNumerico(self):
        """Conserva solo columnas numéricas."""
        self._df = self._df.select_dtypes(include=["number"])

    def analisisCompleto(self):
        """Convierte categóricas a dummies (one-hot)."""
        cats = self._df.select_dtypes(
            include=["object", "category"]).columns.tolist()
        self._df = pd.get_dummies(
            self._df, columns=cats, drop_first=True).astype(int)

    def eliminarColumnas(self, columnas: list):
        self._df.drop(columns=columnas, inplace=True, errors="ignore")

    def renombrarColumnas(self, mapa: dict):
        self._df.rename(columns=mapa, inplace=True)

    def eliminarDuplicados(self) -> int:
        antes = len(self._df)
        self._df.drop_duplicates(inplace=True)
        return antes - len(self._df)

    def eliminarNulos(self) -> int:
        antes = len(self._df)
        self._df.dropna(inplace=True)
        return antes - len(self._df)

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
                                  figsize=(5 * ncols, 4 * nrows), dpi=130)
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

    def fig_densidad(self) -> plt.Figure:
        return self._fig_grid(
            fn=lambda col, ax: sns.kdeplot(
                data=self._df, x=col, fill=True, ax=ax, linewidth=2),
            titulo=lambda c: f"Densidad — {c}",
            ylabel=lambda _: "Densidad",
            palette="husl",
        )

    def fig_distribucion(self) -> plt.Figure:
        return self._fig_grid(
            fn=lambda col, ax: sns.histplot(
                self._df[col], kde=True, ax=ax, bins=30),
            titulo=lambda c: f"Distribución — {c}",
            ylabel=lambda _: "Frecuencia",
            palette="coolwarm",
        )

    def fig_correlacion(self) -> plt.Figure:
        corr = self._df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 8), dpi=130)
        cmap = sns.diverging_palette(240, 10, as_cmap=True).reversed()
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=cmap, annot=True,
                    fmt=".2f", linewidths=0.5, linecolor="white",
                    square=True, ax=ax,
                    cbar_kws={"shrink": 0.8, "label": "Correlación"},
                    annot_kws={"size": 9, "color": "black"})
        ax.set_title("Mapa de Calor — Correlaciones", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig

    def fig_histograma_clase(self, columna: str) -> plt.Figure:
        if columna not in self._df.columns:
            return self._fig_vacia(f"Columna '{columna}' no encontrada")
        fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
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
    def recodificar(col: pd.Series, nuevo_codigo: dict) -> pd.Series:
        return col.replace(nuevo_codigo)

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
        fig, axes = plt.subplots(1, k, figsize=(5 * k, 6), dpi=130)
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
        fig = plt.figure(figsize=(8, 8), dpi=130)
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
                                  figsize=(5 * ncols, 4 * nrows), dpi=130)
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
