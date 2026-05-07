# =============================================================================
# backend/no_supervisado.py
# CLASE: NoSupervisado  (hereda de analisisEDA)
# ─────────────────────────────────────────────────────────────────────────────
# Técnicas incluidas:
#   • ACP   — Análisis de Componentes Principales (prince.PCA)
#   • HAC   — Clustering Jerárquico Aglomerativo (Ward/Average/Single/Complete)
#   • KMedia — K-Medias y K-Medoids
#   • UMAP  — Reducción de dimensiones
#   • TSNE  — t-Distributed Stochastic Neighbor Embedding
#
# Todos los gráficos devuelven plt.Figure → st.pyplot(fig) en Streamlit
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster         import KMeans, AgglomerativeClustering
from sklearn.decomposition   import PCA
from sklearn.manifold        import TSNE as TSNE_sklearn
from sklearn.metrics         import silhouette_score
from sklearn.preprocessing   import StandardScaler
from sklearn_extra.cluster   import KMedoids
from scipy.cluster.hierarchy import (dendrogram, ward, single,
                                      complete, average, linkage, fcluster)
import umap as um
from prince import PCA as PCA_Prince

from backend.eda_base import analisisEDA


# ═════════════════════════════════════════════════════════════════════════════
# Clase auxiliar interna: ACP
# ═════════════════════════════════════════════════════════════════════════════
class _ACPInterno:
    """Análisis de Componentes Principales (envuelve prince.PCA)."""

    def __init__(self, datos: pd.DataFrame, n_comp: int = 2):
        self.datos = datos
        modelo = PCA_Prince(n_components=n_comp).fit(datos)
        self.modelo          = modelo
        self.correlacion_var = modelo.column_correlations
        self.coordenadas_ind = modelo.row_coordinates(datos)
        self.contribucion_ind = modelo.row_contributions_
        self.cos2_ind        = modelo.row_cosine_similarities(datos)
        self.var_explicada   = modelo.percentage_of_variance_

    # ---- etiqueta helper
    def _etiqueta(self, eje: int) -> str:
        return f"Comp {eje} ({round(self.var_explicada[eje], 2)}%)"

    # ---- gráficos (devuelven fig)
    def fig_plano_principal(self, ejes=[0, 1]) -> plt.Figure:
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
        ax.scatter(x, y, color="gray", alpha=0.7)
        ax.axhline(0, color="dimgrey", ls="--", lw=0.8)
        ax.axvline(0, color="dimgrey", ls="--", lw=0.8)
        ax.set_xlabel(self._etiqueta(ejes[0]))
        ax.set_ylabel(self._etiqueta(ejes[1]))
        ax.set_title("Plano Principal")
        for i, txt in enumerate(self.coordenadas_ind.index):
            ax.annotate(txt, (x[i], y[i]), fontsize=7)
        plt.tight_layout()
        return fig

    def fig_circulo(self, ejes=[0, 1]) -> plt.Figure:
        cor = self.correlacion_var.iloc[:, ejes].values
        fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
        ax.add_patch(plt.Circle((0, 0), 1, color="steelblue", fill=False))
        ax.set_aspect("equal")
        ax.axhline(0, color="dimgrey", ls="--", lw=0.8)
        ax.axvline(0, color="dimgrey", ls="--", lw=0.8)
        ax.set_xlabel(self._etiqueta(ejes[0]))
        ax.set_ylabel(self._etiqueta(ejes[1]))
        ax.set_title("Círculo de Correlación")
        for i in range(cor.shape[0]):
            ax.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95,
                     color="steelblue", alpha=0.6,
                     head_width=0.04, head_length=0.04)
            ax.text(cor[i, 0] * 1.08, cor[i, 1] * 1.08,
                    self.correlacion_var.index[i],
                    color="steelblue", ha="center", va="center", fontsize=8)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        plt.tight_layout()
        return fig

    def fig_sobreposicion(self, ejes=[0, 1]) -> plt.Figure:
        x   = self.coordenadas_ind[ejes[0]].values
        y   = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes].values
        scale = min(
            (max(x) - min(x)) / (max(cor[:, 0]) - min(cor[:, 0]) + 1e-9),
            (max(y) - min(y)) / (max(cor[:, 1]) - min(cor[:, 1]) + 1e-9),
        ) * 0.7
        fig, ax = plt.subplots(figsize=(9, 6), dpi=130)
        ax.scatter(x, y, color="gray", alpha=0.6)
        ax.axhline(0, color="dimgrey", ls="--", lw=0.8)
        ax.axvline(0, color="dimgrey", ls="--", lw=0.8)
        ax.set_xlabel(self._etiqueta(ejes[0]))
        ax.set_ylabel(self._etiqueta(ejes[1]))
        ax.set_title("Sobreposición Plano – Círculo")
        for i, txt in enumerate(self.coordenadas_ind.index):
            ax.annotate(txt, (x[i], y[i]), fontsize=6, alpha=0.7)
        for i in range(cor.shape[0]):
            ax.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale,
                     color="steelblue", alpha=0.6,
                     head_width=0.05, head_length=0.05)
            ax.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15,
                    self.correlacion_var.index[i],
                    color="steelblue", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        return fig


# ═════════════════════════════════════════════════════════════════════════════
# Clase principal: NoSupervisado
# ═════════════════════════════════════════════════════════════════════════════
class NoSupervisado(analisisEDA):
    """
    Análisis no supervisado: ACP + Clustering.

    Constructor
    -----------
    df : pd.DataFrame — datos numéricos, limpios y estandarizados.
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df   = df
        self._log  = []      # historial para benchmark

    # ─────────────────────── ACP ────────────────────────────────────────────
    def ACP(self, n_componentes: int = 2) -> dict:
        """
        Ejecuta ACP y devuelve dict con las 3 figuras + datos del modelo.

        Retorna
        -------
        dict con keys: 'plano', 'circulo', 'sobreposicion',
                       'var_explicada', 'correlacion_var',
                       'coordenadas_ind', 'cos2_ind'
        """
        acp = _ACPInterno(self._df, n_componentes)
        return {
            "plano":           acp.fig_plano_principal(),
            "circulo":         acp.fig_circulo(),
            "sobreposicion":   acp.fig_sobreposicion(),
            "var_explicada":   acp.var_explicada,
            "correlacion_var": acp.correlacion_var,
            "coordenadas_ind": acp.coordenadas_ind,
            "cos2_ind":        acp.cos2_ind,
        }

    # ─────────────────────── HAC ────────────────────────────────────────────
    def HAC(self, n_clusters: int = 3) -> dict:
        """
        Clustering jerárquico con 4 métodos de enlace.

        Retorna
        -------
        dict con: 'dendrogramas' (list[fig]), 'centroides' (fig),
                  'silhouette' (dict[str, dict])
        """
        metodos = {
            "Ward":     ward(self._df),
            "Average":  average(self._df),
            "Single":   single(self._df),
            "Complete": complete(self._df),
        }
        dendros = {}
        for nombre, res in metodos.items():
            dendros[nombre] = self._fig_dendrograma(res, nombre)

        fig_centros = self._fig_centroides_hac(n_clusters)
        sil = {}
        for nombre in metodos:
            k, score, fig = self._silhouette_hac(nombre.lower(), nombre)
            sil[nombre] = {"mejor_k": k, "score": score, "fig": fig}
            self._registrar(f"HAC-{nombre}", k, score)

        return {"dendrogramas": dendros, "centroides": fig_centros,
                "silhouette": sil}

    def _fig_dendrograma(self, res, titulo: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=130)
        dendrogram(res, labels=self._df.index.tolist(), ax=ax)
        ax.set_title(f"Dendrograma — {titulo}")
        ax.grid(False)
        plt.tight_layout()
        return fig

    def _fig_centroides_hac(self, n_clusters: int) -> plt.Figure:
        grupos = fcluster(
            linkage(self._df, method="ward", metric="euclidean"),
            n_clusters, criterion="maxclust") - 1
        centros = np.array(pd.concat([
            analisisEDA.centroide(i, self._df, grupos)
            for i in range(n_clusters)
        ]))
        return analisisEDA.fig_bar_clusters(centros, self._df.columns,
                                             scale=True)

    def _silhouette_hac(self, metodo: str, etiqueta: str):
        scores = []
        for k in range(2, 11):
            labels = AgglomerativeClustering(
                n_clusters=k, linkage=metodo,
                metric="euclidean").fit_predict(self._df)
            scores.append(silhouette_score(self._df, labels))
        mejor_k = int(np.argmax(scores)) + 2
        fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
        ax.plot(range(2, 11), scores, marker="o", color="steelblue")
        ax.set_title(f"Silhouette Score — HAC {etiqueta}")
        ax.set_xlabel("Clusters")
        ax.set_ylabel("Score")
        ax.grid(True)
        plt.tight_layout()
        return mejor_k, max(scores), fig

    # ─────────────────────── K-MEDIAS / K-MEDOIDS ──────────────────────────
    def KMedia(self, n_clusters: int = 3) -> dict:
        """
        K-Medias y K-Medoids.

        Retorna
        -------
        dict con keys: 'kmedias_scatter', 'kmedias_radar',
                       'kmedoids_scatter', 'kmedoids_radar',
                       'score_kmedias', 'score_kmedoids'
        """
        r = {}
        # K-Medias
        km = KMeans(n_clusters=n_clusters, max_iter=500, n_init=150)
        km.fit(self._df)
        r["kmedias_scatter"] = self._fig_scatter_cluster(
            km.predict(self._df), f"K-Medias (k={n_clusters})")
        r["kmedias_radar"]   = analisisEDA.fig_radar(
            km.cluster_centers_, self._df.columns)
        sc_km = silhouette_score(self._df, km.labels_)
        r["score_kmedias"] = sc_km
        self._registrar("K-Medias", n_clusters, sc_km)

        # K-Medoids
        kmd = KMedoids(n_clusters=n_clusters, max_iter=500, metric="cityblock")
        kmd.fit(self._df)
        r["kmedoids_scatter"] = self._fig_scatter_cluster(
            kmd.predict(self._df), f"K-Medoids (k={n_clusters})")
        r["kmedoids_radar"]   = analisisEDA.fig_radar(
            kmd.cluster_centers_, self._df.columns)
        sc_kmd = silhouette_score(self._df, kmd.labels_)
        r["score_kmedoids"] = sc_kmd
        self._registrar("K-Medoids", n_clusters, sc_kmd)
        return r

    def _fig_scatter_cluster(self, etiquetas, titulo: str) -> plt.Figure:
        pca  = PCA(n_components=2)
        comp = pca.fit_transform(self._df)
        colores = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12",
                   "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
                   "#16a085", "#c0392b"]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
        ax.scatter(comp[:, 0], comp[:, 1],
                   c=[colores[l % len(colores)] for l in etiquetas],
                   alpha=0.7)
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.set_title(titulo)
        ax.grid(False)
        plt.tight_layout()
        return fig

    # ─────────────────────── UMAP ──────────────────────────────────────────
    def UMAP(self, n_componentes: int = 2,
             n_neighbors: int = 15) -> dict:
        """
        UMAP – reducción de dimensiones.

        Retorna
        -------
        dict con 'fig' (plt.Figure) y 'componentes' (np.ndarray)
        """
        if self._df.isnull().any().any():
            raise ValueError("El DataFrame contiene valores nulos.")
        scaled = StandardScaler().fit_transform(self._df)
        modelo = um.UMAP(n_components=n_componentes, n_neighbors=n_neighbors)
        comp   = modelo.fit_transform(scaled)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
        ax.scatter(comp[:, 0], comp[:, 1], alpha=0.7)
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.set_title("UMAP")
        ax.grid(False)
        plt.tight_layout()
        return {"fig": fig, "componentes": comp}

    # ─────────────────────── T-SNE ─────────────────────────────────────────
    def TSNE(self, random_state: int = 42) -> dict:
        """
        t-SNE + selección automática de k óptimo por Silhouette.

        Retorna
        -------
        dict con 'fig_silhouette', 'fig_scatter',
                 'mejor_k', 'mejor_score'
        """
        tsne    = TSNE_sklearn(n_components=2, random_state=random_state)
        df_tsne = tsne.fit_transform(self._df)
        scores  = []
        for k in range(2, 11):
            labels = KMeans(n_clusters=k, random_state=random_state,
                             n_init="auto").fit_predict(df_tsne)
            scores.append(silhouette_score(df_tsne, labels))

        mejor_k = int(np.argmax(scores)) + 2
        labels  = KMeans(n_clusters=mejor_k, random_state=random_state,
                          n_init="auto").fit_predict(df_tsne)

        # Silhouette curve
        fig1, ax1 = plt.subplots(figsize=(8, 4), dpi=130)
        ax1.plot(range(2, 11), scores, marker="o", color="steelblue")
        ax1.set_xlabel("Clusters")
        ax1.set_ylabel("Silhouette Score")
        ax1.set_title("Silhouette Score — t-SNE")
        ax1.grid(True)
        plt.tight_layout()

        # Scatter
        fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=130)
        sns.scatterplot(x=df_tsne[:, 0], y=df_tsne[:, 1],
                        hue=labels, palette="tab10", legend="full", ax=ax2)
        ax2.set_title("t-SNE")
        ax2.set_xlabel("Dimensión 1")
        ax2.set_ylabel("Dimensión 2")
        ax2.legend(title="Cluster")
        plt.tight_layout()

        self._registrar("t-SNE", mejor_k, max(scores))
        return {"fig_silhouette": fig1, "fig_scatter": fig2,
                "mejor_k": mejor_k, "mejor_score": max(scores)}

    # ─────────────────────── BENCHMARK ─────────────────────────────────────
    def _registrar(self, alg: str, k: int, score: float):
        self._log.append([alg, k, round(score, 4)])

    def benchmark(self) -> pd.DataFrame:
        """Tabla comparativa de todos los modelos ejecutados."""
        return pd.DataFrame(
            self._log,
            columns=["Algoritmo", "Clusters", "Silhouette Score"])
