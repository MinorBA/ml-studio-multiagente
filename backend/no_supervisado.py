# =============================================================================
# backend/no_supervisado.py
# CLASE: NoSupervisado  (hereda de analisisEDA)
# ─────────────────────────────────────────────────────────────────────────────
# Técnicas incluidas:
#   • ACP   — Análisis de Componentes Principales (delegado a backend.CACP.ACP)
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
from datetime import datetime
try:
    from shared_utils import limpiar_df_para_agente as _limpiar
except ImportError:
    _limpiar = None

# ── ACP: módulo independiente (ya no se duplica aquí) ────────────────────────
from backend.CACP import ACP

from sklearn.cluster         import KMeans, AgglomerativeClustering
from sklearn.decomposition   import PCA
from sklearn.manifold        import TSNE as TSNE_sklearn
from sklearn.metrics         import (silhouette_score, pairwise_distances,
                                      davies_bouldin_score,
                                      calinski_harabasz_score)
from sklearn.preprocessing   import StandardScaler
from scipy.cluster.hierarchy import (dendrogram, ward, single,
                                      complete, average, linkage, fcluster)
import umap as um

from backend.eda_base import analisisEDA


# =============================================================================
# KMedoids propio (elimina dependencia de sklearn-extra que falla en Py3.10)
# =============================================================================
class _KMedoids:
    """Implementación ligera de K-Medoids (PAM simplificado)."""

    def __init__(self, n_clusters=3, max_iter=500, metric="cityblock"):
        self.n_clusters       = n_clusters
        self.max_iter         = max_iter
        self.metric           = metric
        self.labels_          = None
        self.cluster_centers_ = None
        self.medoid_indices_  = None

    def fit(self, X):
        X = np.asarray(X)
        D = pairwise_distances(X, metric=self.metric)
        indices = np.argsort(D.sum(axis=1))[:self.n_clusters]
        for _ in range(self.max_iter):
            labels = np.argmin(D[:, indices], axis=1)
            new_indices = np.copy(indices)
            for c in range(self.n_clusters):
                mask = np.where(labels == c)[0]
                if len(mask) == 0:
                    continue
                sub_D = D[np.ix_(mask, mask)]
                new_indices[c] = mask[np.argmin(sub_D.sum(axis=1))]
            if np.array_equal(new_indices, indices):
                break
            indices = new_indices
        self.medoid_indices_  = indices
        self.cluster_centers_ = X[indices]
        self.labels_          = np.argmin(D[:, indices], axis=1)
        return self

    def predict(self, X):
        X = np.asarray(X)
        D = pairwise_distances(X, self.cluster_centers_, metric=self.metric)
        return np.argmin(D, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# =============================================================================
# Clase principal: NoSupervisado
# =============================================================================
class NoSupervisado(analisisEDA):
    """
    Análisis no supervisado: ACP + Clustering.

    El ACP es delegado completamente a backend.CACP.ACP.
    Los gráficos de ACP se extraen llamando a los métodos fig_* de esa clase,
    que devuelven plt.Figure compatibles con st.pyplot() en Streamlit.

    Constructor
    -----------
    df : pd.DataFrame — datos numéricos, limpios y estandarizados.
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df  = df
        self._log = []          # historial para benchmark()
        # ── Atributos del agente inteligente ─────────────────────────────── #
        self.conocimiento: dict = {}   # memoria acumulada del agente
        self.mensajes:     list = []   # historial de mensajes MAS
        self._errores:     list = []   # errores registrados por etapa
        self._X_scaled           = None  # datos escalados reutilizados
        # ─────────────────────────────────────────────────────────────────── #

    # =========================================================================
    # ACP — delegado a backend.CACP.ACP
    # =========================================================================
    def ACP(self, n_componentes: int = 2) -> dict:
        """
        Ejecuta el Análisis de Componentes Principales.

        Delega toda la lógica a backend.CACP.ACP.
        Adapta sus métodos de gráfico para devolver plt.Figure
        (compatible con Streamlit) en lugar de llamar plt.show().

        Retorna
        -------
        dict con keys:
            'plano'          → plt.Figure  (plano principal)
            'circulo'        → plt.Figure  (círculo de correlación)
            'sobreposicion'  → plt.Figure  (sobreposición plano-círculo)
            'var_explicada'  → list[float] (varianza explicada por componente)
            'correlacion_var'→ pd.DataFrame
            'coordenadas_ind'→ pd.DataFrame
            'cos2_ind'       → pd.DataFrame
        """
        acp = ACP(self._df, n_componentes)
        return {
            "plano":           self._fig_acp(acp, "plano"),
            "circulo":         self._fig_acp(acp, "circulo"),
            "sobreposicion":   self._fig_acp(acp, "sobreposicion"),
            "var_explicada":   acp.var_explicada,
            "correlacion_var": acp.correlacion_var,
            "coordenadas_ind": acp.coordenadas_ind,
            "cos2_ind":        acp.cos2_ind,
        }

    def _fig_acp(self, acp: ACP, tipo: str) -> plt.Figure:
        """
        Llama al método de gráfico de ACP y captura la figura activa
        sin que se dispare plt.show().

        La clase CACP.ACP dibuja sobre el eje activo de matplotlib;
        este wrapper lo intercepta devolviendo un plt.Figure limpio
        que Streamlit puede renderizar con st.pyplot(fig).
        """
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)              
        plt.sca(ax)                     # establece el eje activo

        if tipo == "plano":
            # Redibujamos manualmente usando los datos de ACP
            # para tener control total sobre la figura devuelta.
            x = acp.coordenadas_ind[0].values
            y = acp.coordenadas_ind[1].values
            ax.scatter(x, y, color="gray", alpha=0.7)
            ax.axhline(0, color="dimgrey", ls="--", lw=0.8)
            ax.axvline(0, color="dimgrey", ls="--", lw=0.8)
            ax.set_xlabel(f"Comp 0 ({round(acp.var_explicada[0], 2)}%)")
            ax.set_ylabel(f"Comp 1 ({round(acp.var_explicada[1], 2)}%)")
            ax.set_title("Plano Principal")
            for i, txt in enumerate(acp.coordenadas_ind.index):
                ax.annotate(txt, (x[i], y[i]), fontsize=7)

        elif tipo == "circulo":
            cor = acp.correlacion_var.iloc[:, [0, 1]].values
            ax.add_patch(plt.Circle((0, 0), 1, color="steelblue", fill=False))
            ax.set_aspect("equal")
            ax.axhline(0, color="dimgrey", ls="--", lw=0.8)
            ax.axvline(0, color="dimgrey", ls="--", lw=0.8)
            ax.set_xlabel(f"Comp 0 ({round(acp.var_explicada[0], 2)}%)")
            ax.set_ylabel(f"Comp 1 ({round(acp.var_explicada[1], 2)}%)")
            ax.set_title("Círculo de Correlación")
            for i in range(cor.shape[0]):
                ax.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95,
                         color="steelblue", alpha=0.6,
                         head_width=0.04, head_length=0.04)
                ax.text(cor[i, 0] * 1.08, cor[i, 1] * 1.08,
                        acp.correlacion_var.index[i],
                        color="steelblue", ha="center", va="center",
                        fontsize=8)
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)

        elif tipo == "sobreposicion":
            x   = acp.coordenadas_ind[0].values
            y   = acp.coordenadas_ind[1].values
            cor = acp.correlacion_var.iloc[:, [0, 1]].values
            scale = min(
                (max(x) - min(x)) / (max(cor[:, 0]) - min(cor[:, 0]) + 1e-9),
                (max(y) - min(y)) / (max(cor[:, 1]) - min(cor[:, 1]) + 1e-9),
            ) * 0.7
            ax.scatter(x, y, color="gray", alpha=0.6)
            ax.axhline(0, color="dimgrey", ls="--", lw=0.8)
            ax.axvline(0, color="dimgrey", ls="--", lw=0.8)
            ax.set_xlabel(f"Comp 0 ({round(acp.var_explicada[0], 2)}%)")
            ax.set_ylabel(f"Comp 1 ({round(acp.var_explicada[1], 2)}%)")
            ax.set_title("Sobreposición Plano – Círculo")
            for i, txt in enumerate(acp.coordenadas_ind.index):
                ax.annotate(txt, (x[i], y[i]), fontsize=6, alpha=0.7)
            for i in range(cor.shape[0]):
                ax.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale,
                         color="steelblue", alpha=0.6,
                         head_width=0.05, head_length=0.05)
                ax.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15,
                        acp.correlacion_var.index[i],
                        color="steelblue", ha="center", va="center",
                        fontsize=8)
        else:
            raise ValueError(f"Tipo de gráfico ACP desconocido: '{tipo}'")

        plt.tight_layout()
        return fig

    # =========================================================================
    # HAC — Clustering Jerárquico Aglomerativo
    # =========================================================================
    def HAC(self, n_clusters: int = 3) -> dict:
        """
        Clustering jerárquico con 4 métodos de enlace.

        Retorna
        -------
        dict con: 'dendrogramas' (dict[str, fig]), 'centroides' (fig),
                  'silhouette' (dict[str, dict])
        """
        metodos = {
            "Ward":     ward(self._df),
            "Average":  average(self._df),
            "Single":   single(self._df),
            "Complete": complete(self._df),
        }
        dendros = {nombre: self._fig_dendrograma(res, nombre)
                   for nombre, res in metodos.items()}

        fig_centros = self._fig_centroides_hac(n_clusters)

        sil = {}
        for nombre in metodos:
            k, score, fig, db, ch = self._silhouette_hac(nombre.lower(), nombre)  
            sil[nombre] = {"mejor_k": k, "score": score, "fig": fig}
            self._registrar(f"HAC-{nombre}", k, score, db, ch)

        return {"dendrogramas": dendros, "centroides": fig_centros,
                "silhouette": sil}

    def _fig_dendrograma(self, res, titulo: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)               
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
        # Calcular DB y CH para el mejor_k                        # MODIFICADO
        labels_opt = AgglomerativeClustering(
            n_clusters=mejor_k, linkage=metodo,
            metric="euclidean").fit_predict(self._df)
        try:
            db_score = davies_bouldin_score(self._df, labels_opt)
            ch_score = calinski_harabasz_score(self._df, labels_opt)
        except Exception:
            db_score = ch_score = float("nan")
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)               
        ax.plot(range(2, 11), scores, marker="o", color="steelblue")
        ax.set_title(f"Silhouette Score — HAC {etiqueta}")
        ax.set_xlabel("Clusters")
        ax.set_ylabel("Score")
        ax.grid(True)
        plt.tight_layout()
        return mejor_k, max(scores), fig, db_score, ch_score

    # =========================================================================
    # K-MEDIAS / K-MEDOIDS
    # =========================================================================
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
        r["score_kmedias"]   = sc_km
        try:                                                       
            db_km = davies_bouldin_score(self._df, km.labels_)
            ch_km = calinski_harabasz_score(self._df, km.labels_)
        except Exception:
            db_km = ch_km = float("nan")
        self._registrar("K-Medias", n_clusters, sc_km, db_km, ch_km)

        # K-Medoids
        kmd = _KMedoids(n_clusters=n_clusters, max_iter=500,
                        metric="cityblock")
        kmd.fit(self._df)
        r["kmedoids_scatter"] = self._fig_scatter_cluster(
            kmd.predict(self._df), f"K-Medoids (k={n_clusters})")
        r["kmedoids_radar"]   = analisisEDA.fig_radar(
            kmd.cluster_centers_, self._df.columns)
        sc_kmd = silhouette_score(self._df, kmd.labels_)
        r["score_kmedoids"]   = sc_kmd
        try:                                                       
            db_kmd = davies_bouldin_score(self._df, kmd.labels_)
            ch_kmd = calinski_harabasz_score(self._df, kmd.labels_)
        except Exception:
            db_kmd = ch_kmd = float("nan")
        self._registrar("K-Medoids", n_clusters, sc_kmd, db_kmd, ch_kmd)

        return r

    def _fig_scatter_cluster(self, etiquetas, titulo: str) -> plt.Figure:
        pca  = PCA(n_components=2)
        comp = pca.fit_transform(self._df)
        colores = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12",
                   "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
                   "#16a085", "#c0392b"]
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100) 
        ax.scatter(comp[:, 0], comp[:, 1],
                   c=[colores[l % len(colores)] for l in etiquetas],
                   alpha=0.7)
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.set_title(titulo)
        ax.grid(False)
        plt.tight_layout()
        return fig

    # =========================================================================
    # UMAP
    # =========================================================================
    def UMAP(self, n_componentes: int = 2, n_neighbors: int = 15) -> dict:
        """
        UMAP — reducción de dimensiones.

        Retorna
        -------
        dict con 'fig' (plt.Figure) y 'componentes' (np.ndarray)
        """
        if self._df.isnull().any().any():
            raise ValueError("El DataFrame contiene valores nulos.")
        scaled = StandardScaler().fit_transform(self._df)
        modelo = um.UMAP(n_components=n_componentes, n_neighbors=n_neighbors)
        comp   = modelo.fit_transform(scaled)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)  
        ax.scatter(comp[:, 0], comp[:, 1], alpha=0.7)
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.set_title("UMAP")
        ax.grid(False)
        plt.tight_layout()
        return {"fig": fig, "componentes": comp}

    # =========================================================================
    # T-SNE
    # =========================================================================
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

        scores = []
        for k in range(2, 11):
            labels = KMeans(n_clusters=k, random_state=random_state,
                             n_init="auto").fit_predict(df_tsne)
            scores.append(silhouette_score(df_tsne, labels))

        mejor_k = int(np.argmax(scores)) + 2
        labels  = KMeans(n_clusters=mejor_k, random_state=random_state,
                          n_init="auto").fit_predict(df_tsne)

        # Curva Silhouette
        fig1, ax1 = plt.subplots(figsize=(5, 4), dpi=100)  
        ax1.plot(range(2, 11), scores, marker="o", color="steelblue")
        ax1.set_xlabel("Clusters")
        ax1.set_ylabel("Silhouette Score")
        ax1.set_title("Silhouette Score — t-SNE")
        ax1.grid(True)
        plt.tight_layout()

        # Scatter
        fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=100)  
        sns.scatterplot(x=df_tsne[:, 0], y=df_tsne[:, 1],
                        hue=labels, palette="tab10",
                        legend="full", ax=ax2)
        ax2.set_title("t-SNE")
        ax2.set_xlabel("Dimensión 1")
        ax2.set_ylabel("Dimensión 2")
        ax2.legend(title="Cluster")
        plt.tight_layout()

        self._registrar("t-SNE", mejor_k, max(scores))
        return {"fig_silhouette": fig1, "fig_scatter": fig2,
                "mejor_k": mejor_k, "mejor_score": max(scores)}

    # =========================================================================
    # BENCHMARK
    # =========================================================================
    def get_nombre(self) -> str:
        """Retorna el nombre de la clase. Usado por el sistema MAS."""
        return self.__class__.__name__

    def _registrar(self, alg: str, k: int, score: float,          
                   db: float = float("nan"),
                   ch: float = float("nan")):
        """
        Registra el resultado de un algoritmo para la tabla benchmark.

        Parámetros
        ----------
        alg   : nombre del algoritmo.
        k     : número de clusters detectados.
        score : Silhouette Score.
        db    : Davies-Bouldin Index  (menor es mejor).
        ch    : Calinski-Harabasz Index (mayor es mejor).
        """
        self._log.append({
            "Algoritmo":         alg,
            "Clusters":          k,
            "Silhouette":        round(score, 4),
            "Davies_Bouldin":    round(db, 4) if not np.isnan(float(db)) else float("nan"),
            "Calinski_Harabasz": round(ch, 4) if not np.isnan(float(ch)) else float("nan"),
        })

    def benchmark(self) -> pd.DataFrame:                           
        """
        Tabla comparativa de todos los modelos ejecutados en la sesión.

        Columnas: Algoritmo, Clusters, Silhouette, Davies_Bouldin,
                  Calinski_Harabasz.
        """
        if not self._log:
            return pd.DataFrame(columns=[
                "Algoritmo", "Clusters", "Silhouette",
                "Davies_Bouldin", "Calinski_Harabasz"
            ])
        return pd.DataFrame(self._log)


    def fig_codo(self, max_k: int = 10) -> plt.Figure:
        """Grafico del Metodo del Codo (Elbow Method)."""
        if self._X_scaled is not None:
            X = self._X_scaled
        else:
            from sklearn.preprocessing import StandardScaler as _SS
            cols = self.conocimiento.get("columnas_seleccionadas",
                   self._df.select_dtypes(include="number").columns.tolist())
            X = _SS().fit_transform(self._df[cols].dropna())
        inercias = []
        ks = range(2, max_k + 1)
        for k in ks:
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X)
                inercias.append(km.inertia_)
            except Exception:
                inercias.append(float("nan"))
        k_optimo = self.conocimiento.get("k_optimo", 3)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        ax.plot(list(ks), inercias, marker="o", color="#3b82f6",
                linewidth=2, markersize=6)
        ax.axvline(x=k_optimo, color="#ef4444", linestyle="--",
                   linewidth=1.5, label=f"K optimo = {k_optimo}")
        ax.set_title("Metodo del Codo - K-Means", fontsize=11)
        ax.set_xlabel("Numero de clusters K", fontsize=9)
        ax.set_ylabel("Inercia", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════════════
    # AGENTE CLUSTERING — 7 PASOS DEL AGENTE INTELIGENTE
    # ═══════════════════════════════════════════════════════════════════════

    def percibir(self, df: pd.DataFrame, columnas: list = None,
                 plan_eda: dict = None) -> dict:
        """
        PASO 1 — PERCEPCIÓN.

        Carga el DataFrame, selecciona las columnas numéricas a usar y
        registra el estado inicial del entorno.

        Parámetros
        ----------
        df       : DataFrame a analizar (se trabaja solo con numéricas).
        columnas : lista de columnas a usar (None → todas las numéricas).
        plan_eda : dict del plan de preprocesamiento del AgenteEDA (opcional).
        """
        self.cargar_dataframe(df.reset_index(drop=True))
        num_cols = self._df.select_dtypes(include="number").columns.tolist()

        # ── Auto-exclusión de posibles IDs desde el plan del AgenteEDA ──
        plan      = plan_eda or {}
        ids_eda   = plan.get("eliminar_columnas", [])   # IDs detectados por EDA

        # Detectar IDs localmente si no viene plan_eda
        n_filas_df = max(len(df), 1)
        ids_local  = [
            col for col in num_cols
            if (not pd.api.types.is_float_dtype(df[col]) and
                df[col].nunique() / n_filas_df > 0.95)
        ]

        # Unión de ambas fuentes de IDs
        ids_excluir = list(set(ids_eda + ids_local))

        if columnas:
            columnas = [c for c in columnas if c in num_cols and c not in ids_excluir]
        else:
            columnas = [c for c in num_cols if c not in ids_excluir]

        self.conocimiento.update({
            "columnas_seleccionadas":    columnas,
            "columnas_excluidas_ids":    ids_excluir,
            "n_filas":                   len(df),
            "n_columnas_usadas":         len(columnas),
            "tiene_suficientes_cols":    len(columnas) >= 2,
            "tiene_suficientes_filas":   len(df) >= 30,
            "plan_eda":                  plan,
        })
        return self.conocimiento

    def analizar(self) -> dict:
        """
        PASO 2 — ANÁLISIS.

        Aplica preprocesamiento básico (elimina nulos, escala con
        StandardScaler), calcula la correlación media entre variables
        y determina el rango de K a explorar.
        """
        cols  = self.conocimiento["columnas_seleccionadas"]
        df_sub = self._df[cols].dropna()

        scaler         = StandardScaler()
        X_scaled       = scaler.fit_transform(df_sub)
        self._X_scaled = np.array(X_scaled, copy=True)

        # Correlación promedio entre variables
        if len(cols) >= 2:
            corr_arr = df_sub.corr().abs().to_numpy(copy=True)
            np.fill_diagonal(corr_arr, 0)
            corr_media = float(corr_arr.mean())
        else:
            corr_media = 0.0

        n_filas = len(df_sub)
        k_max   = min(10, max(2, n_filas // 10))

        self.conocimiento.update({
            "n_filas_limpias":  n_filas,
            "correlacion_media": round(corr_media, 4),
            "k_rango":          (2, k_max),
        })
        return self.conocimiento

    def razonar(self) -> list:
        """
        PASO 3 — RAZONAMIENTO.

        Aplica reglas para recomendar algoritmos, número óptimo de clusters
        y estrategia de análisis.

        Retorna lista combinada de alertas y recomendaciones.
        """
        alertas         = []
        recomendaciones = []
        n               = self.conocimiento.get("n_filas_limpias", 0)
        corr            = self.conocimiento.get("correlacion_media", 0)
        k_min, k_max    = self.conocimiento.get("k_rango", (2, 10))

        if not self.conocimiento.get("tiene_suficientes_cols"):
            alertas.append("⚠ ALERTA: Se necesitan al menos 2 columnas "
                           "numéricas para clustering.")

        if n < 30:
            alertas.append(f"⚠ ALERTA: Dataset muy pequeño ({n} filas). "
                           "Los clusters pueden no ser representativos.")
        elif n < 100:
            recomendaciones.append(
                f"Dataset pequeño ({n} filas). Se recomienda K=2 o K=3 "
                "para evitar sobreajuste en el agrupamiento.")

        if corr > 0.6:
            recomendaciones.append(
                f"Correlación media alta entre variables ({corr}). "
                "Se recomienda aplicar ACP antes del clustering para "
                "reducir redundancia.")

        if n > 200:
            recomendaciones.append(
                "Dataset suficientemente grande para DBSCAN. "
                "Útil para detectar ruido y clusters de forma irregular.")

        # Calcular K óptimo con Silhouette
        k_optimo = 2
        if self._X_scaled is not None and n >= 30:
            mejor_score = -1
            for k in range(k_min, k_max + 1):
                try:
                    lbl = KMeans(n_clusters=k, random_state=42,
                                 n_init=10).fit_predict(self._X_scaled)
                    sc  = silhouette_score(self._X_scaled, lbl)
                    if sc > mejor_score:
                        mejor_score = sc
                        k_optimo    = k
                except Exception:
                    pass

        recomendaciones.append(
            f"K óptimo estimado: {k_optimo} clusters "
            f"(según Silhouette Score).")

        self.conocimiento["k_optimo"]          = k_optimo
        self.conocimiento["alertas"]           = alertas
        self.conocimiento["recomendaciones"]   = recomendaciones
        return alertas + recomendaciones

    def decidir_algoritmos(self) -> list:
        """
        PASO 4 — DECISIÓN.

        Selecciona qué algoritmos ejecutar según las características del
        dataset y los resultados del razonamiento.

        Retorna lista de nombres de algoritmos a ejecutar.
        """
        algoritmos = ["KMeans", "HAC"]

        if self.conocimiento.get("n_filas_limpias", 0) > 200:
            algoritmos.append("DBSCAN")

        if self.conocimiento.get("correlacion_media", 0) > 0.6:
            algoritmos.append("ACP")

        self.conocimiento["algoritmos_a_ejecutar"] = algoritmos
        return algoritmos

    def entrenar_y_evaluar(self) -> pd.DataFrame:
        """
        PASO 5 — ENTRENAMIENTO Y EVALUACIÓN.

        Ejecuta cada algoritmo seleccionado llamando a los métodos
        existentes de la clase (KMedia, HAC, ACP) y construye la tabla
        benchmark con Silhouette, Davies-Bouldin y Calinski-Harabasz.

        Retorna DataFrame del benchmark.
        """
        k         = self.conocimiento.get("k_optimo", 3)
        algoritmos = self.conocimiento.get("algoritmos_a_ejecutar", ["KMeans"])
        figuras   = {}

        for alg in algoritmos:
            try:
                if alg == "KMeans":
                    r = self.KMedia(n_clusters=k)
                    figuras["kmedias_scatter"] = r.get("kmedias_scatter")
                    figuras["kmedias_radar"]   = r.get("kmedias_radar")

                elif alg == "HAC":
                    r = self.HAC(n_clusters=k)
                    for nombre, fig in r.get("dendrogramas", {}).items():
                        figuras[f"hac_dendro_{nombre.lower()}"] = fig
                    figuras["hac_centroides"] = r.get("centroides")

                elif alg == "DBSCAN":
                    from sklearn.cluster import DBSCAN as DBSCAN_sk
                    db     = DBSCAN_sk(eps=0.5, min_samples=5)
                    labels = db.fit_predict(self._X_scaled)
                    n_cls  = len(set(labels)) - (1 if -1 in labels else 0)
                    n_ruid = int((labels == -1).sum())
                    if n_cls >= 2:
                        sc  = silhouette_score(self._X_scaled, labels)
                        db_ = davies_bouldin_score(self._X_scaled, labels)
                        ch  = calinski_harabasz_score(self._X_scaled, labels)
                        self._registrar("DBSCAN", n_cls, sc, db_, ch)
                    self.conocimiento["dbscan_ruido"] = n_ruid
                    figuras["dbscan_scatter"] = self._fig_scatter_cluster(
                        labels, f"DBSCAN ({n_cls} clusters)")

                elif alg == "ACP":
                    r = self.ACP(n_componentes=2)
                    figuras["acp_plano"]  = r.get("plano")
                    figuras["acp_circulo"] = r.get("circulo")

            except Exception as e:
                self._errores.append({"paso": f"entrenar_{alg}",
                                       "error": str(e)})

        # Gráfico del Método del Codo
        try:
            figuras["codo"] = self.fig_codo()
        except Exception as e:
            self._errores.append({"paso": "fig_codo", "error": str(e)})

        self.conocimiento["figuras_generadas"] = figuras
        bench = self.benchmark()

        # Mejor algoritmo por Silhouette
        if not bench.empty and "Silhouette" in bench.columns:
            mejor_idx = bench["Silhouette"].idxmax()
            self.conocimiento["mejor_algoritmo"] = bench.loc[mejor_idx, "Algoritmo"]
            self.conocimiento["mejor_silhouette"] = round(
                float(bench.loc[mejor_idx, "Silhouette"]), 4)
        else:
            self.conocimiento["mejor_algoritmo"]  = "N/A"
            self.conocimiento["mejor_silhouette"] = float("nan")

        return bench

    def comunicar(self) -> dict:
        """
        PASO 6 — COMUNICACIÓN CON EL AGENTE COORDINADOR.

        Construye el mensaje estandarizado MAS con los resultados del
        análisis de clustering.

        Retorna dict con estructura: agente, tipo, timestamp, contenido.
        """
        bench = self.benchmark()
        bench_list = bench.to_dict(orient="records") if not bench.empty else []

        mensaje = {
            "agente":    "AgenteClustering",
            "tipo":      "REPORTE_CLUSTERING",
            "timestamp": datetime.now().isoformat(),
            "contenido": {
                "columnas_usadas":       self.conocimiento.get(
                                             "columnas_seleccionadas", []),
                "columnas_excluidas_ids": self.conocimiento.get(
                                             "columnas_excluidas_ids", []),
                "n_filas":               self.conocimiento.get("n_filas", 0),
                "k_optimo":              self.conocimiento.get("k_optimo", 2),
                "algoritmos_ejecutados": self.conocimiento.get(
                                             "algoritmos_a_ejecutar", []),
                "mejor_algoritmo":       self.conocimiento.get(
                                             "mejor_algoritmo", "N/A"),
                "mejor_silhouette":      self.conocimiento.get(
                                             "mejor_silhouette", float("nan")),
                "dbscan_ruido":          self.conocimiento.get("dbscan_ruido", 0),
                "benchmark_resumen":     bench_list,
                "alertas":               self.conocimiento.get("alertas", []),
                "recomendaciones":       self.conocimiento.get(
                                             "recomendaciones", []),
                "errores_etapas":        self._errores,
            },
        }
        self.mensajes.append(mensaje)
        return mensaje

    def ejecutar(self, df: pd.DataFrame, columnas: list = None,
                 plan_eda: dict = None) -> dict:
        """
        PASO 7 — EJECUCIÓN COMPLETA DEL AGENTE.

        Orquesta los 6 pasos anteriores con try/except individual por etapa.

        Parámetros
        ----------
        df       : DataFrame a analizar.
        columnas : columnas numéricas a usar (None → todas).
        plan_eda : plan de preprocesamiento del AgenteEDA (opcional).

        Retorna dict con: mensaje, benchmark, figuras, conocimiento.
        """
        # ── Pre-limpieza antes de pasar al agente ────────────────────────
        try:
            if _limpiar is not None:
                # Obtener variable objetivo del plan EDA o session si existe
                var_obj = (plan_eda or {}).get("variable_objetivo_inferida", None)
                df, _elim, _imp = _limpiar(df, plan_eda=plan_eda,
                                            variable_objetivo=var_obj)
                if _elim:
                    self.conocimiento["pre_eliminadas"] = _elim
                    self.conocimiento["pre_imputadas"]  = _imp
            # Forzar solo columnas numéricas para clustering
            df = df.select_dtypes(include="number").copy()
        except Exception as e:
            self._errores.append({"paso": "pre_limpieza", "error": str(e)})

        for nombre_paso, fn, args in [
            ("percibir",          self.percibir,          (df, columnas, plan_eda)),
            ("analizar",          self.analizar,          ()),
            ("razonar",           self.razonar,           ()),
            ("decidir_algoritmos",self.decidir_algoritmos,()),
            ("entrenar_y_evaluar",self.entrenar_y_evaluar,()),
        ]:
            try:
                fn(*args)
            except Exception as e:
                self._errores.append({"paso": nombre_paso, "error": str(e)})

        try:
            mensaje = self.comunicar()
        except Exception as e:
            self._errores.append({"paso": "comunicar", "error": str(e)})
            mensaje = {}

        return {
            "mensaje":      mensaje,
            "benchmark":    self.benchmark(),
            "figuras":      self.conocimiento.get("figuras_generadas", {}),
            "conocimiento": self.conocimiento,
        }
