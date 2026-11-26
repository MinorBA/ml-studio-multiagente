# =============================================================================
# CACP.py
# Clase ACP (Análisis de Componentes Principales) — archivo independiente.
# Utilizada por NoSupervisado en No_Supervisados.py
# =============================================================================

import matplotlib.pyplot as plt
from prince import PCA as PCA_Prince


class ACP:
    """
    Clase auxiliar para el Análisis de Componentes Principales (ACP/PCA).
    Encapsula el modelo Prince PCA y expone gráficos estándar:
      - Plano principal
      - Círculo de correlación
      - Sobreposición plano-círculo
    """

    def __init__(self, datos, n_componentes=2):
        self.__datos         = datos
        self.__modelo        = PCA_Prince(n_components=n_componentes).fit(self.__datos)
        self.__correlacion_var   = self.__modelo.column_correlations
        self.__coordenadas_ind   = self.__modelo.row_coordinates(self.__datos)
        self.__contribucion_ind  = self.__modelo.row_contributions_
        self.__cos2_ind          = self.__modelo.row_cosine_similarities(self.__datos)
        self.__var_explicada     = self.__modelo.percentage_of_variance_

    # ------------------------------------------------------------------
    # Properties (getters / setters)
    # ------------------------------------------------------------------
    @property
    def datos(self):
        return self.__datos

    @datos.setter
    def datos(self, datos):
        self.__datos = datos

    @property
    def modelo(self):
        return self.__modelo

    @modelo.setter
    def modelo(self, modelo):
        self.__modelo = modelo

    @property
    def correlacion_var(self):
        return self.__correlacion_var

    @correlacion_var.setter
    def correlacion_var(self, correlacion_var):
        self.__correlacion_var = correlacion_var

    @property
    def coordenadas_ind(self):
        return self.__coordenadas_ind

    @coordenadas_ind.setter
    def coordenadas_ind(self, coordenadas_ind):
        self.__coordenadas_ind = coordenadas_ind

    @property
    def contribucion_ind(self):
        return self.__contribucion_ind

    @contribucion_ind.setter
    def contribucion_ind(self, contribucion_ind):
        self.__contribucion_ind = contribucion_ind

    @property
    def cos2_ind(self):
        return self.__cos2_ind

    @cos2_ind.setter
    def cos2_ind(self, cos2_ind):
        self.__cos2_ind = cos2_ind

    @property
    def var_explicada(self):
        return self.__var_explicada

    @var_explicada.setter
    def var_explicada(self, var_explicada):
        self.__var_explicada = var_explicada

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------
    def plot_plano_principal(self, ejes=[0, 1], ind_labels=True,
                             titulo='Plano Principal'):
        x = self.__coordenadas_ind[ejes[0]].values
        y = self.__coordenadas_ind[ejes[1]].values
        plt.style.use('ggplot')
        plt.scatter(x, y, color='gray')
        plt.title(titulo)
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        inercia_x = round(self.__var_explicada[ejes[0]], 2)
        inercia_y = round(self.__var_explicada[ejes[1]], 2)
        plt.xlabel(f'Componente {ejes[0]} ({inercia_x}%)')
        plt.ylabel(f'Componente {ejes[1]} ({inercia_y}%)')
        if ind_labels:
            for i, txt in enumerate(self.__coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))

    def plot_circulo(self, ejes=[0, 1], var_labels=True,
                     titulo='Círculo de Correlación'):
        cor = self.__correlacion_var.iloc[:, ejes].values
        plt.style.use('ggplot')
        c = plt.Circle((0, 0), radius=1, color='steelblue', fill=False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        inercia_x = round(self.__var_explicada[ejes[0]], 2)
        inercia_y = round(self.__var_explicada[ejes[1]], 2)
        plt.xlabel(f'Componente {ejes[0]} ({inercia_x}%)')
        plt.ylabel(f'Componente {ejes[1]} ({inercia_y}%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95,
                      color='steelblue', alpha=0.5,
                      head_width=0.05, head_length=0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05,
                         self.__correlacion_var.index[i],
                         color='steelblue', ha='center', va='center')

    def plot_sobreposicion(self, ejes=[0, 1], ind_labels=True,
                           var_labels=True,
                           titulo='Sobreposición Plano-Círculo'):
        x   = self.__coordenadas_ind[ejes[0]].values
        y   = self.__coordenadas_ind[ejes[1]].values
        cor = self.__correlacion_var.iloc[:, ejes]
        scale = min(
            (max(x) - min(x)) / (max(cor.iloc[:, 0]) - min(cor.iloc[:, 0]) + 1e-9),
            (max(y) - min(y)) / (max(cor.iloc[:, 1]) - min(cor.iloc[:, 1]) + 1e-9)
        ) * 0.7
        cor = cor.values
        plt.style.use('ggplot')
        plt.axhline(y=0, color='dimgrey', linestyle='--')
        plt.axvline(x=0, color='dimgrey', linestyle='--')
        inercia_x = round(self.__var_explicada[ejes[0]], 2)
        inercia_y = round(self.__var_explicada[ejes[1]], 2)
        plt.xlabel(f'Componente {ejes[0]} ({inercia_x}%)')
        plt.ylabel(f'Componente {ejes[1]} ({inercia_y}%)')
        plt.scatter(x, y, color='gray')
        if ind_labels:
            for i, txt in enumerate(self.__coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale,
                      color='steelblue', alpha=0.5,
                      head_width=0.05, head_length=0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15,
                         self.__correlacion_var.index[i],
                         color='steelblue', ha='center', va='center')
