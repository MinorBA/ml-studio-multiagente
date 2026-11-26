# =============================================================================
# pages/_shared.py  (prefijo _ = Streamlit lo IGNORA en el menú)
# Módulo de utilidad: persistencia de datos con session_state
# =============================================================================

import streamlit as st
import pandas as pd


def cargar_csv_en_eda():
    """
    Muestra el uploader de CSV SOLO en la página EDA.
    Guarda el DataFrame en session_state para todas las páginas.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Cargar datos")

    archivo = st.sidebar.file_uploader(
        "Subir archivo CSV", type=["csv"], key="eda_uploader"
    )
    sep = st.sidebar.radio(
        "Separador", [",", ";"], horizontal=True, key="eda_sep"
    )

    if archivo is not None:
        try:
            df = pd.read_csv(archivo, sep=sep, decimal=".")
            st.session_state["df_cargado"] = df.copy()
            st.session_state["df_limpio"]  = df.copy()
            st.session_state["nombre_csv"] = archivo.name
        except Exception as e:
            st.sidebar.error(f"Error al leer CSV: {e}")
            return None

    return obtener_df()


def obtener_df():
    """
    Obtiene el DataFrame activo desde session_state.
    Muestra info del dataset en el sidebar (sin uploader).
    Retorna None si no hay datos cargados.
    """
    if "df_limpio" not in st.session_state:
        return None

    df = st.session_state["df_limpio"]
    nombre = st.session_state.get("nombre_csv", "dataset")
    st.sidebar.markdown("---")
    st.sidebar.success(f"Dataset activo: **{nombre}**")
    st.sidebar.caption(f"{df.shape[0]} filas x {df.shape[1]} columnas")
    return df


def actualizar_df(df: pd.DataFrame):
    """Guarda el DataFrame limpio para que otras páginas lo usen."""
    st.session_state["df_limpio"] = df.copy()


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDAD COMPARTIDA: Limpieza previa antes de pasar df a los agentes
# ─────────────────────────────────────────────────────────────────────────────

def limpiar_df_para_agente(df: "pd.DataFrame",
                            plan_eda: dict = None,
                            variable_objetivo: str = None) -> tuple:
    """
    Limpia el DataFrame antes de pasarlo a cualquier agente supervisado o
    de clustering. Aplica en orden:

    1. Elimina columnas 100% vacías (todos NaN).
    2. Elimina columnas marcadas en plan_eda["eliminar_columnas"].
    3. Elimina columnas que son posibles IDs (int, >95% únicos), excepto
       la variable objetivo.
    4. Imputa NaN restantes: mediana en numéricas, moda en categóricas.
    5. Resetea el índice.

    Retorna
    -------
    (df_limpio, columnas_eliminadas, columnas_imputadas)
    """
    import pandas as _pd
    import numpy as _np

    df = df.copy().reset_index(drop=True)
    eliminadas = []
    imputadas  = []
    n          = max(len(df), 1)

    # 1. Columnas 100% vacías
    cols_vacias = [c for c in df.columns if df[c].isnull().all()]
    if cols_vacias:
        df = df.drop(columns=cols_vacias)
        eliminadas.extend(cols_vacias)

    # 2. Columnas del plan EDA
    if plan_eda:
        for c in plan_eda.get("eliminar_columnas", []):
            if c in df.columns and c != variable_objetivo:
                df = df.drop(columns=[c])
                if c not in eliminadas:
                    eliminadas.append(c)

    # ── Helper: detectar si una columna es de tipo texto/string ─────────
    def _es_texto(serie):
        """Funciona en pandas 2.x (dtype 'str') y anteriores ('object')."""
        d = str(serie.dtype)
        return (d in ("object", "string", "str") or
                _pd.api.types.is_object_dtype(serie))

    # 3. Columnas ID: numéricas enteras O strings con >95% únicos
    for c in list(df.columns):
        if c == variable_objetivo:
            continue
        ratio_uniq = df[c].nunique() / n
        es_id_num  = (_pd.api.types.is_integer_dtype(df[c]) and ratio_uniq > 0.95)
        es_id_str  = (_es_texto(df[c]) and ratio_uniq > 0.90)
        if es_id_num or es_id_str:
            df = df.drop(columns=[c])
            if c not in eliminadas:
                eliminadas.append(c)

    # 3b. Columnas string que en realidad son numéricas (ej. "TotalCharges")
    for c in list(df.columns):
        if c == variable_objetivo:
            continue
        if _es_texto(df[c]):
            # Convertir espacios en blanco a NaN antes de intentar parsear
            serie_strip = df[c].astype(str).str.strip().replace("", _np.nan)
            converted   = _pd.to_numeric(serie_strip, errors="coerce")
            pct_conv    = converted.notna().mean()
            if pct_conv >= 0.90:   # al menos 90% convertible → es numérica
                df[c] = converted
                if df[c].isnull().sum() > 0:
                    df[c] = df[c].fillna(df[c].median())

    # 4. Imputar NaN restantes
    for c in df.columns:
        if df[c].isnull().sum() == 0:
            continue
        if _pd.api.types.is_numeric_dtype(df[c]):
            mediana = df[c].median()
            df[c]   = df[c].fillna(mediana)
            imputadas.append(f"{c} (mediana={round(mediana,4)})")
        else:
            moda = df[c].mode()
            if len(moda) > 0:
                df[c] = df[c].fillna(moda[0])
                imputadas.append(f"{c} (moda='{moda[0]}')")

    return df.reset_index(drop=True), eliminadas, imputadas


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS AUTOMÁTICO DE GRÁFICOS — estilo casos de estudio Minor Brenes
# ─────────────────────────────────────────────────────────────────────────────

def observaciones_histogramas(df: "pd.DataFrame") -> list:
    """
    Genera observaciones relevantes de los histogramas al estilo de
    los casos de estudio: destaca sesgos, distribuciones especiales
    y patrones clínicos o de dominio.
    Retorna lista de strings con las 3-4 observaciones más importantes.
    """
    obs = []
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return obs

    sesgadas_der, sesgadas_izq, normales = [], [], []
    for col in num_cols:
        serie = df[col].dropna()
        if len(serie) < 10:
            continue
        skew = float(serie.skew())
        if skew > 1.0:
            sesgadas_der.append((col, round(skew, 2)))
        elif skew < -1.0:
            sesgadas_izq.append((col, round(skew, 2)))
        elif abs(skew) <= 0.5:
            normales.append(col)

    if sesgadas_der:
        cols_str = ", ".join(f"`{c}`" for c, _ in sesgadas_der[:3])
        obs.append(
            f"{cols_str} presentan distribución sesgada hacia la derecha "
            f"(cola larga positiva), indicando presencia de valores atípicamente altos."
        )
    if sesgadas_izq:
        cols_str = ", ".join(f"`{c}`" for c, _ in sesgadas_izq[:3])
        obs.append(
            f"{cols_str} muestran asimetría hacia la izquierda, "
            f"con concentración de valores en rangos superiores."
        )
    if normales:
        cols_str = ", ".join(f"`{c}`" for c in normales[:4])
        obs.append(
            f"La mayoría de las variables ({cols_str}) muestran distribuciones "
            f"aproximadamente normales o ligeramente asimétricas."
        )
    if not obs:
        obs.append(
            "En general las variables muestran rangos normales. "
            "Este es un patrón típico en datos donde la población se concentra "
            "en valores centrales y los casos extremos representan condiciones relevantes."
        )
    return obs[:4]


def observaciones_boxplots(df: "pd.DataFrame") -> list:
    """
    Genera observaciones de outliers en boxplots: destaca las columnas
    con mayor cantidad de outliers y su posición (superior/inferior/ambos).
    """
    obs = []
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return obs

    ambos, superior, inferior, sin_outliers = [], [], [], []
    for col in num_cols:
        serie = df[col].dropna()
        if len(serie) < 10:
            continue
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        hay_sup = (serie > q3 + 1.5 * iqr).any()
        hay_inf = (serie < q1 - 1.5 * iqr).any()
        if hay_sup and hay_inf:
            ambos.append(col)
        elif hay_sup:
            superior.append(col)
        elif hay_inf:
            inferior.append(col)
        else:
            sin_outliers.append(col)

    if ambos:
        cols_str = ", ".join(f"`{c}`" for c in ambos[:4])
        obs.append(
            f"{cols_str} presentan outliers en ambos extremos (superior e inferior), "
            f"lo que indica variabilidad tanto en valores muy bajos como muy altos."
        )
    if superior:
        cols_str = ", ".join(f"`{c}`" for c in superior[:3])
        obs.append(
            f"{cols_str} muestran outliers concentrados principalmente en el extremo "
            f"superior, sugiriendo la presencia de muestras con valores excepcionalmente altos."
        )
    if inferior:
        cols_str = ", ".join(f"`{c}`" for c in inferior[:3])
        obs.append(
            f"{cols_str} presentan valores atípicos en el extremo inferior, "
            f"indicando posibles registros con mediciones inusualmente bajas."
        )
    if not ambos and not superior and not inferior:
        obs.append(
            "La mayoría de las variables presentan distribuciones compactas "
            "sin outliers significativos, lo que indica buena calidad en los datos."
        )
    return obs[:4]


def observaciones_correlacion(df: "pd.DataFrame",
                               variable_objetivo: str = None) -> list:
    """
    Genera observaciones del heatmap de correlación al estilo de
    los casos de estudio: destaca pares de alta correlación,
    relación con la variable objetivo y multicolinealidad.
    """
    import numpy as _np
    obs = []
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return obs

    corr = df[num_cols].corr().abs()

    # Correlaciones altas entre predictoras (>= 0.7)
    pares_altos = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            if c1 == variable_objetivo or c2 == variable_objetivo:
                continue
            val = corr.loc[c1, c2]
            if val >= 0.7:
                pares_altos.append((c1, c2, round(float(val), 2)))

    # Correlaciones con la variable objetivo
    corr_con_y = {}
    if variable_objetivo and variable_objetivo in num_cols:
        corr_con_y = (corr[variable_objetivo]
                      .drop(variable_objetivo, errors="ignore")
                      .sort_values(ascending=False))

    corr_media_pred = 0.0
    pred_cols = [c for c in num_cols if c != variable_objetivo]
    if len(pred_cols) >= 2:
        arr = df[pred_cols].corr().abs().to_numpy(copy=True)
        _np.fill_diagonal(arr, 0)
        corr_media_pred = round(float(arr.mean()), 3)

    if pares_altos:
        top3 = sorted(pares_altos, key=lambda x: -x[2])[:3]
        pairs_str = ", ".join(f"`{a}`–`{b}` ({v})" for a, b, v in top3)
        obs.append(
            f"Se detectan correlaciones altas entre: {pairs_str}. "
            f"Esto podría indicar multicolinealidad; se recomienda Lasso o Ridge."
        )
    else:
        if corr_media_pred < 0.15:
            obs.append(
                f"Las correlaciones entre las variables predictoras son bajas "
                f"(|r| < 0.15), indicando independencia entre ellas y bajo riesgo "
                f"de multicolinealidad."
            )
        else:
            obs.append(
                f"La correlación media entre predictoras es {corr_media_pred}, "
                f"lo que sugiere cierto grado de asociación lineal entre variables."
            )

    if variable_objetivo and len(corr_con_y) > 0:
        top_pred = corr_con_y.head(3)
        pred_str = ", ".join(f"`{c}` ({round(v,2)})"
                              for c, v in top_pred.items() if v > 0.1)
        max_corr = float(corr_con_y.max())
        if max_corr < 0.15:
            obs.append(
                f"La correlación entre las predictoras y `{variable_objetivo}` es "
                f"muy baja en todos los casos. Esto anticipa que los modelos lineales "
                f"tendrán R² reducido."
            )
        elif pred_str:
            obs.append(
                f"Las variables más correlacionadas con `{variable_objetivo}` son: "
                f"{pred_str}. Estas serán las predictoras más influyentes."
            )

    if not pares_altos:
        obs.append(
            "La ausencia de correlaciones altas entre predictoras es favorable "
            "para los modelos lineales regularizados (Lasso, Ridge)."
        )

    return obs[:4]


def observaciones_clustering(bench_df, k_optimo: int,
                              mejor_algoritmo: str) -> list:
    """
    Genera observaciones del benchmarking de clustering:
    interpreta Silhouette, Davies-Bouldin y K óptimo.
    """
    obs = []
    if bench_df is None or bench_df.empty:
        return obs

    mejor_fila = bench_df.loc[bench_df["Silhouette"].idxmax()]
    sil = float(mejor_fila["Silhouette"])
    db  = float(mejor_fila.get("Davies_Bouldin", float("nan")))

    cal = "excelente" if sil >= 0.7 else "aceptable" if sil >= 0.5 else "débil"
    obs.append(
        f"El mejor agrupamiento es **{mejor_algoritmo}** con K={k_optimo} clusters "
        f"y Silhouette Score de {round(sil,4)} ({cal})."
    )

    if sil < 0.5:
        obs.append(
            "El Silhouette Score bajo sugiere que los clusters no están bien separados. "
            "Puede indicar que los datos no tienen una estructura de agrupamiento clara, "
            "o que se necesita ajustar los parámetros o intentar con más columnas."
        )
    elif sil >= 0.7:
        obs.append(
            "Un Silhouette Score mayor a 0.70 indica clusters bien definidos "
            "y separados entre sí, lo que respalda la validez del agrupamiento encontrado."
        )

    if db == db and db < 0.5:
        obs.append(
            f"El Davies-Bouldin Index de {round(db,4)} es bajo (cercano a 0), "
            f"confirmando que los clusters están compactos e internamente cohesionados."
        )

    n_algs = len(bench_df)
    if n_algs > 1:
        obs.append(
            f"Se evaluaron {n_algs} configuraciones de algoritmos. "
            f"La comparativa permite seleccionar objetivamente el método "
            f"con mejor relación separación/cohesión."
        )

    return obs[:4]


def observaciones_clasificacion(bench_df, mejor_modelo: str,
                                  desbalanceado: bool,
                                  metrica: str, valor: float) -> list:
    """
    Genera observaciones del benchmarking de clasificación al estilo
    de los casos de estudio de Minor.
    """
    obs = []
    if bench_df is None or bench_df.empty:
        return obs

    cal = "excelente" if valor >= 0.9 else "bueno" if valor >= 0.8 else "aceptable"
    obs.append(
        f"El mejor modelo es **{mejor_modelo}** con {metrica}={round(valor,4)} ({cal})."
    )

    if desbalanceado:
        obs.append(
            "El dataset presenta desbalance de clases. Se utilizó SMOTE para "
            "compensarlo y se priorizaron F1 y AUC como métricas principales, "
            "ya que Accuracy puede ser engañosa en estas condiciones."
        )

    if "AUC" in bench_df.columns:
        auc_vals = bench_df["AUC"].dropna().sort_values(ascending=False)
        if len(auc_vals) >= 2:
            top2 = auc_vals.head(2)
            diff = float(top2.iloc[0]) - float(top2.iloc[1])
            if diff < 0.02:
                obs.append(
                    "Los dos mejores modelos presentan AUC muy similares "
                    f"(diferencia < 0.02), sugiriendo que ambos tienen un "
                    f"rendimiento comparable para este dataset."
                )

    if "Accuracy" in bench_df.columns:
        acc_min = float(bench_df["Accuracy"].min())
        if acc_min < 0.7:
            obs.append(
                f"El modelo con menor rendimiento alcanzó solo {round(acc_min,4)} "
                f"de Accuracy, evidenciando la dificultad del problema "
                f"para algunos algoritmos."
            )

    return obs[:4]


def observaciones_regresion(bench_df, mejor_modelo: str,
                              mejor_rmse: float, mejor_r2: float) -> list:
    """
    Genera observaciones del benchmarking de regresión al estilo
    de los casos de estudio de Minor Brenes.
    """
    obs = []
    if bench_df is None or bench_df.empty:
        return obs

    cal = "excelente" if mejor_r2 >= 0.9 else "bueno" if mejor_r2 >= 0.7 else "bajo"
    obs.append(
        f"El mejor modelo es **{mejor_modelo}** con RMSE={round(mejor_rmse,4)} "
        f"y R²={round(mejor_r2,4)} ({cal})."
    )

    if mejor_r2 < 0.3:
        obs.append(
            f"Todos los modelos presentan R² bajo (≤{round(mejor_r2,4)}), lo que "
            f"indica que ningún modelo explica bien la variabilidad del target. "
            f"Esto puede deberse a baja correlación lineal entre las predictoras y el objetivo."
        )
    elif mejor_r2 >= 0.9:
        obs.append(
            f"El R²={round(mejor_r2,4)} indica que el modelo explica más del 90% "
            f"de la variabilidad del target. Excelente ajuste."
        )

    if "RMSE" in bench_df.columns:
        rmse_vals = bench_df["RMSE"].dropna().sort_values()
        if len(rmse_vals) >= 2:
            rango = float(rmse_vals.max()) - float(rmse_vals.min())
            if rango < float(rmse_vals.min()) * 0.1:
                obs.append(
                    "La mayoría de los modelos se concentran en valores de RMSE similares, "
                    "lo que indica que el error es intrínseco al problema más que "
                    "al algoritmo utilizado."
                )

    lineales = ["Múltiple","Lasso CV","Ridge CV","SVM linear"]
    arboles  = ["Árbol","Bosque","XGBoost","Gradient Boost"]
    df_lin   = bench_df[bench_df["Modelo"].isin(lineales)]["RMSE"].min()
    df_arb   = bench_df[bench_df["Modelo"].isin(arboles)]["RMSE"].min()
    if df_lin == df_lin and df_arb == df_arb:
        mejor_fam = "lineales" if df_lin < df_arb else "basados en árboles"
        obs.append(
            f"Los modelos {mejor_fam} obtuvieron menor RMSE en este dataset, "
            f"sugiriendo que la relación entre las variables es "
            f"{'lineal' if mejor_fam == 'lineales' else 'no lineal'}."
        )

    return obs[:4]
