# =============================================================================
# backend/agente_coordinador.py
# CLASE: AgenteCoordinador  —  Superagente MAS
# ─────────────────────────────────────────────────────────────────────────────
# Orquesta analisisEDA, NoSupervisado, Clasificacion y Regresion
# siguiendo la arquitectura MAS con relación de composición.
# =============================================================================

import os
import pandas as pd
from datetime import datetime

try:
    from shared_utils import limpiar_df_para_agente as _limpiar
except ImportError:
    _limpiar = None

from backend.eda_base       import analisisEDA
from backend.no_supervisado import NoSupervisado
from backend.clasificacion  import Clasificacion
from backend.regresion      import Regresion


class AgenteCoordinador:
    """
    Superagente MAS que coordina por composición a los 4 agentes
    especializados del sistema de Machine Learning.

    Uso típico
    ----------
    coord  = AgenteCoordinador()
    result = coord.ejecutar(df)           # inferencia automática
    result = coord.ejecutar(df, var="Y")  # variable objetivo explícita
    """

    def __init__(self):
        # ── Agentes especializados (lazy init) ─────────────────────────── #
        self.agente_eda            = None
        self.agente_clustering     = None
        self.agente_clasificacion  = None
        self.agente_regresion      = None

        # ── Estado del coordinador ──────────────────────────────────────── #
        self.conocimiento: dict = {}
        self.mensajes:     list = []
        self._errores:     list = []
        self.resultados:   dict = {}   # resultados por agente

    # ═══════════════════════════════════════════════════════════════════════
    # PASO 1 — PERCEPCIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def percibir(self, fuente, variable_objetivo: str = None,
                 use_sql: bool = False, db_params: dict = None) -> dict:
        """
        Carga uno o varios datasets y registra la estructura del entorno.

        Parámetros
        ----------
        fuente           : pd.DataFrame | str (ruta CSV o carpeta).
        variable_objetivo: columna objetivo (None → inferencia automática).
        use_sql          : True para cargar desde base de datos SQL.
        db_params        : dict con credenciales SQL.
        """
        datasets = {}

        try:
            if use_sql and db_params:
                from sqlalchemy import create_engine
                conn_str = (
                    f"mssql+pymssql://{db_params['username']}:"
                    f"{db_params['password']}@{db_params['host']}:"
                    f"{db_params['port']}/{db_params['database']}"
                )
                engine = create_engine(conn_str)
                query  = db_params.get("query", "SELECT TOP 1000 * FROM datos")
                df_sql = pd.read_sql(query, engine)
                datasets["sql_dataset"] = df_sql

            elif isinstance(fuente, pd.DataFrame):
                datasets["dataset"] = fuente.copy()

            elif isinstance(fuente, str) and os.path.isdir(fuente):
                for fname in sorted(os.listdir(fuente)):
                    if fname.endswith(".csv"):
                        try:
                            df_i = pd.read_csv(os.path.join(fuente, fname))
                            datasets[fname.replace(".csv", "")] = df_i
                        except Exception as e:
                            self._errores.append(
                                {"paso": f"cargar_{fname}", "error": str(e)})

            elif isinstance(fuente, str) and os.path.isfile(fuente):
                df_f = pd.read_csv(fuente)
                nombre = os.path.basename(fuente).replace(".csv", "")
                datasets[nombre] = df_f

        except Exception as e:
            self._errores.append({"paso": "percibir_carga", "error": str(e)})

        self.conocimiento.update({
            "n_datasets":        len(datasets),
            "nombres_datasets":  list(datasets.keys()),
            "shapes":            {k: list(v.shape) for k, v in datasets.items()},
            "variable_objetivo": variable_objetivo,
            "datasets":          datasets,
        })
        return self.conocimiento

    # ═══════════════════════════════════════════════════════════════════════
    # PASO 2 — ANÁLISIS
    # ═══════════════════════════════════════════════════════════════════════

    def analizar(self) -> dict:
        """
        Para cada dataset determina el tipo de problema usando la
        heurística del AgenteEDA.
        """
        tipos = {}
        var_obj_global = self.conocimiento.get("variable_objetivo")

        for nombre, df in self.conocimiento.get("datasets", {}).items():
            try:
                # Usar AgenteEDA solo para inferir el tipo
                eda_tmp = analisisEDA()
                eda_tmp.percibir(df)
                tipo = eda_tmp.inferir_tipo_problema(var_obj_global)
                tipos[nombre] = {
                    "tipo":             tipo,
                    "variable_objetivo": eda_tmp.conocimiento.get(
                                             "variable_objetivo"),
                    "subtipo":          eda_tmp.conocimiento.get("subtipo"),
                    "justificacion":    eda_tmp.conocimiento.get(
                                             "justificacion_objetivo"),
                }
            except Exception as e:
                tipos[nombre] = {"tipo": "error", "error": str(e)}

        self.conocimiento["tipos_problema"] = tipos
        return self.conocimiento

    # ═══════════════════════════════════════════════════════════════════════
    # PASO 3 — RAZONAMIENTO
    # ═══════════════════════════════════════════════════════════════════════

    def razonar(self) -> list:
        """
        Construye el plan de ejecución para cada dataset y genera
        justificaciones legibles.
        """
        planes = {}
        justificaciones = []

        for nombre, info in self.conocimiento.get("tipos_problema", {}).items():
            tipo = info.get("tipo", "no_supervisado")
            var  = info.get("variable_objetivo")

            plan = ["EDA", "Clustering"]     # siempre se ejecutan

            if tipo == "clasificacion":
                plan.append("Clasificacion")
                justificaciones.append(
                    f"[{nombre}] Variable '{var}' es categórica → "
                    f"se ejecutará Clasificación."
                )
            elif tipo == "regresion":
                plan.append("Regresion")
                justificaciones.append(
                    f"[{nombre}] Variable '{var}' es numérica continua → "
                    f"se ejecutará Regresión."
                )
            else:
                justificaciones.append(
                    f"[{nombre}] Sin variable objetivo clara → "
                    f"solo EDA y Clustering."
                )

            planes[nombre] = plan

        self.conocimiento["planes"]          = planes
        self.conocimiento["justificaciones"] = justificaciones
        return justificaciones

    # ═══════════════════════════════════════════════════════════════════════
    # PASO 4 — DECISIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def decidir_agentes(self) -> dict:
        """
        Instancia los agentes necesarios para cada dataset.
        Retorna dict {nombre_dataset: [lista_agentes_activos]}.
        """
        agentes_por_dataset = {}

        for nombre, plan in self.conocimiento.get("planes", {}).items():
            activos = []

            if "EDA" in plan:
                activos.append("EDA")

            if "Clustering" in plan:
                activos.append("Clustering")

            if "Clasificacion" in plan:
                activos.append("Clasificacion")

            if "Regresion" in plan:
                activos.append("Regresion")

            agentes_por_dataset[nombre] = activos

        self.conocimiento["agentes_por_dataset"] = agentes_por_dataset
        return agentes_por_dataset

    # ═══════════════════════════════════════════════════════════════════════
    # PASO 5 — EJECUCIÓN DE AGENTES
    # ═══════════════════════════════════════════════════════════════════════

    def ejecutar_agentes(self,
                         progress_callback=None) -> dict:
        """
        Ejecuta cada agente en el orden correcto para cada dataset.
        Acepta un callback opcional para actualizar barras de progreso.

        progress_callback(step: int, total: int, mensaje: str)
        """
        resultados = {}
        plan_eda_global = {}

        datasets   = self.conocimiento.get("datasets", {})
        tipos      = self.conocimiento.get("tipos_problema", {})
        agentes_m  = self.conocimiento.get("agentes_por_dataset", {})
        total_steps = sum(len(v) for v in agentes_m.values())
        step = 0

        for nombre, df in datasets.items():
            res_dataset  = {}
            agentes_activos = agentes_m.get(nombre, [])
            var = tipos.get(nombre, {}).get("variable_objetivo")

            # ── EDA ──────────────────────────────────────────────────────
            if "EDA" in agentes_activos:
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps,
                                      f"[{nombre}] EDA…")
                try:
                    eda = analisisEDA()
                    res_eda = eda.ejecutar(df, variable_objetivo=var)
                    res_dataset["EDA"] = res_eda
                    # Guardar plan para los siguientes agentes
                    plan_eda_global = res_eda["mensaje"]["contenido"].get(
                        "plan_preprocesamiento", {})
                    # Actualizar variable objetivo inferida
                    var = res_eda["conocimiento"].get(
                        "variable_objetivo", var)
                    tipos[nombre]["variable_objetivo"] = var
                except Exception as e:
                    self._errores.append(
                        {"paso": f"EDA_{nombre}", "error": str(e)})

            # ── Clustering ───────────────────────────────────────────────
            if "Clustering" in agentes_activos:
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps,
                                      f"[{nombre}] Clustering…")
                try:
                    ns  = NoSupervisado(df)
                    res_ns = ns.ejecutar(df, plan_eda=plan_eda_global)
                    res_dataset["Clustering"] = res_ns
                except Exception as e:
                    self._errores.append(
                        {"paso": f"Clustering_{nombre}", "error": str(e)})

            # ── Clasificación ────────────────────────────────────────────
            if "Clasificacion" in agentes_activos and var:
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps,
                                      f"[{nombre}] Clasificación…")
                try:
                    clf = Clasificacion(df, var)
                    res_clf = clf.ejecutar(df, var,
                                           plan_eda=plan_eda_global)
                    res_dataset["Clasificacion"] = res_clf
                except Exception as e:
                    self._errores.append(
                        {"paso": f"Clasificacion_{nombre}", "error": str(e)})

            # ── Regresión ────────────────────────────────────────────────
            if "Regresion" in agentes_activos and var:
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps,
                                      f"[{nombre}] Regresión…")
                try:
                    reg = Regresion(df, var)
                    res_reg = reg.ejecutar(df, var,
                                           plan_eda=plan_eda_global)
                    res_dataset["Regresion"] = res_reg
                except Exception as e:
                    self._errores.append(
                        {"paso": f"Regresion_{nombre}", "error": str(e)})

            resultados[nombre] = res_dataset

        self.resultados = resultados
        return resultados

    # ═══════════════════════════════════════════════════════════════════════
    # PASO 6 — CONSOLIDACIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def consolidar(self) -> dict:
        """
        Extrae métricas clave de cada agente, selecciona el mejor modelo
        global y genera conclusiones automáticas en texto.
        """
        consolidado = {}

        for nombre, res_ds in self.resultados.items():
            resumen = {"dataset": nombre}

            # EDA
            if "EDA" in res_ds:
                eda_c = res_ds["EDA"]["conocimiento"]
                resumen["eda"] = {
                    "filas":          eda_c.get("filas", 0),
                    "columnas":       eda_c.get("columnas", 0),
                    "densidad":       eda_c.get("densidad_datos", 100),
                    "n_alertas":      len(eda_c.get("alertas", [])),
                    "tipo_problema":  eda_c.get("tipo_problema", ""),
                    "variable_obj":   eda_c.get("variable_objetivo", ""),
                    "pre_eliminadas": eda_c.get("pre_eliminadas", []),
                }

            # Clustering
            if "Clustering" in res_ds:
                cl_msg = res_ds["Clustering"]["mensaje"]["contenido"]
                resumen["clustering"] = {
                    "mejor_algoritmo":  cl_msg.get("mejor_algoritmo", "N/A"),
                    "mejor_silhouette": cl_msg.get("mejor_silhouette",
                                                    float("nan")),
                    "k_optimo":         cl_msg.get("k_optimo", 0),
                }

            # Clasificación
            if "Clasificacion" in res_ds:
                clf_msg = res_ds["Clasificacion"]["mensaje"]["contenido"]
                resumen["clasificacion"] = {
                    "mejor_modelo":    clf_msg.get("mejor_modelo", "N/A"),
                    "metrica":         clf_msg.get("metrica_principal", ""),
                    "valor_metrica":   clf_msg.get("valor_metrica",
                                                    float("nan")),
                    "desbalanceado":   clf_msg.get("desbalanceado", False),
                }

            # Regresión
            if "Regresion" in res_ds:
                reg_msg = res_ds["Regresion"]["mensaje"]["contenido"]
                resumen["regresion"] = {
                    "mejor_modelo":  reg_msg.get("mejor_modelo", "N/A"),
                    "mejor_rmse":    reg_msg.get("mejor_rmse", float("nan")),
                    "mejor_r2":      reg_msg.get("mejor_r2", float("nan")),
                }

            consolidado[nombre] = resumen

        # ── Selección del mejor modelo supervisado global ─────────────────
        mejor_global = {"dataset": None, "agente": None,
                        "modelo": None, "metrica": None, "valor": None}
        for nombre, resumen in consolidado.items():
            if "clasificacion" in resumen:
                val = resumen["clasificacion"].get("valor_metrica", 0)
                if mejor_global["valor"] is None or val > mejor_global["valor"]:
                    mejor_global = {
                        "dataset": nombre,
                        "agente":  "Clasificacion",
                        "modelo":  resumen["clasificacion"]["mejor_modelo"],
                        "metrica": resumen["clasificacion"]["metrica"],
                        "valor":   val,
                    }
            if "regresion" in resumen:
                r2 = resumen["regresion"].get("mejor_r2", -1)
                if (mejor_global["agente"] == "Regresion" and
                        r2 > mejor_global.get("valor", -1)):
                    mejor_global = {
                        "dataset": nombre,
                        "agente":  "Regresion",
                        "modelo":  resumen["regresion"]["mejor_modelo"],
                        "metrica": "R²",
                        "valor":   r2,
                    }
                elif mejor_global["agente"] is None:
                    mejor_global = {
                        "dataset": nombre,
                        "agente":  "Regresion",
                        "modelo":  resumen["regresion"]["mejor_modelo"],
                        "metrica": "R²",
                        "valor":   r2,
                    }

        # ── Conclusiones automáticas ──────────────────────────────────────
        conclusiones = self._generar_conclusiones(consolidado, mejor_global)

        self.conocimiento["consolidado"]   = consolidado
        self.conocimiento["mejor_global"]  = mejor_global
        self.conocimiento["conclusiones"]  = conclusiones
        return consolidado

    def _generar_conclusiones(self, consolidado: dict,
                               mejor_global: dict) -> list:
        """Genera lista de conclusiones automáticas en lenguaje natural."""
        conclusiones = []

        for nombre, res in consolidado.items():
            eda = res.get("eda", {})
            if eda:
                densidad = eda.get("densidad", 100)
                alertas  = eda.get("n_alertas", 0)
                pre_elim = eda.get("pre_eliminadas", [])

                if pre_elim:
                    conclusiones.append(
                        f"📋 [{nombre}] Se eliminaron automáticamente "
                        f"{len(pre_elim)} columna(s) sin valor predictivo: "
                        f"{', '.join(pre_elim)}."
                    )
                if densidad < 90:
                    conclusiones.append(
                        f"⚠️ [{nombre}] El dataset tiene solo {densidad}% "
                        f"de datos completos. Se imputaron los valores faltantes."
                    )
                if alertas > 0:
                    conclusiones.append(
                        f"🚨 [{nombre}] El AgenteEDA detectó {alertas} "
                        f"alerta(s) crítica(s) que requieren atención."
                    )

            cl = res.get("clustering", {})
            if cl and cl.get("mejor_silhouette") == cl.get("mejor_silhouette"):
                sil = cl.get("mejor_silhouette", 0)
                alg = cl.get("mejor_algoritmo", "")
                k   = cl.get("k_optimo", 0)
                cal = ("excelente" if sil >= 0.7 else
                       "aceptable" if sil >= 0.5 else "débil")
                conclusiones.append(
                    f"🔬 [{nombre}] Clustering: el mejor agrupamiento es "
                    f"{alg} con K={k} clusters. "
                    f"Silhouette={sil:.4f} ({cal})."
                )

            clf = res.get("clasificacion", {})
            if clf and clf.get("mejor_modelo"):
                val = clf.get("valor_metrica", 0)
                met = clf.get("metrica", "")
                mod = clf.get("mejor_modelo", "")
                cal = ("excelente" if val >= 0.9 else
                       "bueno" if val >= 0.8 else "aceptable")
                conclusiones.append(
                    f"🎯 [{nombre}] Clasificación: mejor modelo {mod} "
                    f"con {met}={val:.4f} ({cal})."
                )
                if clf.get("desbalanceado"):
                    conclusiones.append(
                        f"  ↳ Se detectó desbalance de clases — "
                        f"se recomienda validar con datos adicionales."
                    )

            reg = res.get("regresion", {})
            if reg and reg.get("mejor_modelo"):
                rmse = reg.get("mejor_rmse", float("nan"))
                r2   = reg.get("mejor_r2", float("nan"))
                mod  = reg.get("mejor_modelo", "")
                if r2 == r2:
                    cal = ("excelente" if r2 >= 0.9 else
                           "bueno" if r2 >= 0.7 else "aceptable")
                    conclusiones.append(
                        f"📈 [{nombre}] Regresión: mejor modelo {mod} "
                        f"con R²={r2:.4f} ({cal}), RMSE={rmse:.4f}."
                    )

        if mejor_global.get("modelo"):
            conclusiones.append(
                f"🏆 Mejor modelo global: {mejor_global['modelo']} "
                f"({mejor_global['agente']}) — "
                f"{mejor_global['metrica']}={mejor_global['valor']:.4f} "
                f"sobre dataset '{mejor_global['dataset']}'."
            )

        return conclusiones

    # ═══════════════════════════════════════════════════════════════════════
    # PASO 7 — COMUNICACIÓN
    # ═══════════════════════════════════════════════════════════════════════

    def comunicar(self) -> dict:
        """
        Construye el mensaje MAS consolidado del coordinador.
        """
        mejor = self.conocimiento.get("mejor_global", {})
        mensaje = {
            "agente":    "AgenteCoordinador",
            "tipo":      "REPORTE_COORDINADOR",
            "timestamp": datetime.now().isoformat(),
            "datasets_procesados": self.conocimiento.get(
                                       "nombres_datasets", []),
            "plan_ejecutado":      self.conocimiento.get("planes", {}),
            "mejor_modelo_global": mejor,
            "conclusiones":        self.conocimiento.get("conclusiones", []),
            "consolidado":         self.conocimiento.get("consolidado", {}),
            "errores":             self._errores,
        }
        self.mensajes.append(mensaje)
        return mensaje

    # ═══════════════════════════════════════════════════════════════════════
    # MÉTODO PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════

    def ejecutar(self, fuente, variable_objetivo: str = None,
                 use_sql: bool = False, db_params: dict = None,
                 progress_callback=None) -> dict:
        """
        Punto de entrada principal del sistema MAS.

        Orquesta los 7 pasos con try/except individual.

        Retorna dict completo con: mensaje, resultados_por_agente,
        consolidado, conclusiones, errores.
        """
        pasos = [
            ("percibir",        self.percibir,
             (fuente, variable_objetivo, use_sql, db_params)),
            ("analizar",        self.analizar,        ()),
            ("razonar",         self.razonar,         ()),
            ("decidir_agentes", self.decidir_agentes, ()),
        ]

        for nombre, fn, args in pasos:
            try:
                fn(*args)
            except Exception as e:
                self._errores.append({"paso": nombre, "error": str(e)})

        # Ejecutar agentes con callback de progreso
        try:
            self.ejecutar_agentes(progress_callback=progress_callback)
        except Exception as e:
            self._errores.append({"paso": "ejecutar_agentes", "error": str(e)})

        try:
            self.consolidar()
        except Exception as e:
            self._errores.append({"paso": "consolidar", "error": str(e)})

        try:
            mensaje = self.comunicar()
        except Exception as e:
            self._errores.append({"paso": "comunicar", "error": str(e)})
            mensaje = {}

        return {
            "mensaje":            mensaje,
            "resultados":         self.resultados,
            "consolidado":        self.conocimiento.get("consolidado", {}),
            "conclusiones":       self.conocimiento.get("conclusiones", []),
            "mejor_global":       self.conocimiento.get("mejor_global", {}),
            "planes":             self.conocimiento.get("planes", {}),
            "tipos_problema":     self.conocimiento.get("tipos_problema", {}),
            "justificaciones":    self.conocimiento.get("justificaciones", []),
            "errores":            self._errores,
        }
