# 🧠 ML Studio MAS — Sistema Multiagente para Machine Learning

> **Proyecto del curso EIF420O Inteligencia Artificial**, Universidad Nacional de Costa Rica (I ciclo de 2026). Autor: **Minor Brenes Aguilar** ([@MinorBA](https://github.com/MinorBA)).
>
> **Demo en vivo:** _[agrega el link cuando lo publiques en Streamlit Community Cloud]_
>
> **Datos de prueba:** la carpeta `datos/` trae los conjuntos de datos usados en el curso: diabetes, potabilidad del agua, precios de Toyota Corolla y derrames de petróleo.

<!-- Captura: agregar docs/captura-coordinador.png y mostrarla aquí -->

**Lo que demuestra este proyecto:** un sistema multiagente con un coordinador que orquesta cuatro agentes (EDA, clustering, clasificación y regresión). Usa más de 15 algoritmos de scikit-learn y XGBoost, maneja datos desbalanceados con SMOTE y busca hiperparámetros con GridSearchCV y algoritmos genéticos. El back-end está separado del front-end hecho en Streamlit.



Proyecto modularizado de Machine Learning con backend en Python puro y
frontend interactivo con Streamlit. Implementa un sistema multiagente
(MAS) donde cada etapa del ciclo de ML (EDA, Clustering, Clasificación,
Regresión) es un agente inteligente con sus propios 7 pasos de
percepción → análisis → razonamiento → decisión → ejecución →
comunicación, coordinados por un `AgenteCoordinador` superagente.

---

## 📁 Estructura del proyecto

```
ml_studio/
├── app.py                          ← Punto de entrada Streamlit (portada)
├── requirements.txt                ← Dependencias
├── README.md                       ← Este archivo
├── shared_utils.py                 ← Utilidades compartidas (carga CSV,
│                                      limpieza de datos, observaciones
│                                      automáticas para cada agente)
│
├── backend/                        ← BACKEND — Lógica de ML (sin Streamlit)
│   ├── __init__.py                 ← Vacío a propósito (ver nota abajo)
│   ├── eda_base.py                 ← Clase raíz: analisisEDA
│   ├── supervisado.py              ← Clase padre: Supervisado
│   ├── clasificacion.py            ← Clasificacion (hereda Supervisado)
│   ├── regresion.py                ← Regresion    (hereda Supervisado)
│   ├── no_supervisado.py           ← NoSupervisado (hereda analisisEDA)
│   ├── CACP.py                     ← Clase ACP (Análisis de Componentes
│   │                                  Principales), usada por NoSupervisado
│   └── agente_coordinador.py       ← AgenteCoordinador (superagente MAS)
│
└── pages/                          ← FRONTEND — Páginas Streamlit
    ├── 1_EDA.py                    ← AgenteEDA: exploración inteligente
    ├── 2_No_Supervisado.py         ← AgenteClustering: ACP, HAC, K-Medias,
    │                                  K-Medoids, DBSCAN
    ├── 3_Clasificacion.py          ← AgenteClasificacion: KNN, Árbol,
    │                                  Random Forest, AdaBoost, XGBoost,
    │                                  Naive Bayes, Regresión Logística, SVM
    ├── 4_Regresion.py              ← AgenteRegresion: Lineal, Lasso/Ridge
    │                                  CV, SVR, Árbol, Bosque, XGBoost
    └── 5_Coordinador.py            ← AgenteCoordinador: orquesta el flujo
                                       completo y genera el reporte MAS
```

> **Nota sobre `backend/__init__.py`:** se deja vacío intencionalmente.
> Cada página importa directamente el submódulo que necesita
> (`from backend.clasificacion import Clasificacion`), por lo que no hay
> necesidad de re-exportar nada a nivel de paquete. Esto evita que una
> dependencia opcional faltante (p. ej. `prince` o `umap-learn`, usadas
> solo por `NoSupervisado`/`CACP`) rompa la carga de módulos que no la
> necesitan, como `Clasificacion` o `Regresion`.

---

## 🏗️ Jerarquía de clases

```
analisisEDA                       ← Clase raíz (carga, limpieza, gráficos EDA,
│                                    7 pasos del AgenteEDA)
│
├── NoSupervisado                 ← AgenteClustering
│   ├── ACP()                     → delega a backend.CACP.ACP
│   ├── HAC()                     → Clustering Jerárquico (Ward/Average/
│   │                                Single/Complete)
│   ├── KMedia()                  → K-Medias y K-Medoids
│   ├── UMAP() / TSNE()           → Reducción de dimensiones
│   ├── benchmark()                → Tabla comparativa (Silhouette,
│   │                                Davies-Bouldin, Calinski-Harabasz)
│   └── ejecutar()                 → 7 pasos del agente
│
└── Supervisado                   ← Clase padre (preparar_datos, evaluar,
    │                                indices_general)
    │
    ├── Clasificacion             ← AgenteClasificacion
    │   ├── benchmark_completo()  → Entrena los 8 algoritmos del enunciado
    │   ├── benchmark_desbalance()→ Comparación con/sin SMOTE
    │   ├── fig_matriz_confusion()
    │   └── ejecutar()             → 7 pasos del agente
    │
    └── Regresion                 ← AgenteRegresion
        ├── comparacion_completa() → Entrena los 9 modelos del enunciado
        ├── RegLinMult/Lasso/RidgeCV/SVM/Árbol/Bosque/XGBoost
        └── ejecutar()              → 7 pasos del agente, incluye
                                       GridSearchCV y GASearchCV (opcional)

AgenteCoordinador                 ← Superagente MAS (composición, NO herencia)
    ├── usa analisisEDA           → infiere tipo de problema por dataset
    ├── usa NoSupervisado         → siempre ejecuta clustering
    ├── usa Clasificacion         → si la variable objetivo es categórica
    └── usa Regresion             → si la variable objetivo es numérica continua
```

---

## 🚀 Instalación y ejecución

### 1. Clonar / descargar el proyecto

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar
```bash
streamlit run app.py
```

### 5. Abrir en el navegador
```
http://localhost:8501
```

---

## 🔧 Uso desde Visual Studio Code

1. Abrir la carpeta `ml_studio/` como proyecto en VS Code.
2. Seleccionar el intérprete Python del entorno virtual.
3. Abrir terminal integrada.
4. Ejecutar: `streamlit run app.py`.
5. El navegador se abre automáticamente en `http://localhost:8501`.

---

## 🧭 Flujo de uso recomendado

1. **EDA** (`pages/1_EDA.py`): cargue el CSV desde la barra lateral. El
   AgenteEDA infiere automáticamente la variable objetivo y el tipo de
   problema (clasificación, regresión o no supervisado), genera el plan
   de preprocesamiento y lo deja disponible para las demás páginas.
2. **Clustering** (`pages/2_No_Supervisado.py`): siempre disponible,
   independientemente del tipo de problema.
3. **Clasificación o Regresión** (`pages/3_Clasificacion.py` /
   `pages/4_Regresion.py`): se habilita según el tipo de problema
   inferido en el paso 1.
4. **Coordinador MAS** (`pages/5_Coordinador.py`): ejecuta el flujo
   completo de forma automática sobre el dataset cargado y permite
   descargar el reporte consolidado en JSON.

---

## 🛡️ Validaciones de robustez incorporadas

El backend incluye validaciones defensivas para datasets pequeños o con
clases muy desbalanceadas, evitando errores no controlados durante el
entrenamiento:

- **Split estratificado seguro:** antes de dividir en train/test con
  `stratify=y`, se verifica que cada clase tenga al menos 2 registros.
  Si alguna clase no cumple ese mínimo, se informa la situación en
  `errores_etapas` y se continúa con un split sin estratificar.
- **SMOTE adaptativo:** el número de vecinos (`k_neighbors`) se ajusta
  automáticamente al tamaño de la clase minoritaria en el conjunto de
  entrenamiento, evitando fallos cuando hay muy pocas muestras.
- **Gráficos comparativos protegidos:** los benchmarks de Clasificación
  y Regresión verifican que exista al menos un modelo con métricas
  válidas antes de intentar graficar; si todos los modelos fallan, se
  registra un mensaje descriptivo en lugar de lanzar una excepción.
- **Fallback de métrica principal:** si la métrica principal (p. ej.
  AUC) no puede calcularse para ningún modelo, el sistema reintenta
  automáticamente con Accuracy para seleccionar el mejor modelo.
- **Matriz de confusión consistente:** se fuerza la lista completa de
  clases como `labels` en `confusion_matrix`, evitando un desajuste de
  tamaño cuando alguna clase no aparece en el conjunto de test.

Todos estos avisos quedan visibles en la sección **"Ver errores de
etapas"** de cada página, junto con el resto de mensajes de diagnóstico
del agente.

---

## 📌 Notas técnicas

- Todos los métodos de gráfico devuelven `plt.Figure` (Matplotlib) o
  `go.Figure` (Plotly) para integrarse con `st.pyplot()` / 
  `st.plotly_chart()` de Streamlit.
- El backend NO depende de Streamlit — puede usarse como librería
  independiente desde Jupyter, scripts o pruebas unitarias.
- `plotly` y `sklearn-genetic-opt` son opcionales: si no están
  instalados, el sistema detecta su ausencia y usa alternativas
  (Matplotlib en vez de Plotly, GridSearchCV en vez de GASearchCV) sin
  interrumpir el flujo.
