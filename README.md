# 🧠 ML Studio — Dashboard Interactivo de Machine Learning

Proyecto modularizado de Machine Learning con backend en Python puro y
frontend interactivo con Streamlit.

---

## 📁 Estructura del proyecto

```
ml_studio/
├── app.py                            ← Punto de entrada Streamlit
├── requirements.txt                  ← Dependencias
├── README.md                         ← Este archivo
├── .streamlit/
│   └── config.toml                   ← Tema y configuración
│
├── backend/                          ← BACKEND — Lógica de ML
│   ├── __init__.py                   ← Re-exportaciones centralizadas
│   ├── eda_base.py                   ← Clase raíz: analisisEDA
│   ├── no_supervisado.py             ← Clustering + ACP (hereda EDA)
│   ├── supervisado.py                ← Clase padre Supervisado
│   ├── clasificacion.py              ← Clasificación (hereda Supervisado)
│   ├── regresion.py                  ← Regresión   (hereda Supervisado)
│   └── matriz_confusion.py           ← Utilidad MC 2×2
│
└── frontend/                         ← FRONTEND — Interfaz Streamlit
    └── pages/
        ├── 1_📊_EDA.py               ← Exploración de datos
        ├── 2_🔬_No_Supervisado.py    ← ACP, HAC, K-Medias, UMAP, t-SNE
        ├── 3_🎯_Clasificacion.py     ← KNN, DT, RF, XGBoost, AdaBoost
        ├── 4_📈_Regresion.py         ← Lineal, Lasso, Ridge, SVM, Árboles
        └── 5_🧮_Matriz_Confusion.py  ← Evaluación manual 2×2
```

---

## 🏗️ Jerarquía de clases

```
analisisEDA                     ← Clase raíz (carga, limpieza, gráficos EDA)
│
├── NoSupervisado               ← Hereda de analisisEDA
│   ├── ACP()                   → Análisis de Componentes Principales
│   ├── HAC()                   → Clustering Jerárquico Aglomerativo
│   ├── KMedia()                → K-Medias y K-Medoids
│   ├── UMAP()                  → Reducción de dimensiones
│   ├── TSNE()                  → t-SNE
│   └── benchmark()             → Tabla comparativa
│
└── Supervisado                 ← Clase padre (preparar_datos, evaluar)
    │
    ├── Clasificacion           ← Hereda de Supervisado
    │   ├── ejecutar_modelo()   → Método genérico por nombre
    │   ├── barrido_knn()       → Barrido de hiperparámetros
    │   ├── barrido_dt()
    │   ├── barrido_rf()
    │   ├── barrido_xg()
    │   ├── barrido_ada()
    │   ├── grid_search_ada()   → GridSearchCV
    │   └── benchmark()         → Tabla comparativa de 5 algoritmos
    │
    └── Regresion               ← Hereda de Supervisado
        ├── RegLinMult()        → Regresión Lineal Múltiple
        ├── RegLasso/CV()       → Lasso / Lasso CV
        ├── RegRidge/CV()       → Ridge / Ridge CV
        ├── SVM()               → SVR (rbf, linear, poly)
        ├── DecisionTreeReg()
        ├── RandomForestReg()
        ├── XGBoostingReg()
        └── comparacion_final() → Tabla con todos los modelos
```

---

## 🚀 Instalación y ejecución

### 1. Clonar / descargar el proyecto

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\\Scripts\\activate           # Windows
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

1. Abrir la carpeta `ml_studio/` como proyecto en VS Code
2. Seleccionar el intérprete Python del entorno virtual
3. Abrir terminal integrada (`Ctrl + ñ`)
4. Ejecutar: `streamlit run app.py`
5. VS Code abrirá automáticamente el navegador

---

## 📌 Notas técnicas

- Todos los métodos de gráfico devuelven `plt.Figure` para compatibilidad
  con `st.pyplot(fig)` de Streamlit.
- El backend NO depende de Streamlit — puede usarse como librería
  independiente desde Jupyter, scripts o tests unitarios.
- La clase `Supervisado` es abstracta: solo provee `preparar_datos()`,
  `predecir()`, `evaluar()` e `indices_general()`.
  Las subclases `Clasificacion` y `Regresion` implementan los algoritmos.
