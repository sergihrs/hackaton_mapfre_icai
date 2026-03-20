# Hackatón IA: COIL 2000 — Seguro de caravana

**Objetivo:** Predecir si un cliente tendrá interés en contratar un **seguro de caravana** usando el dataset **COIL 2000** (UCI). Integrar el flujo completo: **datos → limpieza → modelo → fine-tuning → API FastAPI → frontend Streamlit**, y competir por el mejor rendimiento.

---

## 1. Dataset COIL 2000 (datos y formato)

### Qué es

[Insurance Company Benchmark (COIL 2000)](https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)) — datos de una aseguradora holandesa (CoIL Challenge 2000).

- **Train:** `TICDATA2000.txt` — 5.822 registros, 86 atributos (tab-delimited, sin cabecera).
- **Test/eval:** `TICEVAL2000.txt` — 4.000 registros (opcional para evaluación).
- **Target:** atributo 86 — **CARAVAN** (número de pólizas de caravana). Se binariza (p. ej. > 0 → clase positiva).
- **Atributos:** 1–43 sociodemográficos (por código postal); 44–86 posesión de productos (pólizas, contribuciones).

### Descarga e instalación de los datos

1. Descarga los ficheros desde UCI:
   - [TICDATA2000.txt](https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt) (entrenamiento)
   - [TICEVAL2000.txt](https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt) (evaluación, opcional)
2. Colócalos en la carpeta **`data/`** del proyecto:
   ```
   data/
     TICDATA2000.txt   # obligatorio
     TICEVAL2000.txt   # opcional
   ```
3. Opcional: en la raíz del proyecto ejecuta `python scripts/download_coil2000.py` para descargar automáticamente los ficheros en `data/`.

### Formato de los ficheros

- **Separador:** tabulador (`\t`)
- **Cabecera:** no hay; la primera fila son datos.
- **Columnas:** 86. En el código se usan los nombres `M1`, `M2`, …, `M85` (entrada) y **`CARAVAN`** (objetivo, columna 86).
- **Target CARAVAN:** número de pólizas de caravana; en el modelo se binariza (> 0 → 1).
- **Contenido:** atributos 1–43 = sociodemográficos (tipo de cliente, hogar, edad, educación, ocupación, clase social, vivienda, vehículo, ingresos, etc.); 44–86 = productos (pólizas y contribuciones: coche, moto, vida, incendios, barca, bicicleta, etc.); 86 = CARAVAN (variable a predecir).
- No hay valores faltantes en el dataset original. La clase positiva (interés en seguro de caravana) es **minoritaria**; tenerlo en cuenta en entrenamiento (pesos de clase, AUC-ROC/AUC-PR).

---

## 2. Estructura del repositorio

- **backend/**
  - `caravan_model.py`: modelo COIL 2000. Carga datos, entrena un pipeline por defecto y expone `load_model`, `get_feature_names`, `predict_single`, `get_metrics`, `get_dataset_info`, `get_dataset_sample`. **Aquí trabajarás:** sustituir/mejorar pipeline, feature selection y tuning.
  - `model.py`: interfaz que reexporta el modelo activo (`caravan_model`).
  - `schemas.py`: esquemas Pydantic (PredictRequest, PredictResponse, ModelMetrics, DatasetInfo).
  - `main.py`: FastAPI con `/health`, `/features`, `/predict`, `/metrics`, `/dataset_info`, `/dataset_sample`.
  - `requirements.txt`: dependencias (pandas, scikit-learn, etc.).
- **frontend/**
  - `app.py`: Streamlit con pestañas **Predicción**, **Dataset y variables**, **Resultados del modelo**.
  - `requirements.txt`.
- **data/**  
  Aquí van `TICDATA2000.txt` y, si quieres, `TICEVAL2000.txt` (tras descargarlos).
- **solution/**  
  Solución de referencia. No la uses durante el reto; sirve para contrastar al final.

---

## 3. Preparación del entorno

### 3.1. Crear el entorno virtual

Abre una terminal en la **raíz del proyecto** (donde está `GUIA_HACKATON.md`) y crea el entorno virtual:

```bash
python -m venv .venv
```

Se creará la carpeta `.venv` con un Python aislado para el proyecto. Si tu sistema tiene varios Python, puedes usar `python3 -m venv .venv` o la ruta completa al ejecutable deseado.

### 3.2. Activar el entorno virtual

Hay que **activar** el entorno antes de instalar dependencias o ejecutar la API y Streamlit. Según tu sistema:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  Si aparece un error de política de ejecución, ejecuta primero (como administrador si hace falta):  
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

- **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

- **Linux o macOS:**
  ```bash
  source .venv/bin/activate
  ```

Cuando esté activado, el prompt mostrará algo como `(.venv)` al inicio. A partir de ahí, `python` y `pip` usan el entorno virtual.

### 3.3. Instalar dependencias

Con el entorno virtual **activado**, instala las dependencias del backend y del frontend:

```bash
# Backend (API)
cd backend
pip install -r requirements.txt
cd ..

# Frontend (Streamlit)
cd frontend
pip install -r requirements.txt
cd ..
```

O en una sola línea desde la raíz (con el venv activado):

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 3.4. Desactivar el entorno virtual

Cuando termines de trabajar, puedes desactivar el entorno con:

```bash
deactivate
```

---

## 4. Lanzar los servicios

Asegúrate de tener el **entorno virtual activado** (sección 3.2). En dos terminales:

- **Terminal 1 — API:**  
  Opción A — entrar en `backend/` y arrancar uvicorn (recomendado):
  ```bash
  cd backend
  uvicorn main:app --reload
  ```
  Opción B — sin cambiar de carpeta, desde la raíz del proyecto:
  ```bash
  uvicorn main:app --reload --app-dir backend
  ```
  Comprueba: `http://localhost:8000/health`, `http://localhost:8000/docs`.

- **Terminal 2 — Streamlit:** desde la raíz del proyecto (con el venv activado):
  ```bash
  cd frontend
  streamlit run app.py
  ```
  Abre en el navegador (p. ej. `http://localhost:8501`).

**Importante:** Si no has colocado `TICDATA2000.txt` en `data/`, la API fallará al usar el modelo (p. ej. al llamar a `/predict` o `/metrics`). Descarga los datos antes (sección 1).

---

## 5. Por hacer (tareas del hackatón)

El proyecto arranca con un pipeline mínimo en `backend/caravan_model.py`. Tu trabajo es mejorarlo y completar lo siguiente.

### 5.1. Obtención y carga

- [ ] Descargar COIL 2000 y colocar los ficheros en `data/`.
- [ ] Cargar en pandas (tab-delimited, sin cabecera; nombres según `caravan_model.py`).
- [ ] Inspeccionar dimensiones, tipos y primeras filas.

### 5.2. Limpieza y EDA

- [ ] Comprobar missing y duplicados.
- [ ] Analizar **desbalanceo** de la clase CARAVAN (minoritaria).
- [ ] Decidir binarización del target (p. ej. `CARAVAN > 0` → 1) y documentar la decisión.

### 5.3. Selección de variables

- [ ] Reducir las 86 variables a un conjunto manejable (p. ej. 10–20) por importancia, correlación u otro criterio. Documentar la elección.
- [ ] Actualizar en `caravan_model.py` la lista de features (p. ej. `_DEFAULT_FEATURE_COLS` o la variable que uses) para que el modelo y la API usen ese subconjunto en el mismo orden que `get_feature_names()`.

### 5.4. Elección de modelo

- [ ] Probar al menos 2–3 familias (regresión logística, árbol / Random Forest, XGBoost o LightGBM).
- [ ] Justificar la elección final y sustituir/ajustar el pipeline en `caravan_model.py`.

### 5.5. Fine-tuning

- [ ] Usar validación cruzada (GridSearchCV, RandomizedSearchCV u Optuna) para hiperparámetros.
- [ ] Definir la **métrica principal** a optimizar (p. ej. AUC-ROC o recall@k).

### 5.6. Evaluación

- [ ] Reportar métricas acordadas: **accuracy**, **AUC-ROC**, **AUC-PR**, matriz de confusión; opcional **recall@800**.
- [ ] Asegurarte de que `get_metrics()` en `caravan_model.py` devuelve estas métricas para que la API y Streamlit las muestren.

### 5.7. Integración y Streamlit

- [ ] Comprobar que la API responde correctamente (`/predict`, `/metrics`, `/dataset_info`, `/dataset_sample`).
- [ ] Comprobar que la app Streamlit muestra: (1) **Predicción** con las features del modelo, (2) **Dataset y variables** (resumen y/o tabla), (3) **Resultados del modelo** (métricas).

---

## 6. Criterios de evaluación de la competición

| Criterio | Descripción |
|----------|-------------|
| **Métrica principal** | AUC-ROC en test (o la acordada, p. ej. recall@800). Peso alto en el ranking. |
| **Reproducibilidad** | Código que, partiendo de los ficheros en `data/`, reproduzca el modelo y las métricas. |
| **Calidad del pipeline** | Limpieza y decisiones documentadas; feature selection justificada; uso correcto de train/test. |
| **Streamlit útil** | App con predicción, vista del dataset/variables y resultados del modelo. Claridad y usabilidad. |
| **Bonus (opcional)** | Interpretabilidad (importancia, SHAP); documentación breve del proceso y de hiperparámetros. |

---

## 7. Resumen de criterios

- **Correctitud técnica:** `/predict` funciona y devuelve JSON coherente; Streamlit se conecta a la API y muestra la predicción sin errores.
- **Robustez:** La API maneja entradas incorrectas de forma controlada (longitud de `features`, etc.).
- **Claridad:** La app explica qué predice (interés en seguro de caravana) y el equipo explica el flujo datos → modelo → API → frontend.
- **Competición:** Mejor métrica principal, pipeline justificado y Streamlit completo (predicción + dataset/variables + resultados del modelo).

---

## 8. Extensiones opcionales

- Endpoint `/predict_batch` para varias instancias.
- Visualizaciones en Streamlit (histograma de probabilidades, importancia de variables).
- Mejor manejo de errores y mensajes al usuario.
