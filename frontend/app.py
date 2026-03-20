"""
App Streamlit para el hackatón COIL 2000: predicción de interés en seguro de caravana.

Muestra: predicción con las features del modelo, resumen del dataset y variables,
y resultados del modelo (métricas). La API debe estar levantada (uvicorn en backend/).
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def fetch_json(endpoint: str, params: dict | None = None):
    """GET a la API; devuelve None si 404 o error."""
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params or {}, timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


st.set_page_config(
    page_title="COIL 2000 - Seguro de caravana",
    layout="wide",
)

st.title("Hackatón IA - COIL 2000: Seguro de caravana")
st.markdown(
    "Predicción de **interés en seguro de caravana** con el dataset UCI Insurance Company "
    "Benchmark (COIL 2000). Usa las pestañas para **predecir**, ver **dataset y variables**, "
    "y los **resultados del modelo** (métricas)."
)

health = fetch_json("/health")
if health and health.get("status") == "ok":
    st.success(f"API conectada en {API_URL}")
else:
    st.warning(
        "No hay conexión con la API. Lanza el backend con: uv run uvicorn backend.main:app --reload"
    )

tab_pred, tab_dataset, tab_metrics = st.tabs(["Predicción", "Dataset y variables", "Resultados del modelo"])

# --- Pestaña Predicción ---
with tab_pred:
    st.header("Predicción")
    st.markdown("Introduce los valores de las variables del modelo y obtén la probabilidad de interés en seguro de caravana.")

    features_data = fetch_json("/features")
    if not features_data or "features" not in features_data:
        st.warning("No se pudo cargar la lista de variables. Comprueba que la API está levantada (`uvicorn main:app --reload` en `backend/`).")
        feature_names = []
    else:
        feature_names = features_data["features"]

    if feature_names:
        n_cols = min(3, len(feature_names))
        cols = st.columns(n_cols)
        values = []
        for i, name in enumerate(feature_names):
            with cols[i % n_cols]:
                val = st.number_input(name, min_value=0.0, value=0.0, key=f"feat_{name}")
                values.append(float(val))
        st.write("Vector de entrada:", values)

        if st.button("Predecir"):
            try:
                r = requests.post(f"{API_URL}/predict", json={"features": values}, timeout=5)
                if r.status_code != 200:
                    st.error(f"Error en la API ({r.status_code}): {r.text}")
                else:
                    data = r.json()
                    pred = data.get("prediction")
                    proba = data.get("probability")
                    st.success(f"Predicción: {pred} — Probabilidad (interés en seguro de caravana): {proba:.3f}")
                    if proba is not None:
                        if proba >= 0.5:
                            st.markdown("**Interpretación:** Alta probabilidad de interés en seguro de caravana (clase positiva).")
                        else:
                            st.markdown("**Interpretación:** Baja probabilidad de interés en seguro de caravana (clase negativa).")
            except Exception as e:
                st.exception(e)

# --- Pestaña Dataset y variables ---
with tab_dataset:
    st.header("Dataset y variables")
    info = fetch_json("/dataset_info")
    if not info:
        st.info(
            "No hay información del dataset disponible. Implementa `get_dataset_info()` en el módulo de modelo "
            "(por ejemplo en `backend/caravan_model.py`) para que la API exponga este resumen."
        )
    else:
        st.subheader("Resumen")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Muestras", info.get("n_samples", "—"))
        c2.metric("Variables (modelo)", info.get("n_features", "—"))
        c3.metric("Variable objetivo", info.get("target_name", "—"))
        dist = info.get("class_distribution", {})
        if dist:
            c4.metric("Distribución de clases", str(dist))

        st.subheader("Variables usadas por el modelo")
        names = info.get("feature_names", [])
        if names:
            st.write(", ".join(names))
        else:
            # Fallback a /features
            fd = fetch_json("/features")
            if fd and "features" in fd:
                st.write(", ".join(fd["features"]))

        st.subheader("Muestra del dataset")
        sample_data = fetch_json("/dataset_sample", {"n": 50})
        if sample_data and "sample" in sample_data:
            import pandas as pd  # noqa: F401
            df = pd.DataFrame(sample_data["sample"])
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("El endpoint /dataset_sample no está disponible o el dataset no está cargado.")

# --- Pestaña Resultados del modelo ---
with tab_metrics:
    st.header("Resultados del modelo")
    st.markdown("Métricas del modelo expuestas por la API (calculadas en validación/test).")
    metrics_data = fetch_json("/metrics")
    if not metrics_data:
        st.info(
            "No hay métricas disponibles. Implementa `get_metrics()` en el módulo de modelo "
            "(por ejemplo en `backend/caravan_model.py`) para que la API exponga accuracy, AUC-ROC, etc."
        )
    else:
        m = metrics_data
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}" if m.get('accuracy') is not None else "—")
        col2.metric("AUC-ROC", f"{m.get('auc_roc', 0):.3f}" if m.get('auc_roc') is not None else "—")
        col3.metric("Tipo de modelo", m.get("model_type", "—"))
        col4.metric("Variables", m.get("n_features", "—"))
        if m.get("n_train") is not None or m.get("n_test") is not None:
            st.caption(f"Entrenamiento: {m.get('n_train', '—')} muestras | Test: {m.get('n_test', '—')} muestras.")
        # Cualquier otra métrica (auc_pr, recall_at_k, etc.)
        extra = {k: v for k, v in m.items() if k not in ("accuracy", "auc_roc", "model_type", "n_features", "n_train", "n_test")}
        if extra:
            st.json(extra)
