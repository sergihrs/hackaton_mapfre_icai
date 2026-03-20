"""
App Streamlit (solución): COIL 2000 - predicción de interés en seguro de caravana.
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"


def fetch_json(endpoint: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params or {}, timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


st.set_page_config(page_title="COIL 2000 - Seguro de caravana (solución)", layout="wide")
st.title("Hackatón IA - COIL 2000: Seguro de caravana (solución)")
st.markdown(
    "Predicción de **interés en seguro de caravana** con el dataset UCI COIL 2000. "
    "Pestañas: Predicción, Dataset y variables, Resultados del modelo."
)

tab_pred, tab_dataset, tab_metrics = st.tabs(["Predicción", "Dataset y variables", "Resultados del modelo"])

with tab_pred:
    st.header("Predicción")
    features_data = fetch_json("/features")
    feature_names = features_data.get("features", []) if features_data else []
    if not feature_names:
        st.warning("No se pudo cargar /features. ¿Está la API levantada?")
    else:
        n_cols = min(3, len(feature_names))
        cols = st.columns(n_cols)
        values = []
        for i, name in enumerate(feature_names):
            with cols[i % n_cols]:
                values.append(float(st.number_input(name, min_value=0.0, value=0.0, key=f"feat_{name}")))
        st.write("Vector de entrada:", values)
        if st.button("Predecir"):
            try:
                r = requests.post(f"{API_URL}/predict", json={"features": values}, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    proba = data.get("probability", 0)
                    st.success(f"Predicción: {data.get('prediction')} — Probabilidad: {proba:.3f}")
                    st.markdown("**Interpretación:** Alta probabilidad de interés en seguro de caravana." if proba >= 0.5 else "**Interpretación:** Baja probabilidad.")
                else:
                    st.error(f"Error {r.status_code}: {r.text}")
            except Exception as e:
                st.exception(e)

with tab_dataset:
    st.header("Dataset y variables")
    info = fetch_json("/dataset_info")
    if not info:
        st.info("No hay dataset_info. Implementa get_dataset_info() en el modelo.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Muestras", info.get("n_samples", "—"))
        c2.metric("Variables", info.get("n_features", "—"))
        c3.metric("Target", info.get("target_name", "—"))
        c4.metric("Distribución", str(info.get("class_distribution", {})))
        st.subheader("Variables del modelo")
        st.write(", ".join(info.get("feature_names", [])))
        sample_data = fetch_json("/dataset_sample", {"n": 50})
        if sample_data and "sample" in sample_data:
            import pandas as pd
            st.dataframe(pd.DataFrame(sample_data["sample"]), use_container_width=True)

with tab_metrics:
    st.header("Resultados del modelo")
    metrics_data = fetch_json("/metrics")
    if not metrics_data:
        st.info("No hay métricas. Implementa get_metrics() en el modelo.")
    else:
        m = metrics_data
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}" if m.get('accuracy') is not None else "—")
        col2.metric("AUC-ROC", f"{m.get('auc_roc', 0):.3f}" if m.get('auc_roc') is not None else "—")
        col3.metric("Modelo", m.get("model_type", "—"))
        col4.metric("N features", m.get("n_features", "—"))
