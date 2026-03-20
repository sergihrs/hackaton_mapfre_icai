"""
Módulo de interfaz del modelo para la API.

Este archivo define la *interfaz mínima* que debe exponer cualquier modelo
para poder ser usado por la API FastAPI y el frontend de Streamlit.

Funciones obligatorias:
  - load_model(): carga o entrena el modelo y lo devuelve.
  - get_feature_names() -> list[str]: nombres de las variables de entrada
    en el mismo orden en el que se espera el vector `features`.
  - predict_single(features: list[float]) -> dict: realiza una predicción
    para una sola instancia y devuelve, como mínimo:
        {"prediction": int, "probability": float}

Funciones opcionales (usadas por GET /metrics, /dataset_info, /dataset_sample):
  - get_metrics() -> dict | None
  - get_dataset_info() -> dict | None
  - get_dataset_sample(n: int) -> list[dict] | None

El modelo activo es el de COIL 2000 (seguro de caravana) en caravan_model.py.
"""

from caravan_model import (
    get_dataset_info,
    get_dataset_sample,
    get_feature_names,
    get_metrics,
    load_model,
    predict_single,
)
