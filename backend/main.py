from fastapi import FastAPI, HTTPException

import model
import schemas


app = FastAPI(
    title="Hackatón IA - API COIL 2000 (seguro de caravana)",
    description=(
        "Servicio FastAPI que expone un modelo de clasificación sobre el dataset "
        "COIL 2000 (UCI): predicción de interés en seguro de caravana.\n\n"
        "Endpoints: /health, /features, /predict, /metrics, /dataset_info, /dataset_sample."
    ),
)


@app.get("/health")
def health() -> dict:
    """
    Endpoint mínimo de salud.

    Los alumnos pueden usarlo para comprobar que la API está levantada.
    """

    return {"status": "ok"}


@app.get("/features")
def features() -> dict:
    """
    Devuelve la lista de nombres de variables esperadas por el modelo.

    Este endpoint es opcional para el hackatón, pero puede ayudar a depurar.
    """

    return {"features": model.get_feature_names()}


@app.post("/predict", response_model=schemas.PredictResponse)
def predict(payload: schemas.PredictRequest):
    """
    Endpoint de predicción.

    Implementa el endpoint de predicción usando el modelo cargado en `backend.model`.

    Este ejemplo asume que:
      - `payload.features` es una lista de floats en el mismo orden que `model.get_feature_names()`.
      - `model.predict_single(...)` devuelve al menos las claves `prediction` y `probability`.
    """

    features = payload.features

    if not features:
        raise HTTPException(status_code=400, detail="La lista de 'features' no puede estar vacía.")

    expected_len = len(model.get_feature_names())
    if len(features) != expected_len:
        raise HTTPException(
            status_code=400,
            detail=f"Se esperaban {expected_len} valores en 'features', pero se recibieron {len(features)}.",
        )

    result = model.predict_single(features)
    return schemas.PredictResponse(**result)


def _get_metrics():
    """Llama a model.get_metrics() si existe."""
    get_metrics_fn = getattr(model, "get_metrics", None)
    if get_metrics_fn is None:
        return None
    out = get_metrics_fn()
    return out if isinstance(out, dict) else None


def _get_dataset_info():
    """Llama a model.get_dataset_info() si existe."""
    get_info_fn = getattr(model, "get_dataset_info", None)
    if get_info_fn is None:
        return None
    out = get_info_fn()
    return out if isinstance(out, dict) else None


def _get_dataset_sample(n: int = 100):
    """Llama a model.get_dataset_sample(n) si existe."""
    get_sample_fn = getattr(model, "get_dataset_sample", None)
    if get_sample_fn is None:
        return None
    out = get_sample_fn(n)
    return out if isinstance(out, list) else None


@app.get("/metrics")
def metrics():
    """
    Métricas del modelo (accuracy, AUC-ROC, etc.).

    Disponible si el módulo de modelo implementa get_metrics().
    """
    data = _get_metrics()
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No hay métricas disponibles. Implementa get_metrics() en el módulo de modelo.",
        )
    return data


@app.get("/dataset_info")
def dataset_info():
    """
    Resumen del dataset (n_samples, feature_names, target, distribución de clases).

    Disponible si el módulo de modelo implementa get_dataset_info().
    """
    data = _get_dataset_info()
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No hay información del dataset. Implementa get_dataset_info() en el módulo de modelo.",
        )
    return data


@app.get("/dataset_sample")
def dataset_sample(n: int = 100):
    """
    Muestra hasta n filas del dataset para la pestaña Streamlit.

    Disponible si el módulo de modelo implementa get_dataset_sample(n).
    """
    data = _get_dataset_sample(n)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No hay muestra del dataset. Implementa get_dataset_sample(n) en el módulo de modelo.",
        )
    return {"sample": data, "n": len(data)}

