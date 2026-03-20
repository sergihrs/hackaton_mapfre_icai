from fastapi import FastAPI, HTTPException

import model
import schemas

app = FastAPI(
    title="Hackatón IA - API COIL 2000 (solución)",
    description="Solución de referencia: predicción seguro de caravana (COIL 2000).",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/features")
def features() -> dict:
    return {"features": model.get_feature_names()}


@app.post("/predict", response_model=schemas.PredictResponse)
def predict(payload: schemas.PredictRequest):
    features_list = payload.features
    if not features_list:
        raise HTTPException(status_code=400, detail="La lista de 'features' no puede estar vacía.")
    expected_len = len(model.get_feature_names())
    if len(features_list) != expected_len:
        raise HTTPException(
            status_code=400,
            detail=f"Se esperaban {expected_len} valores en 'features', pero se recibieron {len(features_list)}.",
        )
    result = model.predict_single(features_list)
    return schemas.PredictResponse(**result)


def _get_metrics():
    fn = getattr(model, "get_metrics", None)
    if fn is None:
        return None
    out = fn()
    return out if isinstance(out, dict) else None


def _get_dataset_info():
    fn = getattr(model, "get_dataset_info", None)
    if fn is None:
        return None
    out = fn()
    return out if isinstance(out, dict) else None


def _get_dataset_sample(n: int = 100):
    fn = getattr(model, "get_dataset_sample", None)
    if fn is None:
        return None
    out = fn(n)
    return out if isinstance(out, list) else None


@app.get("/metrics")
def metrics():
    data = _get_metrics()
    if data is None:
        raise HTTPException(status_code=404, detail="No hay métricas disponibles.")
    return data


@app.get("/dataset_info")
def dataset_info():
    data = _get_dataset_info()
    if data is None:
        raise HTTPException(status_code=404, detail="No hay información del dataset.")
    return data


@app.get("/dataset_sample")
def dataset_sample(n: int = 100):
    data = _get_dataset_sample(n)
    if data is None:
        raise HTTPException(status_code=404, detail="No hay muestra del dataset.")
    return {"sample": data, "n": len(data)}
