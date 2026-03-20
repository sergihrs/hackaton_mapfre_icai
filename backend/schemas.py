from typing import Any, Dict, List

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Cuerpo de la petición de predicción: lista de features en orden."""

    features: List[float]


class PredictResponse(BaseModel):
    """Respuesta de predicción: etiqueta y probabilidad de la clase positiva."""

    prediction: int
    probability: float


class ModelMetrics(BaseModel):
    """Métricas del modelo expuestas por GET /metrics (campos opcionales)."""

    accuracy: float | None = None
    auc_roc: float | None = None
    auc_pr: float | None = None
    n_features: int | None = None
    model_type: str | None = None
    n_train: int | None = None
    n_test: int | None = None

    class Config:
        extra = "allow"  # permitir más campos (recall_at_k, etc.)


class DatasetInfo(BaseModel):
    """Resumen del dataset expuesto por GET /dataset_info."""

    n_samples: int
    n_features: int
    feature_names: List[str]
    target_name: str
    class_distribution: Dict[str, int]

