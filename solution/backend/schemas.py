from typing import Dict, List

from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    prediction: int
    probability: float


class ModelMetrics(BaseModel):
    accuracy: float | None = None
    auc_roc: float | None = None
    auc_pr: float | None = None
    n_features: int | None = None
    model_type: str | None = None
    n_train: int | None = None
    n_test: int | None = None

    class Config:
        extra = "allow"


class DatasetInfo(BaseModel):
    n_samples: int
    n_features: int
    feature_names: List[str]
    target_name: str
    class_distribution: Dict[str, int]
