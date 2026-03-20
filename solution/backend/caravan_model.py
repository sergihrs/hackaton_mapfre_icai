"""
Modelo COIL 2000 (solución de referencia): predicción de interés en seguro de caravana.

Carga datos desde data/TICDATA2000.txt. La ruta a data/ busca la raíz del proyecto
tanto si se ejecuta desde backend/ como desde solution/backend/.
"""

from pathlib import Path
from functools import lru_cache
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Ruta a data/: desde backend/ -> parent.parent; desde solution/backend/ -> parent.parent.parent
_here = Path(__file__).resolve().parent
_root = _here.parent.parent
_DATA_DIR = (_root.parent / "data") if _root.name == "solution" else (_root / "data")
_TRAIN_FILE = _DATA_DIR / "TICDATA2000.txt"
_COL_NAMES = [f"M{i}" for i in range(1, 86)] + ["CARAVAN"]
_DEFAULT_FEATURE_COLS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]
_TARGET_COL = "CARAVAN"

_cached_df: pd.DataFrame | None = None
_cached_metrics: dict | None = None


def _load_data() -> pd.DataFrame:
    if not _TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {_TRAIN_FILE}. Ejecuta python scripts/download_coil2000.py desde la raíz."
        )
    return pd.read_csv(_TRAIN_FILE, sep="\t", header=None, names=_COL_NAMES)


def _get_df() -> pd.DataFrame:
    global _cached_df
    if _cached_df is None:
        _cached_df = _load_data()
    return _cached_df


@lru_cache(maxsize=1)
def load_model() -> Any:
    df = _get_df()
    y = (df[_TARGET_COL] > 0).astype(int)
    X = df[_DEFAULT_FEATURE_COLS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)
    global _cached_metrics
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    _cached_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)) if y_test.nunique() > 1 else 0.0,
        "n_features": len(_DEFAULT_FEATURE_COLS),
        "model_type": "LogisticRegression",
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pipeline


def get_feature_names() -> list[str]:
    return _DEFAULT_FEATURE_COLS.copy()


def predict_single(features: list[float]) -> dict:
    model = load_model()
    proba = model.predict_proba([features])[0][1]
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability": float(proba)}


def get_metrics() -> dict | None:
    load_model()
    return _cached_metrics


def get_dataset_info() -> dict | None:
    try:
        df = _get_df()
    except FileNotFoundError:
        return None
    y = (df[_TARGET_COL] > 0).astype(int)
    counts = y.value_counts().to_dict()
    return {
        "n_samples": len(df),
        "n_features": len(_COL_NAMES) - 1,
        "feature_names": get_feature_names(),
        "target_name": _TARGET_COL,
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
    }


def get_dataset_sample(n: int = 100) -> list[dict] | None:
    try:
        df = _get_df()
    except FileNotFoundError:
        return None
    subset = df[get_feature_names() + [_TARGET_COL]].head(n)
    return subset.to_dict(orient="records")
