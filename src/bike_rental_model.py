"""Reusable training and inference code for the bike rental demand model.

This module is the production-friendly counterpart to
``notebooks/bike_rental_demand_predictor_organised.ipynb``. The notebook is where
the analysis and experiments happen; this file contains only the pieces needed
to retrain, save, load and use the model from another application.

The important idea: the saved joblib artifact is the **whole pipeline**
(ColumnTransformer + OneHotEncoder + RandomForestRegressor), so the application
only has to pass raw feature rows to ``predict``. All preprocessing happens
inside the loaded object, exactly as it did during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "bike_rental_pipeline.joblib"

# Optional offline copy of the UCI dataset. If it does not exist the data is
# downloaded with `ucimlrepo`, exactly like the notebook does.
LOCAL_DATA_PATH = PROJECT_ROOT / "data" / "bike_sharing.csv"

TARGET_COLUMN = "cnt"
DATE_COLUMN = "dteday"

# Canonical input schema expected by the model. `dteday` and `cnt` are not
# model inputs: the date is only used for the chronological split and `cnt` is
# the target.
FEATURE_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

# Nominal codes -> one-hot encoding. Ordered codes (season, yr, mnth, hr) and
# continuous measurements stay numeric; one-hot encoding `hr` in particular
# costs a lot of accuracy with a Random Forest because it breaks the ordered
# hour thresholds into many binary splits.
CATEGORICAL_FEATURES = ["holiday", "weekday", "workingday", "weathersit"]
NUMERICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]

# Features that callers may omit. The value is filled in before prediction so
# the saved model still receives every column it was trained on.
# yr: 0 = 2011, 1 = 2012. Defaulting to 1 means "assume the most recent year",
# i.e. the higher demand level seen in 2012. Change this if your app predicts
# historical 2011-style demand instead.
DEFAULT_FEATURE_VALUES: dict[str, Any] = {"yr": 1}

# Best configuration found in the notebook: the original Random Forest with
# default hyperparameters and random_state=42.
MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 50,
    "max_depth": 25,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "random_state": 42,
}


def build_pipeline(
    categorical_features: Sequence[str] | None = None,
    **model_kwargs: Any,
) -> Pipeline:
    """Return a Pipeline that turns raw feature rows into rental predictions.

    Parameters
    ----------
    categorical_features:
        Columns to one-hot encode. Defaults to ``CATEGORICAL_FEATURES``.
        Pass ``[]`` to keep every input column numeric (the exact setup used
        by the original notebook model), or e.g.
        ``["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
        "weathersit"]`` to encode every code column.
    **model_kwargs:
        Overrides for ``MODEL_PARAMS`` (e.g. ``n_estimators=50``).
    """
    categorical = list(
        CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    )
    unknown = [c for c in categorical if c not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown categorical feature(s): {unknown}")

    numerical = [c for c in FEATURE_COLUMNS if c not in categorical]

    transformers: list[tuple[str, Any, list[str] | str]] = []
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    transformers.append(("numeric", "passthrough", numerical))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    params = {**MODEL_PARAMS, **model_kwargs}
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(**params)),
        ]
    )


def load_dataset() -> pd.DataFrame:
    """Load the UCI Bike Sharing dataset (hourly data), features + target."""
    if LOCAL_DATA_PATH.exists():
        return pd.read_csv(LOCAL_DATA_PATH)

    from ucimlrepo import fetch_ucirepo

    bike_sharing = fetch_ucirepo(id=275)
    return bike_sharing.data.features.join(bike_sharing.data.targets)


def prepare_data(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sort chronologically, drop the date column and make an 80/20 split."""
    data = df.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data = data.sort_values([DATE_COLUMN, "hr"]).reset_index(drop=True)

    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    split_index = int(len(data) * (1 - test_size))

    return (
        features.iloc[:split_index],
        features.iloc[split_index:],
        target.iloc[:split_index],
        target.iloc[split_index:],
    )


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Return holdout metrics for a fitted pipeline."""
    predictions = pipeline.predict(X)
    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
    }


def save_pipeline(pipeline: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist the fitted pipeline (preprocessing + model) with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    return path


def load_pipeline(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a previously saved pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Run `python src/train_model.py` first."
        )
    return joblib.load(path)


def train_and_save(
    path: str | Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    categorical_features: Sequence[str] | None = None,
    **model_kwargs: Any,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit the pipeline on the chronological training set and save it.

    Returns the fitted pipeline and the holdout metrics.
    """
    X_train, X_test, y_train, y_test = prepare_data(load_dataset(), test_size)
    pipeline = build_pipeline(categorical_features, **model_kwargs)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)
    save_pipeline(pipeline, path)
    return pipeline, metrics


def prepare_input(
    records: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Validate incoming rows and put them in the model's expected schema."""
    if isinstance(records, pd.Series):
        records = records.to_dict()
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, Mapping):
        # A single observation: a dict of scalars needs an explicit row.
        frame = pd.DataFrame([records])
    else:
        frame = pd.DataFrame(records)

    # Target is never an input. `dteday` is not used by the model, but it can
    # be used to derive `yr` when that column is omitted.
    if TARGET_COLUMN in frame.columns:
        frame = frame.drop(columns=[TARGET_COLUMN])

    if "yr" not in frame.columns and DATE_COLUMN in frame.columns:
        years = pd.to_datetime(frame[DATE_COLUMN], errors="coerce").dt.year
        # Dataset encoding: 0 = 2011, 1 = 2012. Map later years to 1 so
        # predictions assume the more recent (higher-demand) level.
        frame["yr"] = (years - 2011).clip(lower=0, upper=1).fillna(
            DEFAULT_FEATURE_VALUES.get("yr", 1)
        ).astype(int)

    if DATE_COLUMN in frame.columns:
        frame = frame.drop(columns=[DATE_COLUMN])

    for column, default in DEFAULT_FEATURE_VALUES.items():
        if column not in frame.columns:
            frame[column] = default

    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature column(s): {missing}")

    frame = frame[FEATURE_COLUMNS]
    # JSON/CSV inputs often contain numbers as strings. OneHotEncoder treats a
    # string "1" as a different category from the integer 1 it saw in training.
    for column in CATEGORICAL_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("season", "yr", "mnth", "hr"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def predict(
    records: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    model: Pipeline | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    as_int: bool = False,
) -> np.ndarray:
    """Predict bike rentals from raw feature rows.

    ``records`` can be a single dict, a list of dicts or a DataFrame. The
    loaded pipeline applies the one-hot encoding / passthrough itself.
    """
    fitted_pipeline = model if model is not None else load_pipeline(model_path)
    predictions = np.asarray(
        fitted_pipeline.predict(prepare_input(records)), dtype=float
    )
    if as_int:
        return np.rint(np.clip(predictions, 0, None)).astype(int)
    return predictions
