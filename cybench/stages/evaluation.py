"""Consistent temporal evaluation used by Stage 1 and Stage 2."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.evaluation.eval import evaluate_predictions
from cybench.stages.data import finite_rows


def evaluate_temporal_folds(
    frame: pd.DataFrame,
    feature_cols: list[str],
    estimator_factory: Callable[[], object],
    test_years: list[int],
) -> tuple[dict, pd.DataFrame]:
    """Fit one estimator per held-out year and return pooled metrics/predictions."""
    predictions = []
    for test_year in test_years:
        train = frame[frame[KEY_YEAR] < test_year].copy()
        test = frame[frame[KEY_YEAR] == test_year].copy()
        train = train[finite_rows(train, feature_cols)]
        test = test[finite_rows(test, feature_cols)]
        if train.empty or test.empty:
            continue
        estimator = estimator_factory()
        estimator.fit(train[feature_cols].to_numpy(), train[KEY_TARGET].to_numpy())
        pred = estimator.predict(test[feature_cols].to_numpy())
        fold = test[[KEY_LOC, KEY_YEAR, KEY_TARGET]].copy()
        fold["prediction"] = np.asarray(pred).reshape(-1)
        predictions.append(fold)
    if not predictions:
        raise ValueError("No evaluable temporal folds were produced")
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics = evaluate_predictions(
        prediction_frame[KEY_TARGET].to_numpy(),
        prediction_frame["prediction"].to_numpy(),
        years=prediction_frame[KEY_YEAR].to_numpy(),
    )
    return metrics, prediction_frame


def predict_training_fit(
    frame: pd.DataFrame,
    feature_cols: list[str],
    estimator_factory: Callable[[], object],
    test_years: list[int],
) -> pd.DataFrame:
    """Fit once before the outer-test boundary and predict that same training set."""
    train = frame[frame[KEY_YEAR] < min(test_years)].copy()
    train = train[finite_rows(train, feature_cols)]
    if train.empty:
        raise ValueError("No training rows exist before the outer-test boundary")
    estimator = estimator_factory()
    estimator.fit(train[feature_cols].to_numpy(), train[KEY_TARGET].to_numpy())
    result = train[[KEY_LOC, KEY_YEAR, KEY_TARGET]].copy()
    result["prediction"] = np.asarray(estimator.predict(train[feature_cols].to_numpy())).reshape(-1)
    result["split"] = "train"
    return result


def metric_row(**metadata) -> dict:
    metrics = metadata.pop("metrics")
    return {**metadata, **{key: float(value) for key, value in metrics.items()}}


METRIC_COLUMNS = ("mape", "normalized_rmse", "r", "r2", "kge")


def metrics_from_prediction_frame(frame: pd.DataFrame) -> dict:
    """Calculate the standard metrics for one saved train or test prediction set."""
    return evaluate_predictions(
        frame[KEY_TARGET].to_numpy(),
        frame["prediction"].to_numpy(),
        years=frame[KEY_YEAR].to_numpy(),
    )


def repeat_seed(base_seed: int, repeat: int) -> int:
    """Return a stable, well-separated seed for a one-based repeat index."""
    if repeat < 1:
        raise ValueError("repeat must be one-based")
    return int(base_seed + (repeat - 1) * 1009)


def summarize_repeated_metrics(
    metrics: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    """Summarize repeated metrics with sample variance (ddof=1)."""
    missing = sorted(set(METRIC_COLUMNS) - set(metrics.columns))
    if missing:
        raise ValueError(f"Missing repeated metric columns: {missing}")
    grouped = metrics.groupby(group_columns, as_index=False, dropna=False)
    mean = grouped[list(METRIC_COLUMNS)].mean().rename(
        columns={column: f"{column}_mean" for column in METRIC_COLUMNS}
    )
    variance = grouped[list(METRIC_COLUMNS)].var(ddof=1).fillna(0.0).rename(
        columns={column: f"{column}_variance" for column in METRIC_COLUMNS}
    )
    return mean.merge(variance, on=group_columns, validate="one_to_one")
