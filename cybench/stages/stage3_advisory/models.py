"""Leakage-aware yield response surrogate for Stage 3 scenario evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cybench.config import KEY_TARGET, KEY_YEAR
from cybench.stages.stage3_advisory.schema import CountryYieldData


@dataclass
class YieldResponseModel:
    estimator: ExtraTreesRegressor
    features: tuple[str, ...]
    residual_quantile: float
    validation_metrics: dict
    importance: dict[str, float]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        prediction = np.asarray(self.estimator.predict(frame[list(self.features)]))
        return pd.DataFrame(
            {
                "predicted_yield": prediction,
                "lower_yield": prediction - self.residual_quantile,
                "upper_yield": prediction + self.residual_quantile,
            },
            index=frame.index,
        )


def _estimator(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=seed,
        n_jobs=-1,
    )


def fit_yield_response_model(
    data: CountryYieldData,
    seed: int = 42,
    interval_coverage: float = 0.9,
    training_end_year: int | None = None,
) -> YieldResponseModel:
    """Validate forward, then fit only on rows before the decision year."""
    frame = data.frame
    if training_end_year is not None:
        frame = frame[frame[KEY_YEAR] < training_end_year].copy()
    if frame[KEY_YEAR].nunique() < 2:
        raise ValueError("At least two pre-target years are required to fit Stage 3")
    years = sorted(frame[KEY_YEAR].unique())
    validation_years = years[-max(1, round(len(years) * 0.3)) :]
    observed, predicted = [], []
    for year in validation_years:
        train = frame[frame[KEY_YEAR] < year]
        test = frame[frame[KEY_YEAR] == year]
        if train.empty or test.empty:
            continue
        model = _estimator(seed)
        model.fit(train[list(data.feature_columns)], train[KEY_TARGET])
        observed.extend(test[KEY_TARGET].to_numpy())
        predicted.extend(model.predict(test[list(data.feature_columns)]))
    if not observed:
        raise ValueError("No forward temporal validation fold could be created")
    y_true = np.asarray(observed, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    residual = np.abs(y_true - y_pred)
    final = _estimator(seed)
    final.fit(frame[list(data.feature_columns)], frame[KEY_TARGET])
    return YieldResponseModel(
        estimator=final,
        features=data.feature_columns,
        residual_quantile=float(np.quantile(residual, interval_coverage)),
        validation_metrics={
            "forward_r2": float(r2_score(y_true, y_pred)),
            "forward_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "forward_mae": float(mean_absolute_error(y_true, y_pred)),
            "validation_years": [int(year) for year in validation_years],
            "prediction_interval_coverage": interval_coverage,
            "training_end_year_exclusive": (
                int(training_end_year) if training_end_year is not None else None
            ),
            "n_training_rows": int(len(frame)),
        },
        importance=dict(zip(data.feature_columns, final.feature_importances_)),
    )
