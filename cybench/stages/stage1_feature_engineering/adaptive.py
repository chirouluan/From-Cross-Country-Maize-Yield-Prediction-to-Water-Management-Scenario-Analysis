"""Leakage-safe evidence and inner validation for adaptive Stage 1 features."""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import pandas as pd

from cybench.config import KEY_DATES, KEY_LOC, KEY_TARGET, KEY_YEAR, TIME_SERIES_INPUTS
from cybench.stages.data import (
    build_feature_table,
    chronological_test_years,
    feature_columns,
    merge_features_labels,
)
from cybench.stages.evaluation import evaluate_temporal_folds
from cybench.util.data import data_to_pandas
from cybench.util.features import DEFAULT_FEATURE_CONFIG, unpack_time_series


def _finite_summary(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q05": float(values.quantile(0.05)),
        "q10": float(values.quantile(0.10)),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "q90": float(values.quantile(0.90)),
        "q95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def calibration_weather_and_labels(dataset, calibration_end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily weather and targets restricted to the outer calibration period."""
    weather_columns = TIME_SERIES_INPUTS["meteo"]
    weather = data_to_pandas(
        dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_DATES, *weather_columns]
    )
    weather = unpack_time_series(weather, weather_columns)
    weather = weather[weather[KEY_YEAR] <= calibration_end_year].copy()
    labels = data_to_pandas(
        dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET]
    )
    labels = labels[labels[KEY_YEAR] <= calibration_end_year].copy()
    return weather, labels


def regional_climate_profile(dataset, calibration_end_year: int) -> dict:
    """Build a country profile using calibration predictors and labels only."""
    weather, labels = calibration_weather_and_labels(dataset, calibration_end_year)
    stats = {
        column: _finite_summary(weather[column])
        for column in ("tmin", "tmax", "tavg", "prec", "rad", "cwb")
        if column in weather
    }
    grouped = weather.groupby([KEY_LOC, KEY_YEAR], observed=True)
    profile = {
        "calibration_end_year": int(calibration_end_year),
        "n_locations": int(labels[KEY_LOC].nunique()),
        "n_location_years": int(len(labels)),
        "year_range": [int(labels[KEY_YEAR].min()), int(labels[KEY_YEAR].max())],
        "weather": stats,
        "yield": _finite_summary(labels[KEY_TARGET]),
        "crop_season_event_days": {
            "tmin_below_0": float(grouped["tmin"].apply(lambda x: (x < 0).sum()).mean()),
            "tmax_above_35": float(grouped["tmax"].apply(lambda x: (x > 35).sum()).mean()),
            "prec_below_1": float(grouped["prec"].apply(lambda x: (x < 1).sum()).mean()),
        },
    }
    return profile


def _candidate_thresholds(weather: pd.DataFrame, variable: str) -> list[float]:
    values = pd.to_numeric(weather[variable], errors="coerce").dropna()
    if variable == "tmin":
        quantiles, anchors, bounds = (0.05, 0.10, 0.20, 0.30), (-5, 0, 5, 10, 15), (-15, 25)
    elif variable == "tmax":
        quantiles, anchors, bounds = (0.70, 0.80, 0.90, 0.95), (25, 30, 35, 40), (15, 55)
    else:
        quantiles, anchors, bounds = (0.10, 0.25, 0.50, 0.75), (0.5, 1, 2, 5), (0, 20)
    raw = [*anchors, *(values.quantile(q) for q in quantiles)]
    return sorted({round(float(value), 2) for value in raw if bounds[0] <= value <= bounds[1]})


def calibration_stress_evidence(dataset, calibration_end_year: int) -> dict:
    """Correlate candidate stress-day counts with yield inside calibration only."""
    weather, labels = calibration_weather_and_labels(dataset, calibration_end_year)
    result: dict[str, dict[float, dict]] = {}
    for variable in ("tmin", "tmax", "prec"):
        variable_result = {}
        for threshold in _candidate_thresholds(weather, variable):
            if variable == "tmax":
                event = pd.to_numeric(weather[variable], errors="coerce") > threshold
            else:
                event = pd.to_numeric(weather[variable], errors="coerce") < threshold
            counts = (
                weather[[KEY_LOC, KEY_YEAR]]
                .assign(event=event.astype(float))
                .groupby([KEY_LOC, KEY_YEAR], as_index=False, observed=True)["event"]
                .sum()
            )
            joined = counts.merge(labels, on=[KEY_LOC, KEY_YEAR], how="inner")
            if len(joined) >= 3 and joined["event"].std() > 0 and joined[KEY_TARGET].std() > 0:
                corr = float(joined["event"].corr(joined[KEY_TARGET]))
            else:
                corr = 0.0
            variable_result[threshold] = {
                "r": corr if np.isfinite(corr) else 0.0,
                "mean_days": float(joined["event"].mean()) if not joined.empty else 0.0,
                "n_location_years": int(len(joined)),
            }
        result[variable] = variable_result
    return result


def calibration_inner_years(dataset, calibration_end_year: int, fraction: float = 0.3) -> list[int]:
    years = sorted({int(year) for year in dataset.years if int(year) <= calibration_end_year})
    if len(years) < 2:
        return []
    return chronological_test_years(years, fraction)


def evaluate_candidate_configs(
    dataset,
    candidates: list[dict],
    calibration_end_year: int,
    regressors: list[str],
    estimator_factory: Callable[[str], object],
) -> tuple[dict, pd.DataFrame]:
    """Select one LLM candidate by mean inner-fold NRMSE across regressors."""
    inner_years = calibration_inner_years(dataset, calibration_end_year)
    rows = []
    frames: dict[str, tuple[pd.DataFrame, list[str], dict]] = {}
    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{index + 1}")
        config = copy.deepcopy(candidate["config"])
        frame = merge_features_labels(build_feature_table(dataset, config), dataset)
        frame = frame[frame[KEY_YEAR] <= calibration_end_year].copy()
        columns = feature_columns(frame)
        frames[candidate_id] = (frame, columns, candidate)
        if not inner_years:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "regressor": "not_evaluable",
                    "inner_test_years": "",
                    "n_features": len(columns),
                    "normalized_rmse": np.nan,
                    "r2": np.nan,
                }
            )
            continue
        for regressor in regressors:
            try:
                metrics, _ = evaluate_temporal_folds(
                    frame,
                    columns,
                    lambda name=regressor: estimator_factory(name),
                    inner_years,
                )
            except ValueError:
                metrics = {"normalized_rmse": np.inf, "r2": -np.inf}
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "regressor": regressor,
                    "inner_test_years": ";".join(map(str, inner_years)),
                    "n_features": len(columns),
                    "normalized_rmse": float(metrics["normalized_rmse"]),
                    "r2": float(metrics["r2"]),
                }
            )
    metrics_frame = pd.DataFrame(rows)
    scores = (
        metrics_frame.replace([np.inf, -np.inf], np.nan)
        .groupby("candidate_id")["normalized_rmse"]
        .mean()
    )
    selected_id = scores.idxmin() if scores.notna().any() else next(iter(frames))
    metrics_frame["mean_candidate_normalized_rmse"] = metrics_frame["candidate_id"].map(scores)
    metrics_frame["selected"] = metrics_frame["candidate_id"].eq(selected_id)
    selected = copy.deepcopy(frames[selected_id][2])
    selected["selection_score_mean_normalized_rmse"] = (
        float(scores[selected_id]) if pd.notna(scores[selected_id]) else None
    )
    selected["inner_test_years"] = inner_years
    return selected, metrics_frame


def baseline_candidate() -> dict:
    return {
        "candidate_id": "baseline",
        "agronomic_rationale": "Hard-coded CY-Bench reference configuration.",
        "config": copy.deepcopy(DEFAULT_FEATURE_CONFIG),
    }
