"""Residualized, direction-aware calibration evidence for LLM prompts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.stages.stage1_feature_engineering.adaptive import (
    _candidate_thresholds,
    calibration_weather_and_labels,
)


def _two_way_residual(values: pd.Series, locations: pd.Series, years: pd.Series):
    frame = pd.DataFrame({"value": values, "location": locations, "year": years})
    return frame["value"] - frame.groupby("location")["value"].transform("mean") - frame.groupby("year")["value"].transform("mean") + frame["value"].mean()


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, seed: int, repeats: int = 200):
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return (0.0, 0.0)
    rng, values = np.random.default_rng(seed), []
    for _ in range(repeats):
        index = rng.integers(0, len(x), len(x))
        xb, yb = x[index], y[index]
        if np.std(xb) > 0 and np.std(yb) > 0:
            values.append(float(np.corrcoef(xb, yb)[0, 1]))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975])) if values else (0.0, 0.0)


def residualized_stress_evidence(dataset, calibration_end_year: int) -> dict:
    weather, labels = calibration_weather_and_labels(dataset, calibration_end_year)
    result = {}
    for variable_index, variable in enumerate(("tmin", "tmax", "prec")):
        variable_result = {}
        for threshold_index, threshold in enumerate(_candidate_thresholds(weather, variable)):
            values = pd.to_numeric(weather[variable], errors="coerce")
            event = values > threshold if variable == "tmax" else values < threshold
            counts = weather[[KEY_LOC, KEY_YEAR]].assign(event=event.astype(float)).groupby([KEY_LOC, KEY_YEAR], as_index=False, observed=True)["event"].sum()
            joined = counts.merge(labels, on=[KEY_LOC, KEY_YEAR], how="inner")
            event_residual = _two_way_residual(joined["event"], joined[KEY_LOC], joined[KEY_YEAR])
            yield_residual = _two_way_residual(joined[KEY_TARGET], joined[KEY_LOC], joined[KEY_YEAR])
            valid = np.isfinite(event_residual) & np.isfinite(yield_residual)
            x, y = event_residual[valid].to_numpy(dtype=float), yield_residual[valid].to_numpy(dtype=float)
            residual_r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 4 and np.std(x) > 0 and np.std(y) > 0 else 0.0
            ci_low, ci_high = _bootstrap_ci(x, y, seed=1103 + variable_index * 100 + threshold_index)
            mean_days = float(joined["event"].mean()) if not joined.empty else 0.0
            variable_result[threshold] = {
                "residual_r": residual_r,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_days": mean_days,
                "n_location_years": int(len(joined)),
                "usable": bool(len(joined) >= 20 and 3 <= mean_days <= 200 and residual_r < 0 and ci_high < 0),
            }
        result[variable] = variable_result
    return result


def prompt_evidence(evidence: dict) -> dict:
    return {variable: {threshold: values for threshold, values in candidates.items() if values.get("usable", False)} for variable, candidates in evidence.items()}
