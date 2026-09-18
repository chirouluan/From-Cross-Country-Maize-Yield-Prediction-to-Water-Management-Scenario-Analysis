"""Candidate expansion and calibration-only parameter selection."""

from __future__ import annotations

import copy
import json

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

from cybench.stages.regressors import make_regressor


def _bounded(value, low, high):
    return float(min(high, max(low, value)))


def expand_candidates(report: dict, crop: str, budget: int = 9) -> list[dict]:
    """Expand each point recommendation into deterministic low/mid/high variants."""
    candidates, signatures = [], set()
    for item in report.get("candidates", []):
        for shift_name, shift in (("low", -1), ("mid", 0), ("high", 1)):
            config = copy.deepcopy(item["config"])
            base = float(config["gdd_base_temp"][crop])
            upper = config["gdd_upper_limit"].get(crop)
            config["gdd_base_temp"][crop] = _bounded(base + shift, -5, 20)
            if upper is not None:
                config["gdd_upper_limit"][crop] = _bounded(float(upper) + 2 * shift, base + 1, 50)
            stress = config["stress_thresholds"]
            stress["tmin"]["threshold"] = _bounded(float(stress["tmin"]["threshold"]) + 2 * shift, -15, 25)
            stress["tmax"]["threshold"] = _bounded(float(stress["tmax"]["threshold"]) + 2 * shift, 15, 55)
            stress["prec"]["threshold"] = _bounded(float(stress["prec"]["threshold"]) + 0.5 * shift, 0, 20)
            signature = json.dumps(config, sort_keys=True)
            if signature in signatures:
                continue
            signatures.add(signature)
            candidates.append(
                {
                    "candidate_id": f"{item['candidate_id']}_{shift_name}",
                    "candidate_source": item.get("candidate_source", "advisor_range"),
                    "agronomic_rationale": item.get("agronomic_rationale", ""),
                    "config": config,
                }
            )
            if len(candidates) >= budget:
                return candidates
    return candidates


def reduced_regressor(name: str, k: int, seed: int = 42):
    """Fit univariate feature selection inside each temporal training fold."""
    return make_pipeline(
        SelectKBest(score_func=f_regression, k=k), make_regressor(name, seed=seed)
    )


def extended_inner_years(dataset, calibration_end_year: int, min_train_years: int = 5, max_validation_years: int = 8):
    years = sorted({int(year) for year in dataset.years if int(year) <= calibration_end_year})
    if len(years) <= min_train_years:
        return years[-1:] if len(years) >= 2 else []
    return years[min_train_years:][-max_validation_years:]
