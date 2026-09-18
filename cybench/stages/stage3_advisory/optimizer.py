"""Historical-analog stress diagnosis and bounded recovery scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.stages.stage3_advisory.models import YieldResponseModel
from cybench.stages.stage3_advisory.schema import CountryYieldData, feature_role


@dataclass(frozen=True)
class OptimizationPolicy:
    reference_yield_quantile: float = 0.75
    stress_score_threshold: float = 0.75
    max_water_proxy_shift_fraction: float = 0.5
    max_actionable_features: int = 5
    top_reported_stresses: int = 8


def policy_dict(policy: OptimizationPolicy) -> dict:
    return asdict(policy)


def sanitize_policy(proposal: dict | None) -> OptimizationPolicy:
    """Clamp an LLM proposal to conservative, data-only scenario bounds."""
    proposal = proposal if isinstance(proposal, dict) else {}

    def bounded_float(key: str, default: float, lower: float, upper: float) -> float:
        try:
            value = float(proposal.get(key, default))
        except (TypeError, ValueError):
            value = default
        return float(np.clip(value, lower, upper))

    def bounded_int(key: str, default: int, lower: int, upper: int) -> int:
        try:
            value = int(proposal.get(key, default))
        except (TypeError, ValueError):
            value = default
        return int(np.clip(value, lower, upper))

    return OptimizationPolicy(
        reference_yield_quantile=bounded_float(
            "reference_yield_quantile", 0.75, 0.65, 0.9
        ),
        stress_score_threshold=bounded_float(
            "stress_score_threshold", 0.75, 0.5, 2.0
        ),
        max_water_proxy_shift_fraction=bounded_float(
            "max_water_proxy_shift_fraction", 0.5, 0.1, 0.5
        ),
        max_actionable_features=bounded_int("max_actionable_features", 5, 1, 8),
        top_reported_stresses=bounded_int("top_reported_stresses", 8, 3, 12),
    )


def select_current_conditions(
    data: CountryYieldData, target_year: int | None = None
) -> pd.DataFrame:
    """Use the requested year, or each administrative unit's latest observation."""
    frame = data.frame
    if target_year is not None:
        selected = frame[frame[KEY_YEAR] == target_year].copy()
        if selected.empty:
            raise ValueError(f"No observations found for target year {target_year}")
        return selected.reset_index(drop=True)
    indices = frame.groupby(KEY_LOC)[KEY_YEAR].idxmax()
    return frame.loc[indices].sort_values(KEY_LOC).reset_index(drop=True)


def _historical_reference(
    data: CountryYieldData, current: pd.Series, policy: OptimizationPolicy
) -> pd.DataFrame:
    history = data.frame[data.frame[KEY_YEAR] < current[KEY_YEAR]]
    local = history[history[KEY_LOC] == current[KEY_LOC]]
    pool = local if len(local) >= 5 else history
    if pool.empty:
        pool = data.frame[data.frame[KEY_YEAR] != current[KEY_YEAR]]
    cutoff = pool[KEY_TARGET].quantile(policy.reference_yield_quantile)
    reference = pool[pool[KEY_TARGET] >= cutoff]
    return reference if not reference.empty else pool


def _predict_row(model: YieldResponseModel, row: pd.Series) -> float:
    return float(model.predict(pd.DataFrame([row]))["predicted_yield"].iloc[0])


def diagnose_stress(
    data: CountryYieldData,
    current: pd.DataFrame,
    model: YieldResponseModel,
    policy: OptimizationPolicy,
) -> pd.DataFrame:
    """Compare current conditions with pre-year high-yield historical analogs."""
    records = []
    for row_index, row in current.iterrows():
        reference = _historical_reference(data, row, policy)
        reference_stats = {}
        candidate_rows = [row.copy()]
        for feature in data.feature_columns:
            median = float(reference[feature].median())
            q25, q75 = reference[feature].quantile([0.25, 0.75])
            scale = max(float(q75 - q25), float(reference[feature].std()), 1e-9)
            candidate = row.copy()
            candidate[feature] = median
            candidate_rows.append(candidate)
            reference_stats[feature] = (median, scale)
        batch_prediction = model.predict(pd.DataFrame(candidate_rows))[
            "predicted_yield"
        ]
        baseline_prediction = float(batch_prediction.iloc[0])
        for position, feature in enumerate(data.feature_columns, start=1):
            median, scale = reference_stats[feature]
            standardized_gap = float(abs(row[feature] - median) / scale)
            analog_prediction = float(batch_prediction.iloc[position])
            potential = analog_prediction - baseline_prediction
            records.append(
                {
                    "row_index": row_index,
                    KEY_LOC: row[KEY_LOC],
                    KEY_YEAR: int(row[KEY_YEAR]),
                    "feature": feature,
                    "feature_role": feature_role(feature),
                    "current_value": float(row[feature]),
                    "high_yield_reference": median,
                    "standardized_gap": standardized_gap,
                    "model_importance": float(model.importance.get(feature, 0.0)),
                    "single_feature_yield_potential": potential,
                    "stress_score": standardized_gap
                    * float(model.importance.get(feature, 0.0))
                    * max(potential, 0.0),
                }
            )
    result = pd.DataFrame(records)
    return result.sort_values(
        [KEY_LOC, "stress_score"], ascending=[True, False]
    ).reset_index(drop=True)


def optimize_recovery_scenario(
    data: CountryYieldData,
    current: pd.DataFrame,
    stress: pd.DataFrame,
    model: YieldResponseModel,
    policy: OptimizationPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily move observed water proxies toward feasible analog values.

    These are environmental target scenarios, not estimates of a specific irrigation
    dose. Temperature, radiation, soil, NDVI and FPAR are never modified.
    """
    selected, changes = [], []
    for row_index, original in current.iterrows():
        candidate = original.copy()
        baseline_prediction = _predict_row(model, original)
        best_prediction = baseline_prediction
        candidates = stress[
            (stress["row_index"] == row_index)
            & (stress["feature_role"] == "water_proxy")
            & (stress["standardized_gap"] >= policy.stress_score_threshold)
            & (stress["single_feature_yield_potential"] > 0)
        ].head(policy.max_actionable_features)
        history = data.frame[data.frame[KEY_YEAR] < original[KEY_YEAR]]
        for _, signal in candidates.iterrows():
            feature = signal["feature"]
            trial = candidate.copy()
            shift = policy.max_water_proxy_shift_fraction * (
                signal["high_yield_reference"] - trial[feature]
            )
            lower, upper = history[feature].quantile([0.1, 0.9])
            trial[feature] = float(np.clip(trial[feature] + shift, lower, upper))
            trial_prediction = _predict_row(model, trial)
            if trial_prediction > best_prediction:
                changes.append(
                    {
                        KEY_LOC: original[KEY_LOC],
                        KEY_YEAR: int(original[KEY_YEAR]),
                        "feature": feature,
                        "feature_role": "water_proxy",
                        "from_value": float(candidate[feature]),
                        "to_value": float(trial[feature]),
                        "predicted_increment": trial_prediction - best_prediction,
                    }
                )
                candidate = trial
                best_prediction = trial_prediction
        candidate["baseline_predicted_yield"] = baseline_prediction
        candidate["predicted_yield"] = best_prediction
        candidate["scenario_yield_gain"] = best_prediction - baseline_prediction
        selected.append(candidate)
    return pd.DataFrame(selected).reset_index(drop=True), pd.DataFrame(changes)


def scenario_summary(
    current: pd.DataFrame,
    recovery: pd.DataFrame,
    stress: pd.DataFrame,
    model: YieldResponseModel,
) -> pd.DataFrame:
    baseline = model.predict(current)
    return pd.DataFrame(
        [
            {
                "scenario": "S1_observed_conditions",
                "n_location_years": len(current),
                "mean_predicted_yield": baseline["predicted_yield"].mean(),
                "mean_scenario_gain": 0.0,
                "n_stress_signals": int((stress["stress_score"] > 0).sum()),
            },
            {
                "scenario": "S2_bounded_water_recovery",
                "n_location_years": len(recovery),
                "mean_predicted_yield": recovery["predicted_yield"].mean(),
                "mean_scenario_gain": recovery["scenario_yield_gain"].mean(),
                "n_stress_signals": int((stress["stress_score"] > 0).sum()),
            },
        ]
    )
