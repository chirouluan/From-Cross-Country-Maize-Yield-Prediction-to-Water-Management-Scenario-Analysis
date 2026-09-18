"""Qwen roles for bounded scenario policy and evidence-grounded advice."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from cybench.config import KEY_LOC, KEY_YEAR
from cybench.llm.agent import DiagSTFNLLMAgent
from cybench.stages.stage3_advisory.optimizer import OptimizationPolicy, policy_dict
from cybench.stages.stage3_advisory.schema import CountryYieldData, feature_role

ALLOWED_ACTIONS = {
    "irrigation_assessment",
    "soil_water_conservation",
    "drainage_assessment",
    "sowing_window_review",
    "stress_tolerant_cultivar_review",
    "field_monitoring",
}


def frame_fingerprint(frame: pd.DataFrame) -> str:
    ordered = frame.sort_index(axis=1).sort_values(list(frame.columns[:1]))
    hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashes).hexdigest()[:16]


def _compact_rag_context(retrieval: dict | None) -> dict | None:
    if not retrieval:
        return None
    useful = {
        "mean_temp_c",
        "annual_precip_mm",
        "dry_days_lt_1mm",
        "max_consecutive_dry_days",
        "frost_days_tmin_lt_0c",
        "heat_days_tmax_gt_35c",
        "mean_vpd_kpa_approx",
        "mean_solar_radiation_mj_m2_day",
    }
    return {
        "retrieval_method": retrieval["retrieval_method"],
        "query": retrieval["query"],
        "information_boundary": retrieval["information_boundary"],
        "historical_summary": {
            key: value for key, value in retrieval["summary"].items() if key in useful
        },
        "representative_rows": [
            {key: value for key, value in row.items() if key in useful | {"year", "retrieval_role"}}
            for row in retrieval["retrieved_rows"]
        ],
    }


def policy_prompt(data: CountryYieldData, retrieval: dict | None = None) -> str:
    roles = {feature: feature_role(feature) for feature in data.feature_columns}
    counts = pd.Series(roles.values()).value_counts().to_dict()
    payload = {
        "country": data.country,
        "n_rows": len(data.frame),
        "year_range": [
            int(data.frame[KEY_YEAR].min()),
            int(data.frame[KEY_YEAR].max()),
        ],
        "feature_role_counts": counts,
        "default_policy": policy_dict(OptimizationPolicy()),
    }
    if retrieval:
        payload["structured_climate_rag"] = _compact_rag_context(retrieval)
    return f"""Review conservative search settings for a crop-yield stress analyzer.
The actual data contain weather, soil and remote-sensing predictors, but no fertilizer,
irrigation amount, cultivar or field-management treatments. You must not invent them.
Only water_proxy variables may be shifted in counterfactual target scenarios. Return
ONLY JSON with reference_yield_quantile (0.65-0.90), stress_score_threshold
(0.5-2.0), max_water_proxy_shift_fraction (0.1-0.5), max_actionable_features
(1-8), top_reported_stresses (3-12), and rationale.

Dataset summary:
{json.dumps(payload, indent=2, ensure_ascii=False)}"""


def advice_prompt(
    data: CountryYieldData,
    summary: pd.DataFrame,
    stress: pd.DataFrame,
    recovery: pd.DataFrame,
    changes: pd.DataFrame,
    validation: dict,
    retrieval: dict | None = None,
) -> str:
    evidence_columns = [
        KEY_LOC,
        KEY_YEAR,
        "feature",
        "feature_role",
        "current_value",
        "high_yield_reference",
        "standardized_gap",
        "single_feature_yield_potential",
    ]
    # Keep the complete diagnostics as stage artifacts, but give the 9B model a
    # compact, decision-focused evidence packet. Expanding eight rows for every
    # administrative unit can exceed the context/KV-cache budget of a 16 GB GPU.
    if stress.empty:
        selected_locations = []
        top = stress
    else:
        location_priority = (
            stress.groupby(KEY_LOC)["stress_score"].sum().sort_values(ascending=False)
        )
        selected_locations = location_priority.head(6).index.tolist()
        top = (
            stress[stress[KEY_LOC].isin(selected_locations)]
            .groupby(KEY_LOC, group_keys=False)
            .head(4)
        )
    selected_changes = changes[changes[KEY_LOC].isin(selected_locations)]
    selected_recovery = recovery[recovery[KEY_LOC].isin(selected_locations)]
    payload = {
        "country": data.country,
        "evidence_selection": {
            "rule": "top_6_locations_by_total_stress_score_top_4_features_each",
            "selected_locations": selected_locations,
        },
        "scenario_summary": summary.round(4).to_dict("records"),
        "stress_evidence": top[evidence_columns].round(4).to_dict("records"),
        "accepted_water_proxy_changes": selected_changes.round(4).to_dict("records"),
        "recovery_estimates": selected_recovery[
            [KEY_LOC, KEY_YEAR, "scenario_yield_gain"]
        ]
        .round(4)
        .to_dict("records"),
        "forward_validation": validation,
        "allowed_actions": sorted(ALLOWED_ACTIONS),
    }
    if retrieval:
        payload["structured_climate_rag"] = _compact_rag_context(retrieval)
    return f"""Convert supplied crop-stress evidence into conditional adaptation advice.
Use exact adm_id, year and feature names from the evidence. Select actions only from
allowed_actions. A water proxy is not an irrigation dose; temperature/radiation and
NDVI/FPAR cannot be directly changed. Never claim causality or invent a yield effect.
Mention that local crop stage, infrastructure, costs and agronomist review are needed.

Return ONLY JSON:
{{
  "recommendations": [{{
    "adm_id": "exact id",
    "year": "exact year",
    "evidence_features": ["exact supplied features"],
    "actions": ["allowed actions only"],
    "reasoning": "conditional explanation",
    "limitations": "data and implementation boundary"
  }}],
  "overall_limitations": ["observational scenario, not a field prescription"]
}}

Evidence:
{json.dumps(payload, indent=2, ensure_ascii=False, default=str)}"""


def validate_advice(advice: dict | None, stress: pd.DataFrame) -> dict:
    recommendations = (
        advice.get("recommendations") if isinstance(advice, dict) else None
    )
    if not isinstance(recommendations, list):
        return {
            "schema_valid": 0,
            "grounded_recommendation_ratio": 0.0,
            "n_recommendations": 0,
            "n_grounded_recommendations": 0,
        }
    valid = 0
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        required = {"adm_id", "year", "evidence_features", "actions", "limitations"}
        if not required <= set(item):
            continue
        try:
            item_year = int(item["year"])
        except (TypeError, ValueError):
            # LLMs sometimes copy schema placeholders such as "exact year".
            # Treat them as ungrounded output instead of aborting the experiment.
            continue
        if not isinstance(item["evidence_features"], list) or not isinstance(
            item["actions"], list
        ):
            continue
        matches = stress[
            (stress[KEY_LOC].astype(str) == str(item["adm_id"]))
            & (stress[KEY_YEAR].astype(int) == item_year)
        ]
        features = set(map(str, item["evidence_features"]))
        actions = set(map(str, item["actions"]))
        if (
            not matches.empty
            and features
            and features <= set(matches["feature"].astype(str))
            and actions
            and actions <= ALLOWED_ACTIONS
        ):
            valid += 1
    count = len(recommendations)
    ratio = valid / count if count else 0.0
    return {
        "schema_valid": int(count > 0 and valid == count),
        "grounded_recommendation_ratio": round(ratio, 4),
        "n_recommendations": count,
        "n_grounded_recommendations": valid,
    }


def llm_agent(variant, advisor_name: str | None = None) -> DiagSTFNLLMAgent:
    return DiagSTFNLLMAgent(
        model_dir=str(variant.model_dir),
        model_id=f"stage3_{advisor_name or variant.name}",
        deterministic=True,
    )
