import copy

from cybench.llm.agronomic_advisor import AgronomicAdvisor
from cybench.util.features import DEFAULT_FEATURE_CONFIG


def _raw_config(tmin, veg="max", soil="mean"):
    return {
        "gdd_base_temp": {"crop": "maize", "recommended": 10},
        "gdd_upper_limit": {"crop": "maize", "recommended": 35},
        "stress_thresholds": {
            "tmin": {"operator": "<", "threshold": tmin},
            "tmax": {"operator": ">", "threshold": 35},
            "prec": {"operator": "<", "threshold": 1},
        },
        "veg_agg_method": {"recommended": veg},
        "soil_moisture_agg_method": {"recommended": soil},
        "include_fpar": True,
        "include_ndvi": True,
        "include_soil_moisture": True,
    }


def _evidence():
    return {
        "tmin": {0.0: {"r": -0.2, "mean_days": 4.0}},
        "tmax": {35.0: {"r": -0.3, "mean_days": 9.0}},
        "prec": {1.0: {"r": -0.4, "mean_days": 60.0}},
    }


def test_regional_prompt_produces_three_distinct_candidates():
    captured = {}

    def generate(prompt):
        captured["prompt"] = prompt
        return {
            "regional_assessment": "highland and seasonal rainfall contrast",
            "candidates": [
                {"candidate_id": "regional_domain", "config": _raw_config(5)},
                {
                    "candidate_id": "empirical_stress",
                    "config": _raw_config(0, veg="mean"),
                },
                {
                    "candidate_id": "robust_alternative",
                    "config": _raw_config(2, soil="median"),
                },
            ],
        }

    advisor = AgronomicAdvisor(generate_fn=generate)
    report = advisor.recommend_candidates(
        "maize",
        "MX",
        climate_profile={"weather": {"tmin": {"q05": 8.0}}},
        stress_corrs=_evidence(),
        baseline_config=copy.deepcopy(DEFAULT_FEATURE_CONFIG),
        default_gdd_base=10,
        default_gdd_upper=35,
        use_cache=False,
    )
    assert report["used_fallback"] is False
    assert len(report["candidates"]) == 3
    assert all(item["candidate_source"] == "llm" for item in report["candidates"])
    assert "Subtropical/tropical region with highland frost contrasts" in captured["prompt"]
    assert "external test years are excluded" in captured["prompt"]
    assert '"q05": 8.0' in captured["prompt"]


def test_valid_llm_candidates_survive_partial_fallback_completion():
    duplicate = _raw_config(0)
    advisor = AgronomicAdvisor(
        generate_fn=lambda _: {
            "candidates": [
                {"candidate_id": "one", "config": duplicate},
                {"candidate_id": "two", "config": duplicate},
                {"candidate_id": "invalid", "config": _raw_config(99)},
            ]
        }
    )
    report = advisor.recommend_candidates(
        "maize",
        "PT",
        climate_profile={"weather": {}},
        stress_corrs=_evidence(),
        baseline_config=copy.deepcopy(DEFAULT_FEATURE_CONFIG),
        default_gdd_base=10,
        default_gdd_upper=35,
        use_cache=False,
    )
    assert report["used_fallback"] is True
    assert len(report["candidates"]) == 3
    sources = [item["candidate_source"] for item in report["candidates"]]
    assert sources.count("llm") == 1
    assert sources.count("fallback") == 2
