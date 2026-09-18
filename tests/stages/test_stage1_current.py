import numpy as np

from cybench.llm.agent import DiagSTFNLLMAgent
from cybench.stages.regressors import make_regressor
from cybench.stages.stage1_feature_engineering.adaptive import baseline_candidate
from cybench.stages.stage1_feature_engineering.advisors import (
    _all_decimal,
    _valid_continuous_parameters,
)
from cybench.stages.stage1_feature_engineering.selection import expand_candidates
from cybench.stages.stage1_feature_engineering.rag_retrieval import (
    RANK_COLUMNS,
    retrieve_climate_metrics,
)


def test_candidate_expansion_produces_distinct_variants():
    source = baseline_candidate()
    source["config"]["gdd_base_temp"]["maize"] = 9.37
    source["config"]["gdd_upper_limit"]["maize"] = 34.62
    source["config"]["stress_thresholds"]["tmin"]["threshold"] = 2.43
    source["config"]["stress_thresholds"]["tmax"]["threshold"] = 36.71
    source["config"]["stress_thresholds"]["prec"]["threshold"] = 1.28
    candidates = expand_candidates({"candidates": [source]}, "maize")
    assert len(candidates) == 3
    assert len({candidate["candidate_id"] for candidate in candidates}) == 3


def test_svr_is_registered_and_predicts():
    x = np.arange(30, dtype=float).reshape(10, 3)
    y = np.linspace(1, 2, 10)
    model = make_regressor("svr").fit(x, y)
    assert model.predict(x).shape == (10,)


def test_integer_valued_llm_parameters_are_valid():
    candidate = baseline_candidate()
    candidate["config"]["gdd_base_temp"]["maize"] = 5.0
    candidate["config"]["gdd_upper_limit"]["maize"] = 30.0
    candidate["config"]["stress_thresholds"]["tmin"]["threshold"] = 2.5
    candidate["config"]["stress_thresholds"]["tmax"]["threshold"] = 35.0
    candidate["config"]["stress_thresholds"]["prec"]["threshold"] = 1.0
    assert _valid_continuous_parameters(candidate, "maize")


def test_malformed_outer_json_recovers_only_complete_candidates():
    text = (
        '{"candidates": [{"candidate_id":"one","config":{"x":1}}}, '
        '{"candidate_id":"two","config":{"x":2}}]}'
    )
    result = DiagSTFNLLMAgent()._extract_json(text)
    assert result["salvaged_from_malformed_outer_json"] is True
    assert [item["candidate_id"] for item in result["candidates"]] == ["one", "two"]


def test_structured_rag_filters_country_crop_and_future_years(tmp_path):
    header = [
        "parent_id", "country_code", "crop", "year", "year",
        "coverage_min", "coverage_min", *RANK_COLUMNS,
    ]
    rows = []
    for country, year in (("CN", 2001), ("CN", 2002), ("CN", 2003), ("CN", 2023), ("PT", 2001)):
        climate = [
            15 + year % 3, 500 + year % 5, 180 + year % 7,
            20 + year % 4, year % 6, year % 2, 1.0 + year % 3 / 10, 18.0,
        ]
        rows.append([
            f"{country}_maize_{year}", country, "maize", year, year,
            1.0, 1.0, *climate,
        ])
    path = tmp_path / "metrics.csv"
    path.write_text(
        ",".join(header) + "\n" + "\n".join(",".join(map(str, row)) for row in rows),
        encoding="utf-8",
    )

    result = retrieve_climate_metrics(
        path, country="CN", crop="maize", calibration_end_year=2003, top_k=3
    )

    assert result["eligible_row_count"] == 3
    assert result["eligible_year_range"] == [2001, 2003]
    assert {row["country_code"] for row in result["retrieved_rows"]} == {"CN"}
    assert max(row["year"] for row in result["retrieved_rows"]) <= 2003
    assert "no embeddings" in result["retrieval_method"]
    assert result["source"]["schema_duplicate_audit"]["year.1"]["identical"]
