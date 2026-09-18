import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.stages.artifacts import (
    read_dataframe_artifact,
    write_dataframe_artifact,
)
from cybench.stages.config import qwen_base_variant, tabpfn_checkpoints
from cybench.stages.data import chronological_test_years
from cybench.stages.evaluation import (
    evaluate_temporal_folds,
    repeat_seed,
    summarize_repeated_metrics,
)
from cybench.stages.stage2_tabpfn_transfer.run import (
    CN_STRICT_TEST_YEARS,
    DEFAULT_COUNTRIES,
    run_stage2,
    validate_checkpoints,
)
from cybench.stages.stage3_advisory.llm import validate_advice
from cybench.stages.stage3_advisory.models import fit_yield_response_model
from cybench.stages.stage3_advisory.optimizer import (
    OptimizationPolicy,
    diagnose_stress,
    optimize_recovery_scenario,
    sanitize_policy,
    select_current_conditions,
)
from cybench.stages.stage3_advisory.run import run_stage3_country
from cybench.stages.stage3_advisory.schema import (
    attach_stage2_predictions,
    feature_role,
    load_stage1_yield_data,
    validate_yield_frame,
)


def test_chronological_test_years_selects_latest_years():
    assert chronological_test_years(range(2010, 2020), 0.3) == [2017, 2018, 2019]


def test_stage2_includes_cn_with_strict_heldout_years():
    assert "CN" in DEFAULT_COUNTRIES
    assert CN_STRICT_TEST_YEARS == (2020, 2021, 2022)


def test_temporal_evaluation_produces_forward_predictions():
    frame = pd.DataFrame(
        {
            KEY_LOC: ["AA_1"] * 4 + ["AA_2"] * 4,
            KEY_YEAR: [2018, 2019, 2020, 2021] * 2,
            "feature": np.arange(8, dtype=float),
            KEY_TARGET: np.arange(8, dtype=float) + 1,
        }
    )
    metrics, predictions = evaluate_temporal_folds(
        frame, ["feature"], lambda: DummyRegressor(strategy="mean"), [2020, 2021]
    )
    assert set(predictions[KEY_YEAR]) == {2020, 2021}
    assert {"r2", "normalized_rmse"} <= set(metrics)


def test_five_repeat_summary_uses_sample_variance():
    rows = []
    for repeat, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0), start=1):
        rows.append(
            {
                "model": "example",
                "repeat": repeat,
                "mape": value,
                "normalized_rmse": value,
                "r": value,
                "r2": value,
                "kge": value,
            }
        )
    summary = summarize_repeated_metrics(pd.DataFrame(rows), ["model"])
    assert summary.loc[0, "r2_mean"] == pytest.approx(3.0)
    assert summary.loc[0, "r2_variance"] == pytest.approx(2.5)
    assert [repeat_seed(42, i) for i in range(1, 6)] == [42, 1051, 2060, 3069, 4078]


def test_identical_tabpfn_checkpoints_are_rejected(tmp_path):
    base = tmp_path / "base.ckpt"
    tuned = tmp_path / "tuned.ckpt"
    base.write_bytes(b"same")
    tuned.write_bytes(b"same")
    with pytest.raises(ValueError, match="identical"):
        validate_checkpoints(base, tuned)


def test_model_layout_supports_base_qwen_and_tabpfn_pair(tmp_path):
    base_variant = qwen_base_variant("base_model")
    assert base_variant.name == "qwen_base"
    assert base_variant.model_dir.name == "base_model"

    base = tmp_path / "original.ckpt"
    tuned = tmp_path / "agronomy.ckpt"
    base.write_bytes(b"base")
    tuned.write_bytes(b"tuned")
    assert tabpfn_checkpoints(base, tuned) == (base.resolve(), tuned.resolve())


def _yield_frame():
    rows = []
    for site_number in range(3):
        for offset, year in enumerate(range(2015, 2023)):
            precipitation = 55 + offset * 5
            soil_moisture = 0.12 + offset * 0.01
            temperature = 29 - offset * 0.2
            rows.append(
                {
                    "adm_id": f"AA_{site_number}",
                    "year": year,
                    "awc": 0.2 + site_number * 0.01,
                    "bulk_density": 1.3,
                    "mean_prec1": precipitation,
                    "mean_cum_cwb1": precipitation - 45,
                    "mean_ssm1": soil_moisture,
                    "mean_tmax1": temperature,
                    "max_cum_ndvi1": 0.3 + offset * 0.03,
                    "yield": 2.0
                    + precipitation * 0.04
                    + soil_moisture * 5
                    - max(temperature - 28, 0) * 0.2
                    + site_number * 0.05,
                }
            )
    return pd.DataFrame(rows)


def _write_stage1_artifact(root, frame=None):
    frame = _yield_frame() if frame is None else frame
    path = root / "maize_AA" / "dataset_hardcoded.csv"
    write_dataframe_artifact(
        frame,
        path,
        stage="stage1_feature_engineering",
        artifact="feature_dataset",
        metadata={"crop": "maize", "country": "AA", "strategy": "hardcoded"},
    )
    return path


def _write_rag_stage1_artifact(root, frame=None):
    frame = _yield_frame() if frame is None else frame
    path = root / "results" / "maize_AA" / "qwen_rag" / "feature_dataset.csv"
    write_dataframe_artifact(
        frame,
        path,
        stage="stage1_feature_engineering",
        artifact="selected_feature_dataset",
        metadata={"country": "AA", "advisor": "qwen_rag"},
    )
    return path


def _write_stage2_predictions(root, frame, source_stage1_sha256):
    latest = frame[frame["year"] == frame["year"].max()]
    for model in ("tabpfn_base", "tabpfn_cn_finetuned"):
        prediction = latest[["adm_id", "year", "yield"]].copy()
        prediction["prediction"] = prediction["yield"] * 0.98
        prediction["country"] = "AA"
        prediction["model"] = model
        write_dataframe_artifact(
            prediction,
            root / "maize_AA" / f"predictions_{model}.csv",
            stage="stage2_tabpfn_transfer",
            artifact="tabpfn_predictions",
            metadata={
                "crop": "maize",
                "country": "AA",
                "model": model,
                "source_stage1_sha256": source_stage1_sha256,
            },
        )


def test_artifact_checksum_detects_manual_changes(tmp_path):
    empty_path = tmp_path / "empty.csv"
    write_dataframe_artifact(
        pd.DataFrame(),
        empty_path,
        stage="stage3_advisory",
        artifact="accepted_proxy_changes",
    )
    empty, _ = read_dataframe_artifact(empty_path)
    assert empty.empty

    path = tmp_path / "artifact.csv"
    write_dataframe_artifact(
        _yield_frame(),
        path,
        stage="stage1_feature_engineering",
        artifact="feature_dataset",
    )
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_dataframe_artifact(path)


def test_stage2_consumes_saved_stage1_dataset(tmp_path, monkeypatch):
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    _write_rag_stage1_artifact(stage1_dir)
    base = tmp_path / "base.ckpt"
    tuned = tmp_path / "tuned.ckpt"
    base.write_bytes(b"base")
    tuned.write_bytes(b"tuned")
    monkeypatch.setattr(
        "cybench.stages.stage2_tabpfn_transfer.run.make_tabpfn",
        lambda checkpoint, device, seed=42: DummyRegressor(strategy="mean"),
    )
    result = run_stage2(
        "maize",
        ["AA"],
        base,
        tuned,
        "cpu",
        0.3,
        stage2_dir,
        stage1_dir,
    )
    assert set(result["model"]) == {"tabpfn_base", "tabpfn_cn_finetuned"}
    read_dataframe_artifact(
        stage2_dir / "maize_AA" / "predictions_tabpfn_base.csv",
        expected_stage="stage2_tabpfn_transfer",
    )


def test_stage3_rejects_predictions_from_another_stage1_run(tmp_path):
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    frame = _yield_frame()
    _write_stage1_artifact(stage1_dir, frame)
    _write_stage2_predictions(stage2_dir, frame, "wrong-stage1-sha")
    data = load_stage1_yield_data(stage1_dir, "maize", "AA")
    with pytest.raises(ValueError, match="different Stage 1 dataset"):
        attach_stage2_predictions(data, stage2_dir, "maize")


def test_stage3_aggregates_five_stage2_repeats_with_uncertainty(tmp_path):
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    frame = _yield_frame()
    stage1_path = _write_rag_stage1_artifact(stage1_dir, frame)
    _, stage1_manifest = read_dataframe_artifact(stage1_path)
    latest = frame[frame["year"] == frame["year"].max()]
    for model in ("tabpfn_base", "tabpfn_cn_finetuned"):
        parts = []
        for repeat, prediction_value in enumerate((1, 2, 3, 4, 5), start=1):
            part = latest[["adm_id", "year", "yield"]].copy()
            part["prediction"] = float(prediction_value)
            part["split"] = "test"
            part["repeat"] = repeat
            part["seed"] = 42 + (repeat - 1) * 1009
            parts.append(part)
        write_dataframe_artifact(
            pd.concat(parts, ignore_index=True),
            stage2_dir / "maize_AA" / f"predictions_{model}.csv",
            stage="stage2_tabpfn_transfer",
            artifact="tabpfn_predictions",
            metadata={
                "crop": "maize",
                "country": "AA",
                "model": model,
                "source_stage1_sha256": stage1_manifest["sha256"],
                "repeats": 5,
            },
        )
    data = load_stage1_yield_data(
        stage1_dir, "maize", "AA", strategy="qwen_rag"
    )
    attached = attach_stage2_predictions(data, stage2_dir, "maize")
    latest_attached = attached.frame[attached.frame["year"] == 2022]
    for model in ("tabpfn_base", "tabpfn_cn_finetuned"):
        prefix = f"stage2_{model}"
        assert np.allclose(latest_attached[f"{prefix}_prediction"], 3.0)
        assert np.allclose(latest_attached[f"{prefix}_prediction_variance"], 2.5)
        assert np.allclose(latest_attached[f"{prefix}_prediction_std"], np.sqrt(2.5))
        assert set(latest_attached[f"{prefix}_n_repeats"]) == {5}
        assert set(latest_attached[f"{prefix}_split"]) == {"test"}


def test_real_feature_schema_and_llm_policy_safety_bounds():
    data = validate_yield_frame(_yield_frame(), "AA")
    policy = sanitize_policy(
        {
            "reference_yield_quantile": 0.2,
            "max_water_proxy_shift_fraction": 0.9,
            "max_actionable_features": 99,
        }
    )
    assert policy.reference_yield_quantile == 0.65
    assert policy.max_water_proxy_shift_fraction == 0.5
    assert policy.max_actionable_features == 8
    assert "n_rate_kg_ha" not in data.feature_columns
    assert feature_role("mean_prec1") == "water_proxy"
    assert feature_role("mean_tmax1") == "temperature_hazard"


def test_stress_optimizer_changes_only_water_proxies():
    frame = _yield_frame()
    # Make the latest year a dry stress year instead of the historical upward trend.
    frame.loc[frame["year"] == 2022, ["mean_prec1", "mean_cum_cwb1", "mean_ssm1"]] = [
        30,
        -20,
        0.08,
    ]
    data = validate_yield_frame(frame, "AA")
    model = fit_yield_response_model(data)
    policy = OptimizationPolicy()
    current = select_current_conditions(data)
    stress = diagnose_stress(data, current, model, policy)
    recovery, changes = optimize_recovery_scenario(data, current, stress, model, policy)
    assert set(changes.get("feature_role", [])) <= {"water_proxy"}
    assert np.all(recovery["scenario_yield_gain"] >= 0)
    assert "n_rate_kg_ha" not in recovery


def test_llm_advice_must_reference_real_stress_evidence():
    stress = pd.DataFrame(
        {
            "adm_id": ["AA_1"],
            "year": [2022],
            "feature": ["mean_prec1"],
        }
    )
    recommendation = {
        "adm_id": "AA_1",
        "year": 2022,
        "evidence_features": ["mean_prec1"],
        "actions": ["irrigation_assessment"],
        "reasoning": "conditional",
        "limitations": "local review required",
    }
    score = validate_advice({"recommendations": [recommendation]}, stress)
    assert score["schema_valid"] == 1
    assert score["grounded_recommendation_ratio"] == 1.0


def test_stage3_end_to_end_without_llm(tmp_path):
    stage1_dir = tmp_path / "stage1"
    stage2_dir = tmp_path / "stage2"
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    frame = _yield_frame()
    stage1_path = _write_rag_stage1_artifact(stage1_dir, frame)
    _, stage1_manifest = read_dataframe_artifact(stage1_path)
    _write_stage2_predictions(stage2_dir, frame, stage1_manifest["sha256"])
    comparison = run_stage3_country(
        "maize",
        "AA",
        stage1_dir,
        output_dir,
        stage2_output_dir=stage2_dir,
        use_llm=False,
    )
    assert set(comparison["scenario"]) == {
        "S1_observed_conditions",
        "S2_bounded_water_recovery",
    }
    result_dir = output_dir / "maize_AA"
    assert (result_dir / "model_validation_repeats.csv").is_file()
    assert (result_dir / "model_validation_summary.csv").is_file()
    assert (result_dir / "deterministic_policy" / "stress_diagnostics.csv").is_file()
    training, manifest = read_dataframe_artifact(
        result_dir / "model_training_data.csv",
        expected_stage="stage3_advisory",
    )
    assert training["year"].max() < 2022
    assert manifest["metadata"]["training_end_year_exclusive"] == 2022
    predictions, _ = read_dataframe_artifact(result_dir / "model_predictions.csv")
    assert set(predictions["repeat"]) == {1, 2, 3, 4, 5}
