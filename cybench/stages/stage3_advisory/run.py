"""Run data-grounded crop-stress diagnosis and bounded recovery scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cybench.stages.runtime import initialize_torch_before_data_libraries

initialize_torch_before_data_libraries()

import pandas as pd

from cybench.config import KEY_YEAR
from cybench.stages.artifacts import write_dataframe_artifact
from cybench.stages.config import (
    DEFAULT_CLIMATE_RAG_METRICS,
    DEFAULT_STAGE1_OUTPUT_DIR,
    DEFAULT_STAGE2_OUTPUT_DIR,
    qwen_base_variant,
    stage_output_dir,
    validate_llm_variant,
)
from cybench.stages.stage1_feature_engineering.rag_retrieval import (
    retrieval_rows_frame,
    retrieve_climate_metrics,
)
from cybench.stages.evaluation import repeat_seed
from cybench.stages.stage3_advisory.llm import (
    advice_prompt,
    frame_fingerprint,
    llm_agent,
    policy_prompt,
    validate_advice,
)
from cybench.stages.stage3_advisory.models import fit_yield_response_model
from cybench.stages.stage3_advisory.optimizer import (
    OptimizationPolicy,
    diagnose_stress,
    optimize_recovery_scenario,
    policy_dict,
    sanitize_policy,
    scenario_summary,
    select_current_conditions,
)
from cybench.stages.stage3_advisory.schema import (
    attach_stage2_predictions,
    load_stage1_yield_data,
)


DEFAULT_COUNTRIES = ("PT", "ZA", "ES", "IT", "MX", "ZM")
DEFAULT_ADVISORS = ("deterministic_policy", "qwen_base", "qwen_rag")


def _run_policy(name, data, current, model, policy, output_dir, metadata, artifact_stage):
    branch = output_dir / name
    branch.mkdir(parents=True, exist_ok=True)
    stress = diagnose_stress(data, current, model, policy)
    recovery, changes = optimize_recovery_scenario(data, current, stress, model, policy)
    summary = scenario_summary(current, recovery, stress, model)
    for frame, filename, artifact in (
        (stress, "stress_diagnostics.csv", "stress_diagnostics"),
        (recovery, "recovery_scenarios.csv", "recovery_scenarios"),
        (changes, "accepted_proxy_changes.csv", "accepted_proxy_changes"),
        (summary, "scenario_summary.csv", "scenario_summary"),
    ):
        write_dataframe_artifact(
            frame,
            branch / filename,
            stage=artifact_stage,
            artifact=artifact,
            metadata={**metadata, "policy": name},
        )
    (branch / "optimization_policy.json").write_text(
        json.dumps(policy_dict(policy), indent=2), encoding="utf-8"
    )
    return summary, stress, recovery, changes


def run_stage3_country(
    crop: str,
    country: str,
    stage1_output_dir: str | Path,
    output_dir: Path,
    stage2_output_dir: str | Path = DEFAULT_STAGE2_OUTPUT_DIR,
    target_year: int | None = None,
    use_llm: bool = True,
    base_model_dir: str | None = None,
    feature_strategy: str = "qwen_rag",
    stage2_prediction_stage: str = "stage2_tabpfn_transfer",
    artifact_stage: str = "stage3_advisory",
    rag_metrics_path: str | Path = DEFAULT_CLIMATE_RAG_METRICS,
    rag_top_k: int = 6,
    repeats: int = 5,
    base_seed: int = 42,
    advisors: tuple[str, ...] = DEFAULT_ADVISORS,
) -> pd.DataFrame:
    data = load_stage1_yield_data(
        stage1_output_dir, crop, country, strategy=feature_strategy
    )
    data = attach_stage2_predictions(
        data, stage2_output_dir, crop, prediction_stage=stage2_prediction_stage
    )
    stage2_columns = [
        column for column in data.frame if column.startswith("stage2_")
    ]
    if not stage2_columns:
        raise ValueError("Stage 3 requires both Stage 2 prediction columns")
    # Stage 2 intentionally skips rows with non-finite predictors because this
    # project does not impute missing values. Select the latest *forecastable*
    # row per administrative unit instead of failing when its absolute latest
    # observation was skipped by Stage 2.
    decision_frame = data.frame.dropna(subset=stage2_columns).copy()
    if decision_frame.empty:
        raise ValueError("Stage 3 found no rows with complete Stage 2 predictions")
    decision_data = type(data)(
        data.country, decision_frame, data.feature_columns, lineage=data.lineage
    )
    current = select_current_conditions(decision_data, target_year)
    missing_context = current[stage2_columns].isna().any(axis=1)
    if missing_context.any():
        raise ValueError("Stage 3 decision-row filtering failed unexpectedly")
    training_end_year = int(current[KEY_YEAR].min())
    models = [
        fit_yield_response_model(
            data,
            seed=repeat_seed(base_seed, repeat),
            training_end_year=training_end_year,
        )
        for repeat in range(1, repeats + 1)
    ]
    model = models[0]
    country_dir = output_dir / f"{crop}_{country}"
    country_dir.mkdir(parents=True, exist_ok=True)
    lineage_metadata = {
        "crop": crop,
        "country": country,
        "feature_strategy": feature_strategy,
        "training_end_year_exclusive": training_end_year,
        "decision_snapshot_rule": "latest_row_with_both_stage2_predictions",
        **(data.lineage or {}),
    }
    training_frame = data.frame[data.frame[KEY_YEAR] < training_end_year].copy()
    prediction_frames = []
    validation_rows = []
    for repeat, repeated_model in enumerate(models, start=1):
        seed = repeat_seed(base_seed, repeat)
        validation_rows.append(
            {"repeat": repeat, "seed": seed, **repeated_model.validation_metrics}
        )
        for split_name, subset in (("train", training_frame), ("test", current)):
            prediction = subset[["adm_id", "year", "yield"]].copy()
            prediction["prediction"] = repeated_model.predict(subset)["predicted_yield"].to_numpy()
            prediction["split"] = split_name
            prediction["repeat"] = repeat
            prediction["seed"] = seed
            prediction_frames.append(prediction)
    write_dataframe_artifact(
        pd.concat(prediction_frames, ignore_index=True),
        country_dir / "model_predictions.csv",
        stage=artifact_stage,
        artifact="yield_response_train_test_predictions",
        metadata={**lineage_metadata, "model": "yield_response_extratrees", "repeats": repeats, "base_seed": base_seed},
    )
    write_dataframe_artifact(
        training_frame,
        country_dir / "model_training_data.csv",
        stage=artifact_stage,
        artifact="model_training_data",
        metadata=lineage_metadata,
    )
    write_dataframe_artifact(
        current,
        country_dir / "decision_input_snapshot.csv",
        stage=artifact_stage,
        artifact="decision_input_snapshot",
        metadata=lineage_metadata,
    )
    validation_frame = pd.DataFrame(validation_rows)
    write_dataframe_artifact(
        validation_frame,
        country_dir / "model_validation_repeats.csv",
        stage=artifact_stage,
        artifact="yield_response_repeated_validation",
        metadata={**lineage_metadata, "repeats": repeats, "base_seed": base_seed},
    )
    validation_metrics = ["forward_r2", "forward_rmse", "forward_mae"]
    validation_summary = pd.DataFrame(
        {
            "metric": validation_metrics,
            "mean": [float(validation_frame[column].mean()) for column in validation_metrics],
            "variance": [float(validation_frame[column].var(ddof=1)) for column in validation_metrics],
        }
    )
    write_dataframe_artifact(
        validation_summary,
        country_dir / "model_validation_summary.csv",
        stage=artifact_stage,
        artifact="yield_response_validation_mean_variance",
        metadata={**lineage_metadata, "repeats": repeats, "variance_ddof": 1},
    )
    write_dataframe_artifact(
        current[["adm_id", "year", "yield", *stage2_columns]],
        country_dir / "stage2_forecast_context.csv",
        stage=artifact_stage,
        artifact="stage2_forecast_context",
        metadata=lineage_metadata,
    )

    rows = []
    if "deterministic_policy" in advisors:
        summary, _, _, _ = _run_policy(
            "deterministic_policy",
            data,
            current,
            model,
            OptimizationPolicy(),
            country_dir,
            lineage_metadata,
            artifact_stage,
        )
        rows.append(summary.assign(country=country, policy="deterministic_policy"))

    llm_advisors = tuple(name for name in advisors if name != "deterministic_policy")
    if use_llm and llm_advisors:
        base_variant = qwen_base_variant(base_model_dir)
        validate_llm_variant(base_variant)
        retrieval = None
        if "qwen_rag" in llm_advisors:
            retrieval = retrieve_climate_metrics(
                rag_metrics_path,
                country=country,
                crop=crop,
                calibration_end_year=training_end_year - 1,
                top_k=rag_top_k,
            )
            write_dataframe_artifact(
                retrieval_rows_frame(retrieval),
                country_dir / "retrieval_qwen_rag.csv",
                stage=artifact_stage,
                artifact="structured_climate_rag_retrieval",
                metadata={
                    **lineage_metadata,
                    "retrieval_cutoff_year": training_end_year - 1,
                    "retrieval_method": retrieval["retrieval_method"],
                    "source_sha256": retrieval["source"]["sha256"],
                },
            )
        branches = tuple(
            (name, retrieval if name == "qwen_rag" else None) for name in llm_advisors
        )
        for advisor_name, branch_retrieval in branches:
            agent = llm_agent(base_variant, advisor_name=advisor_name)
            try:
                proposal = agent.run_advisor(
                    "stress_scenario_policy",
                    policy_prompt(data, branch_retrieval),
                    cache_key=(
                        f"policy_{advisor_name}_{country}_{frame_fingerprint(data.frame)}_"
                        f"{retrieval['source']['sha256'][:16] if branch_retrieval else 'no_rag'}"
                    ),
                    fallback=None,
                )
                summary, stress, recovery, changes = _run_policy(
                    advisor_name,
                    data,
                    current,
                    model,
                    sanitize_policy(proposal),
                    country_dir,
                    lineage_metadata,
                    artifact_stage,
                )
                advice = agent.run_advisor(
                    "stress_adaptation_advice",
                    advice_prompt(
                        data,
                        summary,
                        stress,
                        recovery,
                        changes,
                        model.validation_metrics,
                        branch_retrieval,
                    ),
                    cache_key=(
                        f"advice_{advisor_name}_{country}_{frame_fingerprint(stress)}_"
                        f"{retrieval['source']['sha256'][:16] if branch_retrieval else 'no_rag'}"
                    ),
                    fallback=None,
                )
                branch = country_dir / advisor_name
                (branch / "llm_policy_proposal.json").write_text(
                    json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                (branch / "llm_advice.json").write_text(
                    json.dumps(advice, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                grounding = validate_advice(advice, stress)
                (branch / "llm_grounding_check.json").write_text(
                    json.dumps(grounding, indent=2), encoding="utf-8"
                )
                rows.append(
                    summary.assign(country=country, policy=advisor_name, **grounding)
                )
            finally:
                # Keep only one 9B checkpoint resident at a time.
                agent.unload()
    comparison = pd.concat(rows, ignore_index=True)
    write_dataframe_artifact(
        comparison,
        country_dir / "policy_comparison.csv",
        stage=artifact_stage,
        artifact="policy_comparison",
        metadata=lineage_metadata,
    )
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", default="maize")
    parser.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES))
    parser.add_argument("--stage1-output-dir", default=str(DEFAULT_STAGE1_OUTPUT_DIR))
    parser.add_argument("--stage2-output-dir", default=str(DEFAULT_STAGE2_OUTPUT_DIR))
    parser.add_argument("--target-year", type=int)
    parser.add_argument(
        "--advisors",
        default=",".join(DEFAULT_ADVISORS),
        help="Comma-separated subset of: " + ",".join(DEFAULT_ADVISORS),
    )
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--base-model-dir")
    parser.add_argument("--rag-metrics", default=str(DEFAULT_CLIMATE_RAG_METRICS))
    parser.add_argument("--rag-top-k", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    advisors = tuple(item.strip() for item in args.advisors.split(",") if item.strip())
    unknown = sorted(set(advisors) - set(DEFAULT_ADVISORS))
    if unknown:
        parser.error(f"unknown advisors: {unknown}; choose from {list(DEFAULT_ADVISORS)}")
    if not advisors:
        parser.error("--advisors must select at least one branch")
    output = stage_output_dir("stage3_advisory", args.output_dir)
    results = []
    for country in [item.strip() for item in args.countries.split(",") if item.strip()]:
        results.append(
            run_stage3_country(
                args.crop,
                country,
                args.stage1_output_dir,
                output,
                stage2_output_dir=args.stage2_output_dir,
                target_year=args.target_year,
                use_llm=not args.skip_llm,
                base_model_dir=args.base_model_dir,
                rag_metrics_path=args.rag_metrics,
                rag_top_k=args.rag_top_k,
                repeats=args.repeats,
                base_seed=args.base_seed,
                advisors=advisors,
            )
        )
    combined = pd.concat(results, ignore_index=True)
    write_dataframe_artifact(
        combined,
        output / "policy_comparison_all.csv",
        stage="stage3_advisory",
        artifact="policy_comparison_all",
        metadata={"crop": args.crop, "feature_strategy": "qwen_rag"},
    )
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
