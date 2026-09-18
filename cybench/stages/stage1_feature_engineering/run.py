"""Run the canonical Stage 1 multi-model feature-parameter experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cybench.stages.runtime import initialize_torch_before_data_libraries

initialize_torch_before_data_libraries()

import pandas as pd

from cybench.stages.artifacts import write_dataframe_artifact
from cybench.stages.config import DEFAULT_STAGE1_OUTPUT_DIR
from cybench.stages.data import (
    build_feature_table,
    chronological_test_years,
    feature_columns,
    load_country_dataset,
    merge_features_labels,
)
from cybench.stages.evaluation import (
    evaluate_temporal_folds,
    metric_row,
    metrics_from_prediction_frame,
    predict_training_fit,
    repeat_seed,
    summarize_repeated_metrics,
)
from cybench.stages.stage1_feature_engineering.estimators import flat_neural_regressor
from cybench.stages.stage1_feature_engineering.selection import reduced_regressor


COUNTRIES = ("CN", "ES", "PT", "ZA", "ZM", "IT", "MX")
ADVISORS = ("hardcode", "qwen_base", "qwen_rag")
MODELS = ("ridge", "xgboost", "svr", "cnn1d", "transformer_flat")
DEFAULT_CONFIG_DIR = DEFAULT_STAGE1_OUTPUT_DIR / "recommendations" / "selected_configs"
DEFAULT_RESULTS_DIR = DEFAULT_STAGE1_OUTPUT_DIR / "results"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _factory(model: str, k: int, epochs: int, seed: int = 42):
    if model in {"ridge", "xgboost", "svr"}:
        return lambda: reduced_regressor(model, k, seed=seed)
    return lambda: flat_neural_regressor(model, k=k, epochs=epochs, seed=seed)


def run_stage1(
    countries: list[str],
    advisors: list[str],
    models: list[str],
    config_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    resume: bool = False,
    repeats: int = 5,
    base_seed: int = 42,
) -> pd.DataFrame:
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for country in countries:
        dataset = load_country_dataset("maize", country)
        test_years = chronological_test_years(dataset.years, 0.3)
        for advisor in advisors:
            selected_path = config_dir / f"maize_{country}" / advisor / "selected_config.json"
            if not selected_path.is_file():
                raise FileNotFoundError(
                    f"Missing prepared Stage 1 configuration: {selected_path}. "
                    "Restore or generate recommendations before evaluation."
                )
            selected = _load_json(selected_path)
            result_dir = output_dir / f"maize_{country}" / advisor
            result_dir.mkdir(parents=True, exist_ok=True)
            generic_frame = merge_features_labels(
                build_feature_table(dataset, selected["config"]), dataset
            )
            generic_columns = feature_columns(generic_frame)
            generic_path = result_dir / "feature_dataset.csv"
            if not (resume and generic_path.is_file()):
                write_dataframe_artifact(
                    generic_frame,
                    generic_path,
                    stage="stage1_feature_engineering",
                    artifact="selected_feature_dataset",
                    metadata={
                        "country": country,
                        "advisor": advisor,
                        "model": None,
                        "candidate_id": selected["candidate_id"],
                        "selected_k": min(int(selected["selected_k"]), len(generic_columns)),
                        "feature_columns": generic_columns,
                    },
                )
            for model in models:
                model_selected_path = (
                    config_dir
                    / f"maize_{country}"
                    / advisor
                    / "by_model"
                    / model
                    / "selected_config.json"
                )
                model_selected = (
                    _load_json(model_selected_path)
                    if model_selected_path.is_file()
                    else selected
                )
                config = model_selected["config"]
                selected_k = int(model_selected["selected_k"])
                frame = merge_features_labels(build_feature_table(dataset, config), dataset)
                columns = feature_columns(frame)
                effective_k = min(selected_k, len(columns))
                feature_path = result_dir / f"feature_dataset_{model}.csv"
                if model_selected_path.is_file() and not (resume and feature_path.is_file()):
                    write_dataframe_artifact(
                        frame,
                        feature_path,
                        stage="stage1_feature_engineering",
                        artifact="selected_feature_dataset",
                        metadata={
                            "country": country,
                            "advisor": advisor,
                            "model": model if model_selected_path.is_file() else None,
                            "candidate_id": model_selected["candidate_id"],
                            "selected_k": effective_k,
                            "feature_columns": columns,
                        },
                    )
                predictions_path = result_dir / f"predictions_{model}.csv"
                metrics_path = result_dir / f"metrics_{model}_repeats.csv"
                print(f"{country} {advisor} {model}", flush=True)
                checkpoint_valid = False
                if resume and predictions_path.is_file() and metrics_path.is_file():
                    predictions = pd.read_csv(predictions_path)
                    repeated_metrics = pd.read_csv(metrics_path)
                    expected_repeats = set(range(1, repeats + 1))
                    checkpoint_valid = (
                        set(predictions.get("repeat", [])) == expected_repeats
                        and set(repeated_metrics.get("repeat", [])) == expected_repeats
                        and set(repeated_metrics.get("split", [])) == {"train", "test"}
                    )
                if checkpoint_valid:
                    print("  checkpoint loaded", flush=True)
                else:
                    prediction_parts, metric_parts = [], []
                    for repeat in range(1, repeats + 1):
                        seed = repeat_seed(base_seed, repeat)
                        factory = _factory(model, effective_k, epochs, seed)
                        test_metrics, test_predictions = evaluate_temporal_folds(
                            frame,
                            columns,
                            factory,
                            test_years,
                        )
                        test_predictions = test_predictions.assign(
                            split="test", repeat=repeat, seed=seed,
                        )
                        training = predict_training_fit(
                            frame,
                            columns,
                            factory,
                            test_years,
                        )
                        training = training.assign(repeat=repeat, seed=seed)
                        prediction_parts.extend((training, test_predictions))
                        identity = {
                            "country": country,
                            "advisor": advisor,
                            "model": model,
                            "candidate_id": model_selected["candidate_id"],
                            "selected_k": effective_k,
                            "repeat": repeat,
                            "seed": seed,
                        }
                        metric_parts.append(
                            metric_row(**identity, split="train", metrics=metrics_from_prediction_frame(training))
                        )
                        metric_parts.append(
                            metric_row(**identity, split="test", metrics=test_metrics)
                        )
                    predictions = pd.concat(prediction_parts, ignore_index=True)
                    repeated_metrics = pd.DataFrame(metric_parts)
                    write_dataframe_artifact(
                        repeated_metrics,
                        metrics_path,
                        stage="stage1_feature_engineering",
                        artifact="repeated_train_test_metrics",
                        metadata={"repeats": repeats, "base_seed": base_seed},
                    )
                    print("  checkpoint written", flush=True)
                write_dataframe_artifact(
                    predictions,
                    predictions_path,
                    stage="stage1_feature_engineering",
                    artifact="train_test_predictions",
                    metadata={
                        "country": country,
                        "advisor": advisor,
                        "model": model,
                        "test_years": test_years,
                        "candidate_id": model_selected["candidate_id"],
                        "selected_k": effective_k,
                        "train_prediction": "in_sample_fit_before_outer_test_per_repeat",
                        "repeats": repeats,
                        "base_seed": base_seed,
                        "neural_epochs": epochs if model in {"cnn1d", "transformer_flat"} else None,
                    },
                )
                rows.extend(repeated_metrics.to_dict("records"))
    result = pd.DataFrame(rows)
    write_dataframe_artifact(
        result,
        output_dir / "metrics_all.csv",
        stage="stage1_feature_engineering",
        artifact="all_repeated_metrics",
        metadata={"countries": countries, "advisors": advisors, "models": models, "neural_epochs": epochs, "repeats": repeats, "base_seed": base_seed},
    )
    summary = summarize_repeated_metrics(
        result,
        ["country", "advisor", "model", "candidate_id", "selected_k", "split"],
    )
    write_dataframe_artifact(
        summary,
        output_dir / "metrics_summary.csv",
        stage="stage1_feature_engineering",
        artifact="repeated_metrics_mean_variance",
        metadata={"repeats": repeats, "variance_ddof": 1, "base_seed": base_seed},
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", default=",".join(COUNTRIES))
    parser.add_argument("--advisors", default=",".join(ADVISORS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    split = lambda value: [item.strip() for item in value.split(",") if item.strip()]
    result = run_stage1(
        split(args.countries),
        split(args.advisors),
        split(args.models),
        Path(args.config_dir),
        Path(args.output_dir),
        epochs=args.epochs,
        resume=args.resume,
        repeats=args.repeats,
        base_seed=args.base_seed,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
