"""Compare default and CN-adapted TabPFN checkpoints across target countries."""

from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path

from cybench.stages.runtime import initialize_torch_before_data_libraries

initialize_torch_before_data_libraries()

import pandas as pd

from cybench.config import KEY_YEAR
from cybench.stages.artifacts import (
    read_dataframe_artifact,
    write_dataframe_artifact,
)
from cybench.stages.config import (
    DEFAULT_STAGE1_OUTPUT_DIR,
    stage_output_dir,
    tabpfn_checkpoints,
)
from cybench.stages.data import (
    chronological_test_years,
    feature_columns,
)
from cybench.stages.evaluation import (
    evaluate_temporal_folds,
    metric_row,
    metrics_from_prediction_frame,
    predict_training_fit,
    repeat_seed,
    summarize_repeated_metrics,
)


FEATURE_STRATEGY = "qwen_rag"
DEFAULT_COUNTRIES = ("CN", "PT", "ZA", "ES", "IT", "MX", "ZM")
CN_STRICT_TEST_YEARS = (2020, 2021, 2022)


def read_stage1_features(stage1_output_dir: Path, crop: str, country: str):
    path = stage1_output_dir / "results" / f"{crop}_{country}" / FEATURE_STRATEGY / "feature_dataset.csv"
    return read_dataframe_artifact(
        path,
        expected_stage="stage1_feature_engineering",
        expected_artifact="selected_feature_dataset",
        expected_metadata={"country": country, "advisor": FEATURE_STRATEGY},
    )


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoints(base: Path, finetuned: Path) -> dict[str, str]:
    hashes = {
        "tabpfn_base": checkpoint_sha256(base),
        "tabpfn_cn_finetuned": checkpoint_sha256(finetuned),
    }
    if hashes["tabpfn_base"] == hashes["tabpfn_cn_finetuned"]:
        raise ValueError("Base and CN-finetuned TabPFN checkpoints are identical")
    return hashes


def make_tabpfn(checkpoint: Path, device: str, seed: int = 42):
    try:
        from tabpfn import TabPFNRegressor
        from tabpfn.inference_config import InferenceConfig
        from tabpfn import model_loading
    except ImportError as exc:
        raise RuntimeError(
            "Stage 2 requires the optional 'tabpfn' package. Install the version "
            "used to create both checkpoints before running the comparison."
        ) from exc
    # The CN continuation checkpoint was produced by a TabPFN build whose
    # inference metadata contains fields no longer present in newer releases.
    # They control preprocessing/runtime limits rather than learned weights.
    # Filter only unknown metadata keys at load time so the original checkpoint
    # and its experiment-tracked SHA256 remain untouched.
    if not getattr(model_loading, "_cybench_compat_installed", False):
        original_rename = model_loading._rename_old_inference_config_keys
        supported = set(inspect.signature(InferenceConfig).parameters)

        def compatible_inference_config(config):
            renamed = original_rename(config)
            return {key: value for key, value in renamed.items() if key in supported}

        model_loading._rename_old_inference_config_keys = compatible_inference_config
        model_loading._cybench_compat_installed = True
    kwargs = {"model_path": str(checkpoint), "device": device}
    if "random_state" in inspect.signature(TabPFNRegressor).parameters:
        kwargs["random_state"] = seed
    return TabPFNRegressor(**kwargs)


def run_stage2(
    crop: str,
    countries: list[str],
    base_checkpoint: Path,
    finetuned_checkpoint: Path,
    device: str,
    test_fraction: float,
    output_dir: Path,
    stage1_output_dir: Path = DEFAULT_STAGE1_OUTPUT_DIR,
    resume: bool = False,
    repeats: int = 5,
    base_seed: int = 42,
) -> pd.DataFrame:
    hashes = validate_checkpoints(base_checkpoint, finetuned_checkpoint)
    checkpoints = {
        "tabpfn_base": base_checkpoint,
        "tabpfn_cn_finetuned": finetuned_checkpoint,
    }
    rows = []
    for country in countries:
        frame, stage1_manifest = read_stage1_features(
            stage1_output_dir, crop, country
        )
        columns = feature_columns(frame)
        if country == "CN":
            available_years = {int(year) for year in frame[KEY_YEAR]}
            test_years = [year for year in CN_STRICT_TEST_YEARS if year in available_years]
            if len(test_years) != len(CN_STRICT_TEST_YEARS):
                raise ValueError("CN strict test requires complete 2020-2022 data")
        else:
            test_years = chronological_test_years(frame[KEY_YEAR], test_fraction)
        country_dir = output_dir / f"{crop}_{country}"
        country_dir.mkdir(parents=True, exist_ok=True)
        input_metadata = {
            "crop": crop,
            "country": country,
            "feature_strategy": FEATURE_STRATEGY,
            "stage1_candidate_id": stage1_manifest["metadata"].get("candidate_id"),
            "source_stage1_sha256": stage1_manifest["sha256"],
            "test_years": list(map(int, test_years)),
            "feature_columns": columns,
        }
        write_dataframe_artifact(
            frame,
            country_dir / "input_dataset.csv",
            stage="stage2_tabpfn_transfer",
            artifact="evaluation_input",
            metadata=input_metadata,
        )
        for model_name, checkpoint in checkpoints.items():
            prediction_path = country_dir / f"predictions_{model_name}.csv"
            metrics_path = country_dir / f"metrics_{model_name}_repeats.csv"
            prediction_metadata = {
                **input_metadata,
                "model": model_name,
                "checkpoint_sha256": hashes[model_name],
            }
            print(f"{country} {model_name} ({len(columns)} RAG-selected features)", flush=True)
            checkpoint_valid = False
            if resume and prediction_path.is_file() and metrics_path.is_file():
                predictions = pd.read_csv(prediction_path)
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
                    factory = lambda path=checkpoint, run_seed=seed: make_tabpfn(
                        path, device, run_seed
                    )
                    test_metrics, test_predictions = evaluate_temporal_folds(
                        frame, columns, factory, test_years
                    )
                    test_predictions = test_predictions.assign(
                        split="test", repeat=repeat, seed=seed
                    )
                    training = predict_training_fit(frame, columns, factory, test_years)
                    training = training.assign(repeat=repeat, seed=seed)
                    prediction_parts.extend((training, test_predictions))
                    identity = {
                        "country": country,
                        "model": model_name,
                        "checkpoint_sha256": hashes[model_name],
                        "test_years": ";".join(map(str, test_years)),
                        "n_features": len(columns),
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
                    stage="stage2_tabpfn_transfer",
                    artifact="repeated_train_test_metrics",
                    metadata={"repeats": repeats, "base_seed": base_seed},
                )
                print("  checkpoint written", flush=True)
            prediction_frame = predictions.assign(country=country, model=model_name)
            write_dataframe_artifact(
                prediction_frame,
                prediction_path,
                stage="stage2_tabpfn_transfer",
                artifact="tabpfn_predictions",
                metadata={**prediction_metadata, "train_prediction": "in_sample_fit_before_outer_test_per_repeat", "repeats": repeats, "base_seed": base_seed},
            )
            rows.extend(repeated_metrics.to_dict("records"))
        write_dataframe_artifact(
            summarize_repeated_metrics(
                pd.DataFrame(rows).query("country == @country"),
                ["country", "model", "checkpoint_sha256", "test_years", "n_features", "split"],
            ),
            country_dir / "metrics.csv",
            stage="stage2_tabpfn_transfer",
            artifact="repeated_metrics_mean_variance",
            metadata={"crop": crop, "country": country, "repeats": repeats},
        )
    result = pd.DataFrame(rows)
    write_dataframe_artifact(
        result,
        output_dir / "metrics_all.csv",
        stage="stage2_tabpfn_transfer",
        artifact="all_repeated_metrics",
        metadata={"crop": crop, "countries": countries, "repeats": repeats, "base_seed": base_seed},
    )
    summary = summarize_repeated_metrics(
        result,
        ["country", "model", "checkpoint_sha256", "test_years", "n_features", "split"],
    )
    write_dataframe_artifact(
        summary,
        output_dir / "metrics_summary.csv",
        stage="stage2_tabpfn_transfer",
        artifact="repeated_metrics_mean_variance",
        metadata={"crop": crop, "countries": countries, "repeats": repeats, "variance_ddof": 1},
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", default="maize")
    parser.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES))
    parser.add_argument(
        "--base-checkpoint",
        help="Override the configured original TabPFN checkpoint",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        help="Override the configured agronomy-finetuned TabPFN checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--stage1-output-dir",
        default=str(DEFAULT_STAGE1_OUTPUT_DIR),
        help="Saved Stage 1 artifact directory",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = stage_output_dir("stage2_tabpfn_transfer/evaluation", args.output_dir)
    base_checkpoint, finetuned_checkpoint = tabpfn_checkpoints(
        args.base_checkpoint, args.finetuned_checkpoint
    )
    result = run_stage2(
        crop=args.crop,
        countries=[item.strip() for item in args.countries.split(",") if item.strip()],
        base_checkpoint=base_checkpoint,
        finetuned_checkpoint=finetuned_checkpoint,
        device=args.device,
        test_fraction=args.test_fraction,
        output_dir=output,
        stage1_output_dir=Path(args.stage1_output_dir),
        resume=args.resume,
        repeats=args.repeats,
        base_seed=args.base_seed,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
