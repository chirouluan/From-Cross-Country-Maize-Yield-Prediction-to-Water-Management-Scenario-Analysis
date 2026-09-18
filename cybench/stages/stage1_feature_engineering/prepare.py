"""Prepare country-conditioned Stage 1 recommendations without outer-test leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cybench.stages.artifacts import write_dataframe_artifact
from cybench.stages.config import (
    DEFAULT_CLIMATE_RAG_METRICS,
    DEFAULT_STAGE1_OUTPUT_DIR,
    qwen_base_variant,
)
from cybench.stages.data import (
    build_feature_table,
    chronological_test_years,
    feature_columns,
    load_country_dataset,
    merge_features_labels,
)
from cybench.stages.evaluation import evaluate_temporal_folds, repeat_seed
from cybench.stages.stage1_feature_engineering.adaptive import (
    baseline_candidate,
    regional_climate_profile,
)
from cybench.stages.stage1_feature_engineering.advisors import (
    recommend_qwen,
    recommend_qwen_rag,
)
from cybench.stages.stage1_feature_engineering.evidence import residualized_stress_evidence
from cybench.stages.stage1_feature_engineering.rag_retrieval import (
    retrieval_rows_frame,
    retrieve_climate_metrics,
)
from cybench.stages.stage1_feature_engineering.selection import (
    expand_candidates,
    extended_inner_years,
    reduced_regressor,
)


COUNTRIES = ("CN", "ES", "PT", "ZA", "ZM", "IT", "MX")
ADVISORS = ("hardcode", "qwen_base", "qwen_rag")
SELECTION_MODELS = ("ridge", "xgboost", "svr", "cnn1d", "transformer_flat")
DEFAULT_RECOMMENDATION_DIR = DEFAULT_STAGE1_OUTPUT_DIR / "recommendations"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _regressor_factory(model: str, selected_k: int, neural_epochs: int, seed: int = 42):
    if model in {"ridge", "xgboost", "svr"}:
        return lambda: reduced_regressor(model, selected_k, seed=seed)
    from cybench.stages.stage1_feature_engineering.estimators import flat_neural_regressor

    return lambda: flat_neural_regressor(
        model, k=selected_k, epochs=neural_epochs, seed=seed
    )


def _prediction_matrix(
    dataset,
    candidates: list[dict],
    years: list[int],
    selected_k: int,
    model: str = "ridge",
    neural_epochs: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    predictions = []
    for candidate in candidates:
        frame = merge_features_labels(build_feature_table(dataset, candidate["config"]), dataset)
        columns = feature_columns(frame)
        k = min(selected_k, len(columns))
        _, prediction = evaluate_temporal_folds(
            frame,
            columns,
            _regressor_factory(model, k, neural_epochs, seed),
            years,
        )
        predictions.append(prediction.rename(columns={"prediction": candidate["candidate_id"]}))
    merged = predictions[0]
    for prediction in predictions[1:]:
        merged = merged.merge(prediction, on=["adm_id", "year", "yield"], how="inner")
    return merged


def _select_k(dataset, inner_years: list[int]) -> tuple[int, list[dict]]:
    baseline = baseline_candidate()
    frame = merge_features_labels(build_feature_table(dataset, baseline["config"]), dataset)
    columns = feature_columns(frame)
    candidates = sorted({min(value, len(columns)) for value in (15, 30, 60, len(columns))})
    rows = []
    for k in candidates:
        metrics, _ = evaluate_temporal_folds(
            frame, columns, lambda size=k: reduced_regressor("ridge", size), inner_years
        )
        rows.append({"selected_k": k, **metrics})
    best = min(row["normalized_rmse"] for row in rows)
    # Prefer the smallest feature set whose error is within 1% of the best.
    selected = min(row["selected_k"] for row in rows if row["normalized_rmse"] <= best * 1.01)
    return selected, rows


def _recommendation(
    advisor: str,
    country: str,
    dataset,
    calibration_end: int,
    raw_dir: Path,
    regenerate_llm: bool,
    rag_metrics_path: Path,
    rag_top_k: int,
) -> dict:
    raw_path = raw_dir / f"maize_{country}" / f"recommendation_{advisor}.json"
    if raw_path.is_file() and not regenerate_llm:
        return _read_json(raw_path)
    profile = regional_climate_profile(dataset, calibration_end)
    if advisor == "qwen_base":
        if not regenerate_llm:
            raise FileNotFoundError(
                f"Missing {raw_path}. Use --regenerate-llm to run the local Qwen model."
            )
        evidence = residualized_stress_evidence(dataset, calibration_end)
        report = recommend_qwen(qwen_base_variant(), "maize", country, profile, evidence)
    elif advisor == "qwen_rag":
        if not regenerate_llm:
            raise FileNotFoundError(
                f"Missing {raw_path}. Use --regenerate-llm to run Base Qwen with RAG."
            )
        retrieval = retrieve_climate_metrics(
            rag_metrics_path,
            country=country,
            crop="maize",
            calibration_end_year=calibration_end,
            top_k=rag_top_k,
        )
        write_dataframe_artifact(
            retrieval_rows_frame(retrieval),
            raw_path.parent / "retrieval_qwen_rag.csv",
            stage="stage1_feature_engineering",
            artifact="structured_climate_rag_retrieval",
            metadata={
                "country": country,
                "crop": "maize",
                "calibration_end_year": calibration_end,
                "retrieval_method": retrieval["retrieval_method"],
                "source_sha256": retrieval["source"]["sha256"],
            },
        )
        evidence = residualized_stress_evidence(dataset, calibration_end)
        report = recommend_qwen_rag(
            qwen_base_variant(),
            "maize",
            country,
            profile,
            evidence,
            retrieval,
        )
    else:
        raise ValueError(f"Unknown advisor: {advisor}")
    _write_json(report, raw_path)
    return report


def prepare_country(
    country: str,
    advisors: list[str],
    output_dir: Path,
    regenerate_llm: bool = False,
    resume: bool = False,
    direct_candidates: bool = False,
    selection_models: tuple[str, ...] = SELECTION_MODELS,
    selection_neural_epochs: int = 20,
    rag_metrics_path: Path = DEFAULT_CLIMATE_RAG_METRICS,
    rag_top_k: int = 6,
    selection_repeats: int = 5,
    base_seed: int = 42,
) -> None:
    dataset = load_country_dataset("maize", country)
    outer_years = chronological_test_years(dataset.years, 0.3)
    calibration_end = min(outer_years) - 1
    inner_years = extended_inner_years(dataset, calibration_end)
    selected_root = output_dir / "selected_configs" / f"maize_{country}"
    raw_root = output_dir / "raw"
    calibration_root = output_dir / "calibration" / f"maize_{country}"

    hardcode_path = selected_root / "hardcode" / "selected_config.json"
    if resume and hardcode_path.is_file():
        selected_k = int(_read_json(hardcode_path)["selected_k"])
        k_scores = []
    else:
        selected_k, k_scores = _select_k(dataset, inner_years)
        hardcode = baseline_candidate()
        _write_json(
            {
                "country": country,
                "advisor": "hardcode",
                "candidate_id": hardcode["candidate_id"],
                "selection_data": "calibration years only",
                "selection_rule": "smallest feature count within 1% of best Ridge NRMSE",
                "config": hardcode["config"],
                "selected_k": selected_k,
            },
            hardcode_path,
        )
    if k_scores:
        write_dataframe_artifact(
            pd.DataFrame(k_scores),
            calibration_root / "hardcode" / "feature_count_scores.csv",
            stage="stage1_feature_engineering",
            artifact="calibration_feature_count_scores",
            metadata={"country": country, "years": inner_years},
        )

    for advisor in advisors:
        if advisor == "hardcode":
            continue
        selected_path = selected_root / advisor / "selected_config.json"
        if resume and selected_path.is_file() and not regenerate_llm:
            print(f"{country} {advisor}: prepared checkpoint loaded", flush=True)
            continue
        report = _recommendation(
            advisor,
            country,
            dataset,
            calibration_end,
            raw_root,
            regenerate_llm,
            rag_metrics_path,
            rag_top_k,
        )
        candidates = (
            report.get("candidates", [])[:3]
            if direct_candidates
            else expand_candidates(report, "maize", budget=9)
        )
        if not candidates:
            raise ValueError(f"No valid candidates for {country} {advisor}")
        advisor_calibration = calibration_root / advisor
        scores = []
        selected_by_model = {}
        for selection_model in selection_models:
            repeat_predictions = []
            for repeat in range(1, selection_repeats + 1):
                seed = repeat_seed(base_seed, repeat)
                prediction = _prediction_matrix(
                    dataset,
                    candidates,
                    inner_years,
                    selected_k,
                    model=selection_model,
                    neural_epochs=selection_neural_epochs,
                    seed=seed,
                )
                repeat_predictions.append(
                    prediction.assign(repeat=repeat, seed=seed)
                )
            predictions = pd.concat(repeat_predictions, ignore_index=True)
            model_scores = [
                {
                    "model": selection_model,
                    "candidate_id": candidate["candidate_id"],
                    "calibration_mse_mean": float(
                        predictions.assign(
                            squared_error=(predictions[candidate["candidate_id"]] - predictions["yield"]) ** 2
                        ).groupby("repeat")["squared_error"].mean().mean()
                    ),
                    "calibration_mse_variance": float(
                        predictions.assign(
                            squared_error=(predictions[candidate["candidate_id"]] - predictions["yield"]) ** 2
                        ).groupby("repeat")["squared_error"].mean().var(ddof=1)
                    ),
                }
                for candidate in candidates
            ]
            scores.extend(model_scores)
            chosen_score = min(model_scores, key=lambda row: row["calibration_mse_mean"])
            chosen = next(
                candidate
                for candidate in candidates
                if candidate["candidate_id"] == chosen_score["candidate_id"]
            )
            selected_by_model[selection_model] = (chosen, chosen_score)
            write_dataframe_artifact(
                predictions,
                advisor_calibration
                / f"inner_candidate_predictions_{advisor}_{selection_model}.csv",
                stage="stage1_feature_engineering",
                artifact="calibration_candidate_predictions",
                metadata={
                    "country": country,
                    "advisor": advisor,
                    "model": selection_model,
                    "years": inner_years,
                },
            )
            _write_json(
                {
                    "country": country,
                    "advisor": advisor,
                    "model": selection_model,
                    "candidate_id": chosen["candidate_id"],
                    "selection_data": "calibration years only",
                    "selection_rule": f"minimum mean pooled calibration MSE across {selection_repeats} repeats under {selection_model}",
                    "calibration_mse_mean": chosen_score["calibration_mse_mean"],
                    "calibration_mse_variance": chosen_score["calibration_mse_variance"],
                    "config": chosen["config"],
                    "selected_k": selected_k,
                },
                selected_root / advisor / "by_model" / selection_model / "selected_config.json",
            )
        write_dataframe_artifact(
            pd.DataFrame(scores),
            advisor_calibration / "selection_scores.csv",
            stage="stage1_feature_engineering",
            artifact="calibration_candidate_scores",
            metadata={"country": country, "advisor": advisor, "metric": "pooled_mse"},
        )
        primary_model = "ridge" if "ridge" in selected_by_model else selection_models[0]
        chosen, chosen_score = selected_by_model[primary_model]
        _write_json(
            {"candidate_ids": [item["candidate_id"] for item in candidates], "selected_k": selected_k},
            advisor_calibration / f"candidate_metadata_{advisor}.json",
        )
        _write_json(
            {
                "country": country,
                "advisor": advisor,
                "candidate_id": chosen["candidate_id"],
                "selection_data": "calibration years only",
                "selection_rule": f"minimum mean pooled calibration MSE across {selection_repeats} repeats under {primary_model}",
                "calibration_mse_mean": chosen_score["calibration_mse_mean"],
                "calibration_mse_variance": chosen_score["calibration_mse_variance"],
                "config": chosen["config"],
                "selected_k": selected_k,
            },
            selected_path,
        )
        choices = ", ".join(
            f"{model}={item[0]['candidate_id']}" for model, item in selected_by_model.items()
        )
        print(f"{country} {advisor}: {choices}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", default=",".join(COUNTRIES))
    parser.add_argument("--advisors", default=",".join(ADVISORS))
    parser.add_argument("--output-dir", default=str(DEFAULT_RECOMMENDATION_DIR))
    parser.add_argument("--regenerate-llm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--direct-candidates", action="store_true")
    parser.add_argument("--selection-models", default=",".join(SELECTION_MODELS))
    parser.add_argument("--selection-neural-epochs", type=int, default=20)
    parser.add_argument("--rag-metrics", default=str(DEFAULT_CLIMATE_RAG_METRICS))
    parser.add_argument("--rag-top-k", type=int, default=6)
    parser.add_argument("--selection-repeats", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()
    countries = [value.strip() for value in args.countries.split(",") if value.strip()]
    advisors = [value.strip() for value in args.advisors.split(",") if value.strip()]
    selection_models = tuple(
        value.strip() for value in args.selection_models.split(",") if value.strip()
    )
    for country in countries:
        prepare_country(
            country,
            advisors,
            Path(args.output_dir),
            args.regenerate_llm,
            args.resume,
            args.direct_candidates,
            selection_models,
            args.selection_neural_epochs,
            Path(args.rag_metrics),
            args.rag_top_k,
            args.selection_repeats,
            args.base_seed,
        )


if __name__ == "__main__":
    main()
