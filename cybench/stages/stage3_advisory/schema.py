"""Data contract for Stage 3 using the project's country-level yield tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cybench.config import KEY_LOC, KEY_TARGET, KEY_YEAR
from cybench.stages.artifacts import (
    read_dataframe_artifact,
    read_stage1_dataset,
    read_stage2_predictions,
)

ID_COLUMNS = (KEY_LOC, KEY_YEAR)
REQUIRED_COLUMNS = ID_COLUMNS + (KEY_TARGET,)


@dataclass(frozen=True)
class CountryYieldData:
    country: str
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    lineage: dict | None = None


def feature_role(column: str) -> str:
    """Classify real CY-Bench predictors by how Stage 3 may use them."""
    name = column.lower()
    if name in {"awc", "bulk_density"}:
        return "fixed_context"
    if any(token in name for token in ("prec", "cwb", "ssm")):
        return "water_proxy"
    if any(token in name for token in ("tavg", "tmin", "tmax", "gdd")):
        return "temperature_hazard"
    if "rad" in name:
        return "radiation_hazard"
    if any(token in name for token in ("ndvi", "fpar")):
        return "crop_state_indicator"
    return "other_predictor"


def validate_yield_frame(frame: pd.DataFrame, country: str) -> CountryYieldData:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required CY-Bench columns: {missing}")
    data = frame.copy()
    data[KEY_LOC] = data[KEY_LOC].astype(str)
    for column in data.columns:
        if column != KEY_LOC:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    features = tuple(
        column
        for column in data.columns
        if column not in REQUIRED_COLUMNS and not column.startswith("stage2_")
    )
    if not features:
        raise ValueError("No predictor columns were found")
    required_numeric = [KEY_YEAR, KEY_TARGET, *features]
    invalid = data[required_numeric].isna() | ~np.isfinite(data[required_numeric])
    if invalid.any().any():
        columns = invalid.columns[invalid.any()].tolist()
        raise ValueError(
            "Stage 3 does not impute missing/non-finite values; clean these columns: "
            f"{columns}"
        )
    if data.duplicated(list(ID_COLUMNS)).any():
        raise ValueError("Duplicate adm_id-year rows found")
    if data[KEY_YEAR].nunique() < 3:
        raise ValueError(
            "At least three years are required for historical analog analysis"
        )
    return CountryYieldData(
        country=str(country),
        frame=data.sort_values([KEY_YEAR, KEY_LOC]).reset_index(drop=True),
        feature_columns=features,
    )


def load_stage1_yield_data(
    stage1_output_dir: str | Path,
    crop: str,
    country: str,
    strategy: str = "hardcoded",
) -> CountryYieldData:
    if strategy == "hardcoded":
        frame, manifest = read_stage1_dataset(
            stage1_output_dir, crop, country, strategy=strategy
        )
    else:
        path = (
            Path(stage1_output_dir)
            / "results"
            / f"{crop}_{country}"
            / strategy
            / "feature_dataset.csv"
        )
        frame, manifest = read_dataframe_artifact(
            path,
            expected_stage="stage1_feature_engineering",
            expected_artifact="selected_feature_dataset",
            expected_metadata={"country": country, "advisor": strategy},
        )
    validated = validate_yield_frame(frame, country)
    return CountryYieldData(
        validated.country,
        validated.frame,
        validated.feature_columns,
        lineage={"stage1_dataset_sha256": manifest["sha256"], "strategy": strategy},
    )


def attach_stage2_predictions(
    data: CountryYieldData,
    prediction_dir: str | Path,
    crop: str,
    prediction_stage: str = "stage2_tabpfn_transfer",
) -> CountryYieldData:
    """Attach checksum-verified Stage 2 ensemble summaries to Stage 1 data.

    Stage 2 stores one row per repeat.  Stage 3 consumes the repeat mean as
    the forecast and keeps the sample variance/standard deviation as explicit
    uncertainty rather than selecting an arbitrary repeat.
    """
    frame = data.frame.copy()
    lineage = dict(data.lineage or {})
    for model in ("tabpfn_base", "tabpfn_cn_finetuned"):
        predictions, manifest = read_stage2_predictions(
            prediction_dir, crop, data.country, model, expected_stage=prediction_stage
        )
        expected_source = (data.lineage or {}).get("stage1_dataset_sha256")
        actual_source = manifest.get("metadata", {}).get("source_stage1_sha256")
        if expected_source and actual_source != expected_source:
            raise ValueError(
                f"Stage 2 {model} was produced from a different Stage 1 dataset"
            )
        required = {KEY_LOC, KEY_YEAR, "prediction"}
        if not required <= set(predictions):
            raise ValueError(f"Invalid Stage 2 predictions for {data.country}/{model}")
        prediction_rows = predictions.copy()
        prediction_rows[KEY_LOC] = prediction_rows[KEY_LOC].astype(str)
        prediction_rows[KEY_YEAR] = pd.to_numeric(
            prediction_rows[KEY_YEAR], errors="raise"
        ).astype(int)
        prediction_rows["prediction"] = pd.to_numeric(
            prediction_rows["prediction"], errors="raise"
        )
        if prediction_rows["prediction"].isna().any() or not np.isfinite(
            prediction_rows["prediction"]
        ).all():
            raise ValueError(
                f"Non-finite Stage 2 predictions for {data.country}/{model}"
            )

        if "repeat" in prediction_rows:
            prediction_rows["repeat"] = pd.to_numeric(
                prediction_rows["repeat"], errors="raise"
            ).astype(int)
            if prediction_rows.duplicated([*ID_COLUMNS, "repeat"]).any():
                raise ValueError(
                    f"Duplicate Stage 2 adm_id-year-repeat rows for "
                    f"{data.country}/{model}"
                )
            expected_repeats = int(
                manifest.get("metadata", {}).get(
                    "repeats", prediction_rows["repeat"].nunique()
                )
            )
            counts = prediction_rows.groupby(list(ID_COLUMNS))["repeat"].nunique()
            if not counts.eq(expected_repeats).all():
                raise ValueError(
                    f"Incomplete Stage 2 repeats for {data.country}/{model}: "
                    f"expected {expected_repeats} per adm_id-year"
                )
        else:
            expected_repeats = 1

        if "split" in prediction_rows:
            split_counts = prediction_rows.groupby(list(ID_COLUMNS))["split"].nunique()
            if not split_counts.eq(1).all():
                raise ValueError(
                    f"Conflicting Stage 2 train/test labels for {data.country}/{model}"
                )

        grouped = prediction_rows.groupby(list(ID_COLUMNS), as_index=False)
        values = grouped.agg(
            prediction_mean=("prediction", "mean"),
            prediction_variance=("prediction", "var"),
            prediction_std=("prediction", "std"),
            prediction_count=("prediction", "count"),
        )
        values[["prediction_variance", "prediction_std"]] = values[
            ["prediction_variance", "prediction_std"]
        ].fillna(0.0)
        prefix = f"stage2_{model}"
        values = values.rename(
            columns={
                "prediction_mean": f"{prefix}_prediction",
                "prediction_variance": f"{prefix}_prediction_variance",
                "prediction_std": f"{prefix}_prediction_std",
                "prediction_count": f"{prefix}_n_repeats",
            }
        )
        if "split" in prediction_rows:
            splits = grouped["split"].first().rename(
                columns={"split": f"{prefix}_split"}
            )
            values = values.merge(
                splits, on=list(ID_COLUMNS), how="left", validate="one_to_one"
            )
        frame = frame.merge(
            values, on=list(ID_COLUMNS), how="left", validate="one_to_one"
        )
        lineage[f"stage2_{model}_sha256"] = manifest["sha256"]
        lineage[f"stage2_{model}_aggregation"] = (
            f"mean_and_sample_variance_across_{expected_repeats}_repeats"
        )
    return CountryYieldData(data.country, frame, data.feature_columns, lineage=lineage)
