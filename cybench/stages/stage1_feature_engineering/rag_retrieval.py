"""Leakage-safe structured retrieval from the climate RAG metrics table.

This module deliberately uses relational filters and transparent scalar rankings.
It does not create embeddings or perform vector-database similarity search.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = (
    "parent_id",
    "country",
    "country_zh",
    "country_code",
    "crop",
    "year",
    "latitude",
    "longitude",
    "source_id",
    "coverage_min",
    "valid_days",
)

CLIMATE_COLUMNS = (
    "mean_temp_c",
    "mean_daily_tmax_c",
    "mean_daily_tmin_c",
    "annual_max_tmax_c",
    "annual_min_tmin_c",
    "annual_precip_mm",
    "daily_precip_p95_mm",
    "wet_days_ge_1mm",
    "heavy_precip_days_ge_20mm",
    "dry_days_lt_1mm",
    "max_consecutive_dry_days",
    "frost_days_tmin_lt_0c",
    "heat_days_tmax_gt_35c",
    "gdd10_c_day",
    "mean_relative_humidity_pct",
    "mean_dewpoint_c",
    "mean_vpd_kpa_approx",
    "vpd_p95_kpa_approx",
    "mean_solar_radiation_mj_m2_day",
    "annual_solar_radiation_mj_m2",
    "mean_wind_speed_m_s",
    "hottest_month",
    "wettest_month",
    "driest_month",
    "monthly_precip_cv",
)

RANK_COLUMNS = (
    "mean_temp_c",
    "annual_precip_mm",
    "dry_days_lt_1mm",
    "max_consecutive_dry_days",
    "heat_days_tmax_gt_35c",
    "frost_days_tmin_lt_0c",
    "mean_vpd_kpa_approx",
    "mean_solar_radiation_mj_m2_day",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_metrics(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read the generated CSV and validate pandas-renamed duplicate columns."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    duplicate_pairs = (("year", "year.1"), ("coverage_min", "coverage_min.1"))
    duplicate_audit = {}
    for original, duplicate in duplicate_pairs:
        if duplicate not in frame.columns:
            continue
        left = pd.to_numeric(frame[original], errors="coerce")
        right = pd.to_numeric(frame[duplicate], errors="coerce")
        equal = bool(left.equals(right))
        duplicate_audit[duplicate] = {"canonical": original, "identical": equal}
        if not equal:
            raise ValueError(
                f"Conflicting duplicate columns in climate RAG CSV: {original}, {duplicate}"
            )
        frame = frame.drop(columns=[duplicate])

    required = {"parent_id", "country_code", "crop", "year", *RANK_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Climate RAG CSV is missing required columns: {missing}")
    frame["country_code"] = frame["country_code"].astype(str).str.upper()
    frame["crop"] = frame["crop"].astype(str).str.lower()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    return frame, duplicate_audit


def _robust_z(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    median = numeric.median()
    scale = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0, np.nan)
    scale = scale.fillna(numeric.std().replace(0, np.nan)).fillna(1.0)
    return (numeric - median) / scale


def _representative_rows(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Select auditable climate regimes, then fill remaining slots by typicality."""
    z = _robust_z(frame, RANK_COLUMNS)
    scores = pd.DataFrame(index=frame.index)
    scores["typical"] = z.abs().mean(axis=1)
    scores["hot_dry"] = (
        z["mean_temp_c"] + z["heat_days_tmax_gt_35c"]
        + z["mean_vpd_kpa_approx"] + z["dry_days_lt_1mm"]
        - z["annual_precip_mm"]
    )
    scores["cool_wet"] = (
        z["annual_precip_mm"] + z["frost_days_tmin_lt_0c"]
        - z["mean_temp_c"] - z["mean_vpd_kpa_approx"]
    )
    scores["long_dry_spell"] = z["max_consecutive_dry_days"]

    choices: list[tuple[int, str, float]] = []
    definitions = (
        ("typical", True),
        ("hot_dry", False),
        ("cool_wet", False),
        ("long_dry_spell", False),
    )
    for label, ascending in definitions:
        ordered = scores[label].sort_values(ascending=ascending, kind="stable")
        for index, value in ordered.items():
            if index not in {item[0] for item in choices}:
                choices.append((index, label, float(value)))
                break
        if len(choices) >= top_k:
            break
    if len(choices) < top_k:
        for index, value in scores["typical"].sort_values(kind="stable").items():
            if index not in {item[0] for item in choices}:
                choices.append((index, "typical_fill", float(value)))
            if len(choices) >= top_k:
                break

    selected = frame.loc[[item[0] for item in choices]].copy()
    selected["retrieval_role"] = [item[1] for item in choices]
    selected["retrieval_score"] = [item[2] for item in choices]
    return selected


def retrieve_climate_metrics(
    metrics_path: str | Path,
    *,
    country: str,
    crop: str,
    calibration_end_year: int,
    top_k: int = 6,
) -> dict:
    """Query target-country historical climate rows without using future years."""
    path = Path(metrics_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Climate RAG metrics CSV not found: {path}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    frame, duplicate_audit = _read_metrics(path)
    country, crop = country.upper(), crop.lower()
    eligible = frame[
        frame["country_code"].eq(country)
        & frame["crop"].eq(crop)
        & frame["year"].le(int(calibration_end_year))
    ].copy()
    if "coverage_min" in eligible:
        eligible = eligible[
            pd.to_numeric(eligible["coverage_min"], errors="coerce").ge(0.9)
        ]
    if eligible.empty:
        raise ValueError(
            f"No exact {country}/{crop} rows at or before {calibration_end_year} in {path}"
        )
    selected = _representative_rows(eligible, min(top_k, len(eligible)))
    climate_columns = [column for column in CLIMATE_COLUMNS if column in eligible]
    numeric = eligible[climate_columns].apply(pd.to_numeric, errors="coerce")
    summary = {
        column: {
            "median": float(numeric[column].median()),
            "q10": float(numeric[column].quantile(0.10)),
            "q90": float(numeric[column].quantile(0.90)),
        }
        for column in climate_columns
    }
    output_columns = [
        column for column in (*IDENTITY_COLUMNS, *CLIMATE_COLUMNS)
        if column in selected.columns
    ] + ["retrieval_role", "retrieval_score"]
    rows = selected[output_columns].sort_values("retrieval_role").to_dict("records")
    return {
        "retrieval_method": "exact CSV filters + robust scalar regime ranking (no embeddings)",
        "query": {
            "country_code": country,
            "crop": crop,
            "year_lte": int(calibration_end_year),
            "coverage_min_gte": 0.9,
        },
        "information_boundary": "predictor-only climate rows through calibration_end_year",
        "source": {
            "path": str(path),
            "sha256": _sha256(path),
            "schema_duplicate_audit": duplicate_audit,
        },
        "eligible_row_count": int(len(eligible)),
        "eligible_year_range": [int(eligible["year"].min()), int(eligible["year"].max())],
        "summary": summary,
        "retrieved_rows": rows,
    }


def retrieval_rows_frame(retrieval: dict) -> pd.DataFrame:
    """Return retrieved evidence as a flat table for artifact inspection."""
    return pd.DataFrame(retrieval.get("retrieved_rows", []))
