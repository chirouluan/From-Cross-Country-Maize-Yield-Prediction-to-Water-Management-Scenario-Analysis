"""Versioned, checksum-verified tabular artifacts shared between stages."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ARTIFACT_SCHEMA_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(data_path: str | Path) -> Path:
    path = Path(data_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def write_dataframe_artifact(
    frame: pd.DataFrame,
    data_path: str | Path,
    *,
    stage: str,
    artifact: str,
    metadata: dict | None = None,
) -> dict:
    """Write a CSV and an adjacent immutable-description manifest."""
    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "artifact": artifact,
        "file": path.name,
        "sha256": file_sha256(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "metadata": metadata or {},
    }
    manifest_path(path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return manifest


def read_dataframe_artifact(
    data_path: str | Path,
    *,
    expected_stage: str | None = None,
    expected_artifact: str | None = None,
    expected_metadata: dict | None = None,
    verify_checksum: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Read an artifact only after its manifest and identity pass validation."""
    path = Path(data_path)
    sidecar = manifest_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Stage artifact data not found: {path}")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Stage artifact manifest not found: {sidecar}")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported artifact schema in {sidecar}")
    if expected_stage and manifest.get("stage") != expected_stage:
        raise ValueError(
            f"Expected stage={expected_stage}, got {manifest.get('stage')} in {sidecar}"
        )
    if expected_artifact and manifest.get("artifact") != expected_artifact:
        raise ValueError(
            f"Expected artifact={expected_artifact}, got "
            f"{manifest.get('artifact')} in {sidecar}"
        )
    for key, value in (expected_metadata or {}).items():
        if manifest.get("metadata", {}).get(key) != value:
            raise ValueError(
                f"Artifact metadata mismatch for {key}: expected {value}, got "
                f"{manifest.get('metadata', {}).get(key)}"
            )
    if verify_checksum and file_sha256(path) != manifest.get("sha256"):
        raise ValueError(f"Artifact checksum mismatch: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame(columns=manifest.get("columns", []))
    if len(frame) != manifest.get("rows"):
        raise ValueError(f"Artifact row-count mismatch: {path}")
    if list(frame.columns) != manifest.get("columns"):
        raise ValueError(f"Artifact column mismatch: {path}")
    return frame, manifest


def stage1_dataset_path(
    stage1_output_dir: str | Path,
    crop: str,
    country: str,
    strategy: str,
) -> Path:
    return Path(stage1_output_dir) / f"{crop}_{country}" / f"dataset_{strategy}.csv"


def stage2_prediction_path(
    stage2_output_dir: str | Path,
    crop: str,
    country: str,
    model: str,
) -> Path:
    return Path(stage2_output_dir) / f"{crop}_{country}" / f"predictions_{model}.csv"


def read_stage1_dataset(
    stage1_output_dir: str | Path,
    crop: str,
    country: str,
    strategy: str = "hardcoded",
) -> tuple[pd.DataFrame, dict]:
    return read_dataframe_artifact(
        stage1_dataset_path(stage1_output_dir, crop, country, strategy),
        expected_stage="stage1_feature_engineering",
        expected_artifact="feature_dataset",
        expected_metadata={
            "crop": crop,
            "country": country,
            "strategy": strategy,
        },
    )


def read_stage2_predictions(
    stage2_output_dir: str | Path,
    crop: str,
    country: str,
    model: str,
    expected_stage: str = "stage2_tabpfn_transfer",
) -> tuple[pd.DataFrame, dict]:
    return read_dataframe_artifact(
        stage2_prediction_path(stage2_output_dir, crop, country, model),
        expected_stage=expected_stage,
        expected_artifact="tabpfn_predictions",
        expected_metadata={"crop": crop, "country": country, "model": model},
    )
