"""Export leakage-safe China data for an external TabPFN continuation run.

TabPFN continuation training depends on the specific TabPFN training code and
checkpoint format. This command owns the CY-Bench side of that boundary: it
creates immutable train/validation tables and a provenance manifest. The
trainer must write its resulting checkpoint path/hash back into that manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cybench.config import KEY_YEAR
from cybench.stages.artifacts import (
    file_sha256,
    read_dataframe_artifact,
    write_dataframe_artifact,
)
from cybench.stages.config import DEFAULT_STAGE1_OUTPUT_DIR, stage_output_dir
from cybench.stages.data import chronological_test_years, feature_columns


def export_cn_data(
    crop: str,
    validation_fraction: float,
    output_dir: Path,
    stage1_output_dir: Path = DEFAULT_STAGE1_OUTPUT_DIR,
) -> dict:
    frame, stage1_manifest = read_dataframe_artifact(
        stage1_output_dir / "results" / f"{crop}_CN" / "qwen_rag" / "feature_dataset.csv",
        expected_stage="stage1_feature_engineering",
        expected_artifact="selected_feature_dataset",
        expected_metadata={"country": "CN", "advisor": "qwen_rag"},
    )
    validation_years = chronological_test_years(frame[KEY_YEAR], validation_fraction)
    train = frame[~frame[KEY_YEAR].isin(validation_years)]
    validation = frame[frame[KEY_YEAR].isin(validation_years)]
    train_path = output_dir / f"{crop}_CN_train.csv"
    validation_path = output_dir / f"{crop}_CN_validation.csv"
    common_metadata = {
        "crop": crop,
        "country": "CN",
        "feature_strategy": "qwen_rag",
        "source_stage1_sha256": stage1_manifest["sha256"],
        "validation_years": validation_years,
    }
    write_dataframe_artifact(
        train,
        train_path,
        stage="stage2_tabpfn_transfer",
        artifact="cn_finetuning_train",
        metadata=common_metadata,
    )
    write_dataframe_artifact(
        validation,
        validation_path,
        stage="stage2_tabpfn_transfer",
        artifact="cn_finetuning_validation",
        metadata=common_metadata,
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "crop": crop,
        "source_country": "CN",
        "feature_protocol": "Stage 1 Base-Qwen structured-CSV RAG features",
        "feature_columns": feature_columns(frame),
        "train_years": sorted(map(int, train[KEY_YEAR].unique())),
        "validation_years": validation_years,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "files": {
            train_path.name: file_sha256(train_path),
            validation_path.name: file_sha256(validation_path),
        },
        "trainer": {
            "implementation": None,
            "base_checkpoint_sha256": None,
            "hyperparameters": None,
            "output_checkpoint": None,
            "output_checkpoint_sha256": None,
        },
    }
    (output_dir / "cn_finetuning_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", default="maize")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--stage1-output-dir", default=str(DEFAULT_STAGE1_OUTPUT_DIR))
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    output = stage_output_dir("stage2_tabpfn_transfer/cn_finetuning", args.output_dir)
    print(
        json.dumps(
            export_cn_data(
                args.crop,
                args.validation_fraction,
                output,
                Path(args.stage1_output_dir),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
