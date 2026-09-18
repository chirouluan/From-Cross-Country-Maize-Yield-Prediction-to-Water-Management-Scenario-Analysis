"""Validate and summarize canonical Stage 1 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cybench.stages.artifacts import write_dataframe_artifact
from cybench.stages.stage1_feature_engineering.run import DEFAULT_RESULTS_DIR


def summarize(input_path: Path, output_dir: Path):
    metrics = pd.read_csv(input_path)
    if "repeat" in metrics.columns:
        metrics = metrics.query("split == 'test'").groupby(
            ["country", "advisor", "model"], as_index=False
        )[["normalized_rmse", "r2", "mape", "r", "kge"]].mean()
    keys = ["country", "advisor", "model"]
    if metrics.duplicated(keys).any():
        raise ValueError("Duplicate country/advisor/model rows found")
    required = ["normalized_rmse", "r2", "mape", "r", "kge"]
    if metrics[required].isna().any().any():
        raise ValueError("Missing metric values found")
    advisor_order = ["hardcode", "qwen_base", "qwen_rag"]
    advisors = [advisor for advisor in advisor_order if advisor in set(metrics["advisor"])]
    nrmse = metrics.pivot(index=["country", "model"], columns="advisor", values="normalized_rmse")[advisors].reset_index()
    r2 = metrics.pivot(index=["country", "model"], columns="advisor", values="r2")[advisors].reset_index()
    macro = metrics.groupby(["model", "advisor"], as_index=False).agg(
        mean_nrmse=("normalized_rmse", "mean"),
        median_nrmse=("normalized_rmse", "median"),
        mean_r2=("r2", "mean"),
        mean_mape=("mape", "mean"),
        mean_kge=("kge", "mean"),
    )
    metadata = {"rows": len(metrics), "countries": sorted(metrics.country.unique()), "advisors": advisors}
    for frame, filename, artifact in (
        (nrmse, "nrmse_by_country_model.csv", "nrmse_wide"),
        (r2, "r2_by_country_model.csv", "r2_wide"),
        (macro, "macro_summary.csv", "macro_summary"),
    ):
        write_dataframe_artifact(frame, output_dir / filename, stage="stage1_feature_engineering", artifact=artifact, metadata=metadata)
    return macro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_RESULTS_DIR / "metrics_all.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()
    print(summarize(Path(args.input), Path(args.output_dir)).to_string(index=False))


if __name__ == "__main__":
    main()
