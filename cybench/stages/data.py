"""Shared, leakage-aware data preparation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cybench.config import (
    KEY_DATES,
    KEY_LOC,
    KEY_TARGET,
    KEY_YEAR,
    SOIL_PROPERTIES,
    TIME_SERIES_INPUTS,
)
from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset
from cybench.util.data import data_to_pandas
from cybench.util.features import design_features, unpack_time_series

ID_COLUMNS = (KEY_LOC, KEY_YEAR)


def load_country_dataset(crop: str, country: str) -> Dataset:
    df_y, dfs_x = load_dfs_crop(crop, [country])
    if df_y.empty:
        raise ValueError(f"No aligned samples found for {crop}_{country}")
    return Dataset(crop, data_target=df_y, data_inputs=dfs_x)


def build_feature_table(
    dataset: Dataset, feature_config: dict | None = None
) -> pd.DataFrame:
    """Create one expert-feature row per location-year from an aligned Dataset."""
    soil = data_to_pandas(
        dataset, data_cols=[KEY_LOC] + SOIL_PROPERTIES
    ).drop_duplicates()
    inputs: dict[str, pd.DataFrame] = {"soil": soil}
    for source, columns in TIME_SERIES_INPUTS.items():
        frame = data_to_pandas(
            dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_DATES] + columns
        )
        frame = unpack_time_series(frame, columns)
        frame = frame.astype({column: "float" for column in columns})
        # Interpolation is confined to each location-year-source sequence.
        frame = (
            frame.set_index([KEY_LOC, KEY_YEAR, "date"])
            .sort_index()
            .groupby(level=[0, 1], group_keys=False)
            .apply(
                lambda group: group.interpolate(method="linear"), include_groups=False
            )
            .reset_index()
        )
        inputs[source] = frame
    return design_features(dataset.crop, inputs, feature_config=feature_config)


def labels_table(dataset: Dataset) -> pd.DataFrame:
    return data_to_pandas(dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET])


def merge_features_labels(features: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    merged = features.merge(labels_table(dataset), on=list(ID_COLUMNS), how="inner")
    if merged.empty:
        raise ValueError("Feature and label tables have no matching location-year rows")
    return merged.sort_values([KEY_YEAR, KEY_LOC]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {KEY_LOC, KEY_YEAR, KEY_TARGET, "country", "strategy", "model"}
    return [column for column in frame.columns if column not in excluded]


def chronological_test_years(years, test_fraction: float = 0.3) -> list[int]:
    unique = sorted({int(year) for year in years})
    if len(unique) < 2:
        raise ValueError("At least two years are required for temporal evaluation")
    count = max(1, round(len(unique) * test_fraction))
    return unique[-count:]


def finite_rows(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].to_numpy(dtype=float)
    return pd.Series(np.isfinite(values).all(axis=1), index=frame.index)
