import copy
import numpy as np
import pandas as pd
from datetime import datetime

from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_DATES,
    GDD_BASE_TEMP,
    GDD_UPPER_LIMIT,
)

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, preserving nested defaults."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# Default feature engineering parameters (fallback when no config override)
DEFAULT_FEATURE_CONFIG = {
    # GDD base temperature by crop
    "gdd_base_temp": dict(GDD_BASE_TEMP),
    "gdd_upper_limit": dict(GDD_UPPER_LIMIT),
    # Stress thresholds: indicator -> config
    "stress_thresholds": {
        "tmin": {"operator": "<", "threshold": 0, "label": "tmin<0"},
        "tmax": {"operator": ">", "threshold": 35, "label": "tmax>35"},
        "prec": {"operator": "<", "threshold": 1, "label": "prec<1"},
    },
    # Which vegetation/soil indices to include
    "include_fpar": True,
    "include_ndvi": True,
    "include_soil_moisture": True,
    # Vegetation index aggregation
    "veg_agg_method": "max",
    # Soil moisture aggregation
    "soil_moisture_agg_method": "mean",
}


def fortnight_from_date(dt: datetime):
    """Get the fortnight number from date.

    Args:
      dt: date

    Returns:
      Fortnight number, "YYYY0101" to "YYYY0115" -> 1.
    """
    month = dt.month
    day_of_month = dt.day
    fortnight_number = (month - 1) * 2
    if day_of_month <= 15:
        return fortnight_number + 1
    else:
        return fortnight_number + 2


def dekad_from_date(dt: datetime):
    """Get the dekad number from date.

    Args:
      dt: date

    Returns:
      Dekad number, e.g. "YYYY0101" to "YYYY0110" -> 1,
                         "YYYY0111" to "YYYY0120" -> 2,
                         "YYYY0121" to "YYYY0131" -> 3
    """
    month = dt.month
    day_of_month = dt.day
    dekad = (month - 1) * 3
    if day_of_month <= 10:
        dekad += 1
    elif day_of_month <= 20:
        dekad += 2
    else:
        dekad += 3

    return dekad


def _add_period(df: pd.DataFrame, period_length: str):
    """Add a period column.

    Args:
      df : pd.DataFrame
      period_length: string, which can be "month", "fortnight" or "dekad"

    Returns:
      pd.DataFrame
    """
    if period_length == "month":
        df["period"] = df["date"].dt.month
    elif period_length == "fortnight":
        df["period"] = df.apply(lambda r: fortnight_from_date(r["date"]), axis=1)
    elif period_length == "dekad":
        df["period"] = df.apply(lambda r: dekad_from_date(r["date"]), axis=1)

    return df


def _aggregate_by_period(
    df: pd.DataFrame, index_cols: list, period_col: str, aggrs: dict, ft_cols: dict
):
    """Aggregate data into features by period.

    Args:
      df : pd.DataFrame
      index_cols: list of indices, which are location and year
      period_col: string, column added by add_period()
      aggrs: dict containing columns to aggregate (keys) and corresponding
             aggregation function (values)
      ft_cols: dict for renaming columns to feature columns

    Returns:
      pd.DataFrame with features
    """
    groupby_cols = index_cols + [period_col]
    ft_df = df.groupby(groupby_cols, observed=True).agg(aggrs).reset_index()

    # rename to indicate aggregation
    ft_df = ft_df.rename(columns=ft_cols)

    # pivot to add a feature column for each period
    ft_df = (
        ft_df.pivot_table(index=index_cols, columns=period_col, values=ft_cols.values())
        .fillna(0.0)
        .reset_index()
    )

    # combine names of two column levels
    ft_df.columns = [first + str(second) for first, second in ft_df.columns]

    return ft_df


def _count_threshold(
    df: pd.DataFrame,
    index_cols: list,
    period_col: str,
    indicator: str,
    threshold_exceed: bool = True,
    threshold: float = 0.0,
    ft_name: str = None,
):
    """Aggregate data into features by period.

    Args:
      df : pd.DataFrame
      index_cols: list of indices, which are location and year
      period_col: string, column added by add_period()
      indicator: string, indicator column to aggregate
      threshold_exceed: boolean
      threshold: float

      ft_name: string name for aggregated indicator

    Returns:
      pd.DataFrame with features
    """
    groupby_cols = index_cols + [period_col]
    if threshold_exceed:
        threshold_lambda = lambda x: 1 if (x[indicator] > threshold) else 0
    else:
        threshold_lambda = lambda x: 1 if (x[indicator] < threshold) else 0

    df["meet_thresh"] = df.apply(threshold_lambda, axis=1)
    ft_df = (
        df.groupby(groupby_cols, observed=True)
        .agg(FEATURE=("meet_thresh", "sum"))
        .reset_index()
    )
    df = df.drop(columns=["meet_thresh"])

    if ft_name is not None:
        ft_df = ft_df.rename(columns={"FEATURE": ft_name})

    # pivot to add a feature column for each period
    ft_df = (
        ft_df.pivot_table(index=index_cols, columns=period_col, values=ft_name)
        .fillna(0.0)
        .reset_index()
    )

    # rename period cols
    period_cols = df["period"].unique()
    rename_cols = {p: ft_name + "p" + str(p) for p in period_cols}
    ft_df = ft_df.rename(columns=rename_cols)

    return ft_df


def unpack_time_series(df: pd.DataFrame, indicators: list):
    """Unpack time series data to rows per date.

    Args:
      df : pd.DataFrame

      indicators: list of indicators to unpack

    Returns:
      pd.DataFrame
    """
    if set(indicators).intersection(set(df.columns)) != set(indicators):
        return None

    df["date"] = df.apply(lambda r: r[KEY_DATES][indicators[0]], axis=1)

    df = df.explode(indicators + ["date"]).drop(columns=[KEY_DATES])

    return df


def growing_degree_days(df: pd.DataFrame, tbase: float):
    gdd = np.maximum(0, df["tavg"] - tbase)
    return gdd.sum()


def design_features(
    crop: str,
    input_dfs: dict,
    feature_config: dict = None,
):
    """Design features based domain expertise.

    Args:
      crop (str): crop name, e.g. maize
      input_dfs (dict): keys are input names, values are pd.DataFrames
      feature_config (dict, optional): override default feature engineering
          parameters. Keys match DEFAULT_FEATURE_CONFIG. If None, defaults
          are used (backward compatible).

    Returns:
      pd.DataFrame of features
    """
    cfg = copy.deepcopy(DEFAULT_FEATURE_CONFIG)
    if feature_config is not None:
        cfg = _deep_merge(cfg, feature_config)

    assert "soil" in input_dfs
    soil_df = input_dfs["soil"]
    if "drainage_class" in soil_df.columns:
        soil_df["drainage_class"] = soil_df["drainage_class"].astype(str)
        soil_one_hot = pd.get_dummies(soil_df, prefix="drainage")
        soil_df = pd.concat([soil_df, soil_one_hot], axis=1).drop(
            columns=["drainage_class"]
        )
    soil_features = soil_df

    index_cols = [KEY_LOC, KEY_YEAR]
    period_length = "month"

    assert "meteo" in input_dfs
    weather_df = input_dfs["meteo"]
    weather_df = _add_period(weather_df, period_length)

    fpar_df = None
    if cfg["include_fpar"] and "fpar" in input_dfs:
        fpar_df = input_dfs["fpar"]
        fpar_df = _add_period(fpar_df, period_length)

    ndvi_df = None
    if cfg["include_ndvi"] and "ndvi" in input_dfs:
        ndvi_df = input_dfs["ndvi"]
        ndvi_df = _add_period(ndvi_df, period_length)

    soil_moisture_df = None
    if cfg["include_soil_moisture"] and "soil_moisture" in input_dfs:
        soil_moisture_df = input_dfs["soil_moisture"]
        soil_moisture_df = _add_period(soil_moisture_df, period_length)

    weather_df = weather_df.sort_values(by=index_cols + ["date"])

    weather_df["tavg"] = weather_df["tavg"].astype(float)
    weather_df["gdd"] = (weather_df["tavg"] - cfg["gdd_base_temp"][crop]).clip(
        0.0, cfg["gdd_upper_limit"][crop]
    )
    weather_df["cum_gdd"] = weather_df.groupby(index_cols, observed=True)[
        "gdd"
    ].cumsum()
    weather_df["cwb"] = weather_df["cwb"].astype(float)
    weather_df["prec"] = weather_df["prec"].astype(float)
    weather_df = weather_df.sort_values(by=index_cols + ["date"])
    weather_df["cum_cwb"] = weather_df.groupby(index_cols, observed=True)[
        "cwb"
    ].cumsum()
    weather_df["cum_prec"] = weather_df.groupby(index_cols, observed=True)[
        "prec"
    ].cumsum()

    if fpar_df is not None:
        fpar_df = fpar_df.sort_values(by=index_cols + ["date"])
        fpar_df["fpar"] = fpar_df["fpar"].astype(float)
        fpar_df["cum_fpar"] = fpar_df.groupby(index_cols, observed=True)[
            "fpar"
        ].cumsum()

    if ndvi_df is not None:
        ndvi_df = ndvi_df.sort_values(by=index_cols + ["date"])
        ndvi_df["ndvi"] = ndvi_df["ndvi"].astype(float)
        ndvi_df["cum_ndvi"] = ndvi_df.groupby(index_cols, observed=True)[
            "ndvi"
        ].cumsum()

    # Aggregate weather by period
    avg_weather_cols = ["tmin", "tmax", "tavg", "prec", "rad", "cum_cwb"]
    max_weather_cols = ["cum_gdd", "cum_prec"]
    avg_weather_aggrs = {ind: "mean" for ind in avg_weather_cols}
    max_weather_aggrs = {ind: "max" for ind in max_weather_cols}
    avg_ft_cols = {ind: "mean_" + ind for ind in avg_weather_cols}
    max_ft_cols = {ind: "max_" + ind for ind in max_weather_cols}

    weather_aggrs = {**avg_weather_aggrs, **max_weather_aggrs}

    weather_fts = _aggregate_by_period(
        weather_df, index_cols, "period", weather_aggrs, {**avg_ft_cols, **max_ft_cols}
    )

    # Count threshold exceedances
    operator_to_bool = {">": True, "<": False}
    for ind, thresh_cfg in cfg["stress_thresholds"].items():
        op = thresh_cfg["operator"]
        threshold = float(thresh_cfg["threshold"])
        threshold_exceed = operator_to_bool.get(op)
        assert threshold_exceed is not None, f"Invalid operator {op} for {ind}"
        ft_name = thresh_cfg["label"]
        ind_fts = _count_threshold(
            weather_df,
            index_cols,
            "period",
            ind,
            threshold_exceed,
            threshold,
            ft_name,
        )
        weather_fts = weather_fts.merge(ind_fts, on=index_cols, how="left")
        weather_fts = weather_fts.fillna(0.0)

    all_fts = soil_features.merge(weather_fts, on=[KEY_LOC])

    if fpar_df is not None:
        fpar_agg = cfg["veg_agg_method"]
        fpar_fts = _aggregate_by_period(
            fpar_df,
            index_cols,
            "period",
            {"cum_fpar": fpar_agg},
            {"cum_fpar": f"{fpar_agg}_cum_fpar"},
        )
        all_fts = all_fts.merge(fpar_fts, on=index_cols)

    if ndvi_df is not None:
        ndvi_agg = cfg["veg_agg_method"]
        ndvi_fts = _aggregate_by_period(
            ndvi_df,
            index_cols,
            "period",
            {"cum_ndvi": ndvi_agg},
            {"cum_ndvi": f"{ndvi_agg}_cum_ndvi"},
        )
        all_fts = all_fts.merge(ndvi_fts, on=index_cols)

    if soil_moisture_df is not None:
        sm_agg = cfg["soil_moisture_agg_method"]
        soil_moisture_fts = _aggregate_by_period(
            soil_moisture_df,
            index_cols,
            "period",
            {"ssm": sm_agg},
            {"ssm": f"{sm_agg}_ssm"},
        )
        all_fts = all_fts.merge(soil_moisture_fts, on=index_cols)

    return all_fts
