"""Regression model registry for validating Stage 1 feature configurations."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def make_regressor(name: str, seed: int = 42):
    name = name.lower()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "svr":
        return make_pipeline(
            StandardScaler(),
            SVR(C=10.0, epsilon=0.1, gamma="scale", kernel="rbf"),
        )
    raise ValueError(f"Unknown regressor: {name}")
