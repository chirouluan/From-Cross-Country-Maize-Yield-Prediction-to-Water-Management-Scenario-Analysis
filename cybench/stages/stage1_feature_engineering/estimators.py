"""Adapters for the five Stage 1 downstream regression models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_regression

from cybench.config import KEY_TARGET
from cybench.models.nn_models import CNN1DRegressor, TransformerRegressor


class FlatNeuralRegressor:
    """Expose CY-Bench flat neural models through fit(X, y)/predict(X)."""

    def __init__(self, architecture: str, k: int, epochs: int = 50, seed: int = 42):
        self.architecture = architecture
        self.k = k
        self.epochs = epochs
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.selector = None
        self.model = None
        self.feature_names = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        size = min(self.k, X.shape[1])
        self.selector = SelectKBest(score_func=f_regression, k=size)
        selected = self.selector.fit_transform(X, y).astype(np.float32)
        self.feature_names = [f"feature_{index}" for index in range(size)]
        frame = pd.DataFrame(selected, columns=self.feature_names)
        frame[KEY_TARGET] = y
        if self.architecture == "cnn1d":
            self.model = CNN1DRegressor(feature_cols=self.feature_names)
        elif self.architecture == "transformer_flat":
            self.model = TransformerRegressor(feature_cols=self.feature_names)
        else:
            raise ValueError(f"Unknown flat neural architecture: {self.architecture}")
        self.model.fit(
            frame,
            epochs=self.epochs,
            batch_size=32,
            patience=10,
            device=self.device,
            seed=self.seed,
        )
        return self

    def predict(self, X):
        selected = self.selector.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        frame = pd.DataFrame(selected, columns=self.feature_names)
        prediction, _ = self.model.predict(frame, device=self.device)
        return np.asarray(prediction).reshape(-1)


def flat_neural_regressor(architecture: str, k: int, epochs: int, seed: int = 42):
    return FlatNeuralRegressor(architecture, k=k, epochs=epochs, seed=seed)
