"""Dataset views that replace labels while preserving aligned predictors."""

import pandas as pd

from cybench.datasets.dataset import Dataset


class ModifiedTargetsDataset(Dataset):
    """A Dataset view used by residual models to train on transformed targets."""

    def __init__(self, dataset: Dataset, modified_targets: pd.DataFrame):
        self._crop = dataset.crop
        self._df_y = modified_targets.sort_index()
        self._dfs_x = dataset._dfs_x
        if set(dataset.indices()) - set(self._df_y.index.values):
            raise ValueError("Modified targets do not cover all dataset indices")
        self._max_season_window_length = dataset.max_season_window_length
        self._allow_incomplete = False
