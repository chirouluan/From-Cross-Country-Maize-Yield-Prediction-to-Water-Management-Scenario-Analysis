import random
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
from collections.abc import Iterable

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from tsai.models.InceptionTime import InceptionTime
from tsai.models.TST import TST

from cybench.datasets.torch_dataset import TorchDataset
from cybench.datasets.dataset import Dataset

from cybench.models.model import BaseModel
from cybench.util.data import data_to_pandas
from cybench.util.features import unpack_time_series, design_features

from cybench.config import (
    KEY_LOC,
    KEY_YEAR,
    KEY_TARGET,
    KEY_DATES,
    SOIL_PROPERTIES,
    LOCATION_PROPERTIES,
    TIME_SERIES_INPUTS,
    STATIC_PREDICTORS,
    TIME_SERIES_PREDICTORS,
    ALL_PREDICTORS,
)


def separate_ts_static_inputs(batch: dict) -> tuple:
    """Stack time series and static inputs separately.

    Args:
      batch (dict): batched inputs

    Returns:
      A tuple of torch tensors for time series and static inputs
    """
    ts = torch.cat([batch[k].unsqueeze(2) for k in TIME_SERIES_PREDICTORS], dim=2)
    static = torch.cat([batch[k].unsqueeze(1) for k in STATIC_PREDICTORS], dim=1)

    return ts, static


class BaseNNModel(BaseModel, nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        super(nn.Module, self).__init__()

        self._norm_params = None
        self._init_args = kwargs
        self._interpolate_time_series = False
        if "interpolate_time_series" in kwargs:
            self._interpolate_time_series = kwargs["interpolate_time_series"]

        self._aggregate_time_series_to = None
        if "aggregate_time_series_to" in kwargs:
            # aggregation requires interpolation to ensure same number of time steps
            assert "interpolate_time_series" in kwargs
            self._aggregate_time_series_to = kwargs["aggregate_time_series_to"]

        self._logger = logging.getLogger(__name__)

    def fit(
        self,
        dataset: Dataset,
        optimize_hyperparameters: bool = False,
        param_space: dict = None,
        optim_kwargs: dict = {},
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        **fit_params,
    ):
        """Fit or train the model.

        Args:
          dataset (Dataset): training dataset.
          optimize_hyperparameters (bool): whether to tune hyperparameters
          param_space (dict): each entry is a hyperparameter name and list or range of values
          optim_kwargs (dict): arguments to the optimizer
          device (str): the device to use.
          seed (float): seed for random number generator
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Default optimizer args
        if not optim_kwargs:
            optim_kwargs = {
                "lr": 0.0001,
                "weight_decay": 0.00001,
            }

        opt_param_setting = {}
        if optimize_hyperparameters and (len(dataset.years) > 1):
            opt_param_setting = self._optimize_hyperparameters(
                dataset,
                param_space,
                optim_kwargs=optim_kwargs,
                device=device,
                **fit_params,
            )
            # replace optimizer args with optimal values
            if "lr" in opt_param_setting:
                optim_kwargs["lr"] = opt_param_setting["lr"]
            if "weight_decay" in opt_param_setting:
                optim_kwargs["weight_decay"] = opt_param_setting["weight_decay"]

        train_losses = self._train_final_model(
            dataset,
            optim_kwargs=optim_kwargs,
            device=device,
            **opt_param_setting,
            **fit_params,
        )

        return self, {"train_losses": train_losses}

    def _optimize_hyperparameters(
        self,
        dataset: Dataset,
        param_space: dict,
        optim_kwargs: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        kfolds: int = 1,
        epochs: int = 10,
        **fit_params,
    ) -> dict:
        """Optimize hyperparameters

        Args:
          dataset (Dataset): training dataset
          param_space (dict): hyperparameters to optimize
          optim_kwargs (dict): arguments to the optimizer
          device (str): the device to use
          kfolds (int): k for k-fold cv (default: 1)
          epochs (int): Number of epochs to train (default: 10)
          **fit_params: Additional parameters.

        Returns:
          A dict of optimal hyperparameter setting
        """
        assert param_space is not None

        opt_loss = float("inf")
        opt_param_setting = {}
        all_years = list(dataset.years)
        random.shuffle(all_years)
        settings = list(ParameterGrid(param_space))
        for i, setting in enumerate(settings):
            if "lr" in setting:
                optim_kwargs["lr"] = setting["lr"]
            if "weight_decay" in setting:
                optim_kwargs["weight_decay"] = setting["weight_decay"]

            val_loss = None
            if kfolds == 1:
                self._logger.debug(f"Running setting {i + 1}/{len(settings)}")
                val_years = all_years[len(all_years) // 2 :]
                train_years = [y for y in all_years if y not in val_years]
                new_model = self.__class__(**self._init_args)
                train_losses, val_losses = new_model._train_and_validate(
                    dataset,
                    train_years,
                    val_years,
                    optim_kwargs=optim_kwargs,
                    epochs=epochs,
                    device=device,
                    **setting,
                    **fit_params,
                )

                val_loss = val_losses[-1]
            else:
                # Split data into k folds
                # For each fold, create new model and datasets, train and record val loss.
                # Finally, average val loss.
                cv_years = [all_years[i::kfolds] for i in range(kfolds)]
                cv_losses = []
                for j, val_years in enumerate(cv_years):
                    self._logger.debug(
                        f"Running inner fold {j + 1}/{kfolds} for hyperparameter setting {i + 1}/{len(settings)}"
                    )
                    train_years = [y for y in all_years if y not in val_years]
                    new_model = self.__class__(**self._init_args)
                    train_losses, val_losses = new_model._train_and_validate(
                        dataset,
                        train_years,
                        val_years,
                        optim_kwargs=optim_kwargs,
                        device=device,
                        epochs=epochs,
                        **setting,
                        **fit_params,
                    )

                    cv_losses.append(val_losses[-1])

                val_loss = np.mean(cv_losses)
            assert val_loss is not None
            assert not np.isnan(val_loss)

            self._logger.debug(
                f"For setting {i + 1}/{len(settings)}, average validation loss: {val_loss}"
            )
            self._logger.debug(f"Settings: {setting}")

            # Store best model setting
            if val_loss < opt_loss:
                opt_loss = val_loss
                opt_param_setting = setting

        return opt_param_setting

    def _train_and_validate(
        self,
        dataset: Dataset,
        train_years: list,
        val_years: list,
        validation_interval: int = 5,
        epochs: int = 10,
        batch_size: int = 16,
        optimizer_fn: callable = torch.optim.Adam,
        optim_kwargs: dict = {},
        loss_fn: callable = torch.nn.functional.mse_loss,
        loss_kwargs: dict = {},
        scheduler_fn: callable = None,
        sched_kwargs: dict = {},
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """
        Fit or train the model and evaluate on validation data.

        Args:
            dataset (Dataset): training dataset
            train_years (list): training years
            val_years (list): validation years
            validation_interval (int): validation frequency (default: 5)
            epochs (int): the number of epochs to train the model (default: 10)
            batch_size (int): the batch size (default: 16)
            optim_fn (callable): the optimizer function (default: Adam)
            optim_kwargs (dict): arguments to the optimizer function
            loss_fn (callable): the loss function (default: mse_loss)
            loss_kwargs (dict): arguments to the loss function
            scheduler_fn (callable): the scheduler function (default: None)
            sched_kwargs (dict): arguments to the scheduler function
            device (str): the device to use
            **kwargs: Additional parameters.

        Returns:
          A tuple training losses, validation losses and maximum epochs to train.
        """
        self.to(device)
        assert epochs > 0

        train_dataset, val_dataset = dataset.split_on_years((train_years, val_years))
        max_season_window_length = train_dataset.max_season_window_length
        train_dataset = TorchDataset(
            train_dataset,
            interpolate_time_series=self._interpolate_time_series,
            aggregate_time_series_to=self._aggregate_time_series_to,
            max_season_window_length=max_season_window_length,
        )
        val_dataset = TorchDataset(
            val_dataset,
            interpolate_time_series=self._interpolate_time_series,
            aggregate_time_series_to=self._aggregate_time_series_to,
            max_season_window_length=max_season_window_length,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=TorchDataset.collate_fn
        )

        # Initialize optimizer and scheduler
        optimizer = optimizer_fn(self.parameters(), **optim_kwargs)
        if scheduler_fn is not None:
            assert sched_kwargs
            scheduler = scheduler_fn(optimizer, **sched_kwargs)
        else:
            scheduler = None

        # Store training set feature means and sds for normalization
        self._norm_params = train_dataset.get_normalization_params(
            normalization="standard"
        )
        train_losses = []
        val_losses = []

        # Training loop
        total_batches = epochs * len(train_loader)  # Total iterations across all epochs
        pbar = tqdm(total=total_batches, desc=f"{self.__class__.__name__}")
        for epoch in range(epochs):
            self.train()
            train_loss = self._train_epoch(
                pbar,
                train_loader,
                device,
                optimizer,
                loss_fn=loss_fn,
                loss_kwargs=loss_kwargs,
                scheduler=scheduler,
            )
            train_losses.append(train_loss)

            if val_loader is not None and (
                (epoch % validation_interval == 0) or (epoch == epochs - 1)
            ):
                with torch.no_grad():
                    self.eval()
                    losses = []
                    for batch in val_loader:
                        batch_preds = self._forward_pass(batch, device)
                        targets = batch[KEY_TARGET]
                        loss = loss_fn(batch_preds, targets, **loss_kwargs)
                        losses.append(loss.item())

                    val_loss = np.mean(losses)
                    val_losses.append(val_loss)

        return train_losses, val_losses

    def _train_final_model(
        self,
        dataset: Dataset,
        epochs: int,
        optimizer_fn: callable = torch.optim.Adam,
        optim_kwargs: dict = {},
        loss_fn: callable = torch.nn.functional.mse_loss,
        loss_kwargs: dict = {"reduction": "mean"},
        scheduler_fn: callable = None,
        sched_kwargs: dict = {},
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
        **kwargs,
    ):
        """
        Fit or train the model on the entire training set.

        Args:
            dataset (Dataset): training dataset,
            epochs (int): number of epochs to train
            optimizer_fn (callable): the optimizer function (default: Adam)
            optim_kwargs (dict): arguments to the optimizer function
            loss_fn (callable): the loss function (default: mse_loss)
            loss_kwargs (dict): arguments to the loss function
            scheduler_fn (callable): the scheduler function (default: None)
            sched_kwargs (dict): arguments to the scheduler function
            device (str): the device to use
            batch_size (int): default is 16
            **kwargs: Additional parameters.

        Returns:
          A list of training losses (one value per epoch).
        """
        self.to(device)

        self._max_season_window_length = dataset.max_season_window_length
        train_dataset = TorchDataset(
            dataset,
            interpolate_time_series=self._interpolate_time_series,
            aggregate_time_series_to=self._aggregate_time_series_to,
            max_season_window_length=self._max_season_window_length,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
        )

        # Initialize optimizer and scheduler
        optimizer = optimizer_fn(self.parameters(), **optim_kwargs)
        if scheduler_fn is not None:
            assert sched_kwargs
            scheduler = scheduler_fn(optimizer, **sched_kwargs)
        else:
            scheduler = None

        # Store training set feature means and sds for normalization
        self._norm_params = train_dataset.get_normalization_params(
            normalization="standard"
        )

        train_losses = []
        total_batches = epochs * len(train_loader)  # Total iterations across all epochs
        pbar = tqdm(total=total_batches, desc=f"{self.__class__.__name__}")
        for epoch in range(epochs):
            self.train()
            train_loss = self._train_epoch(
                pbar,
                train_loader,
                device,
                optimizer,
                loss_fn=loss_fn,
                loss_kwargs=loss_kwargs,
                scheduler=scheduler,
            )
            train_losses.append(train_loss)

        return train_losses

    def _train_epoch(
        self,
        pbar: tqdm,
        dataloader: torch.utils.data.DataLoader,
        device: str,
        optimizer: torch.optim.Optimizer,
        loss_fn: callable = torch.nn.functional.mse_loss,
        loss_kwargs: dict = {"reduction": "mean"},
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ):
        """Run one epoch during training

        Args:
          pbar (tqdm): tqdm progress bar
          dataloader (dataloader): data loader with progress bar
          device (str): the device to use
          optimizer (torch.optim.Optimizer): the optimizer
          loss_fn (callable): the loss function, default mse_loss
          loss_kwargs (dict): the arguments to loss_fn
          scheduler (torch.optim.lr_scheduler.LRScheduler): scheduler for learning rate of optimizer

        Returns:
          The average of all batch losses
        """
        losses = []
        for batch in dataloader:
            # Set gradients to zero
            optimizer.zero_grad()

            batch_preds = self._forward_pass(batch, device)
            targets = batch[KEY_TARGET]
            loss = loss_fn(batch_preds, targets, **loss_kwargs)

            # Backward pass
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            pbar.update(1)
            losses.append(loss.item())

        return np.mean(losses)

    def _forward_pass(self, batch: dict, device: str):
        """A forward pass for batched data.

        Args:
          batch (dict): batched inputs
          device (str): the device to use

        Returns:
          An np.ndarray
        """
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Normalize inputs
        inputs = {k: v for k, v in batch.items() if k != KEY_TARGET}
        inputs = self._normalize_inputs(inputs)
        batch_preds = self(inputs)
        if batch_preds.dim() > 1:
            batch_preds = batch_preds.squeeze(-1)

        return batch_preds

    def _normalize_inputs(self, inputs):
        """Normalize inputs using saved normalization parameters.

        Args:
          inputs (dict): unnormalized inputs

        Returns:
          The same dict after normalizing the entries
        """
        for pred in ALL_PREDICTORS:
            try:
                inputs[pred] = (
                    inputs[pred] - self._norm_params[pred]["mean"]
                ) / self._norm_params[pred]["std"]
            except KeyError:
                raise Exception(f"Unexpected input {pred}")

        return inputs

    def predict_items(
        self,
        X: list,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **predict_params,
    ):
        """Run fitted model on a list of data items.

        Args:
          X (list): a list of data items, each of which is a dict
          device (str): str, the device to use
          **predict_params: Additional parameters

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        self.to(device)
        self.eval()
        if self._aggregate_time_series_to is not None:
            assert self._interpolate_time_series
            assert self._max_season_window_length is not None
            X = TorchDataset.interpolate_and_aggregate(
                X,
                self._max_season_window_length,
                aggregate_time_series_to=self._aggregate_time_series_to,
            )

        with torch.no_grad():
            X_collated = TorchDataset.collate_fn(
                [TorchDataset.cast_to_tensor(x) for x in X]
            )
            y_pred = self._forward_pass(X_collated, device)
            y_pred = y_pred.cpu().numpy()
            return y_pred, {}

    def predict(
        self,
        dataset: Dataset,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
        **predict_params,
    ):
        """Run fitted model on batched data items.

        Args:
          dataset (Dataset): validation dataset
          device (str): the device to use
          **predict_params: Additional parameters

        Returns:
          A tuple containing a np.ndarray and a dict with additional information.
        """
        self.to(device)
        self.eval()
        test_dataset = TorchDataset(
            dataset,
            interpolate_time_series=self._interpolate_time_series,
            aggregate_time_series_to=self._aggregate_time_series_to,
            max_season_window_length=self._max_season_window_length,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, collate_fn=TorchDataset.collate_fn
        )

        with torch.no_grad():
            predictions = None
            for batch in test_loader:
                batch_preds = self._forward_pass(batch, device).cpu().numpy()
                if predictions is None:
                    predictions = batch_preds
                else:
                    predictions = np.concatenate((predictions, batch_preds), axis=0)

            return predictions, {}

    def save(self, model_name):
        """Save model using torch.save.

        Args:
          model_name (str): Filename that will be used to save the model.
        """
        torch.save(self, model_name)

    @classmethod
    def load(cls, model_name):
        """Load model using torch.load.

        Args:
            model_name (str): Filename that was used to save the model.

        Returns:
            The loaded model.
        """
        return torch.load(model_name)


class BaselineLSTM(BaseNNModel):
    """LSTM model.

    Args:
        time_series_have_same_length (bool): whether time series have the same length
        hidden_size (int): The number of features InceptionTime outputs
        num_layers (int): The number of InceptionBlocks. Defaults to 6.
        output_size (int): The number of output classes. Defaults to 1.
        **kwargs: Additional keyword arguments passed to the base class.
    """

    def __init__(
        self,
        time_series_have_same_length=False,
        hidden_size=64,
        num_layers=1,
        output_size=1,
        **kwargs,
    ):
        # Add all arguments to init_args to enable model reconstruction in fit method
        n_ts_inputs = len(TIME_SERIES_PREDICTORS)
        n_static_inputs = len(STATIC_PREDICTORS)
        if not time_series_have_same_length:
            kwargs["interpolate_time_series"] = True
            kwargs["aggregate_time_series_to"] = "dekad"

        kwargs["hidden_size"] = hidden_size
        kwargs["num_layers"] = num_layers
        kwargs["output_size"] = output_size

        super().__init__(**kwargs)
        self._lstm = nn.LSTM(n_ts_inputs, hidden_size, num_layers, batch_first=True)
        self._fc = nn.Linear(hidden_size + n_static_inputs, output_size)

    def fit(
        self,
        dataset: Dataset,
        optimize_hyperparameters: bool = False,
        param_space: dict = {},
        kfolds: int = 1,
        epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        **fit_params,
    ):
        """Fit or train the model.

        Args:
          dataset (Dataset): Training dataset.
          optimize_hyperparameters (bool): Flag to tune hyperparameters.
          param_space (dict): Each entry is a hyperparameter name and list or range of values.
          kfolds (int): k in k-fold cv.
          epochs (int): Number of epochs to train.
          seed (float): seed for random number generator.
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        if ("scheduler_fn" in fit_params) and (fit_params["scheduler_fn"] is not None):
            fit_params["sched_kwargs"] = {"step_size": 2, "gamma": 0.5}

        if optimize_hyperparameters and not param_space:
            param_space = {
                "lr": [0.0001, 0.00001],
                "weight_decay": [0.0001, 0.00001],
            }

        super().fit(
            dataset,
            optimize_hyperparameters=optimize_hyperparameters,
            param_space=param_space,
            kfolds=kfolds,
            epochs=epochs,
            device=device,
            seed=seed,
            **fit_params,
        )

    def forward(self, x):
        x_ts, x_static = separate_ts_static_inputs(x)
        x_ts, _ = self._lstm(x_ts)
        x = torch.cat([x_ts[:, -1, :], x_static], dim=1)
        output = self._fc(x)
        return output


class BaselineInceptionTime(BaseNNModel):
    """InceptionTime model.

    Args:
        time_series_have_same_length (bool): whether time series have the same length
        hidden_size (int): The number of features InceptionTime outputs
        num_layers (int): The number of InceptionBlocks. Defaults to 6.
        num_features (int): The number of features within the InceptionBlocks. Defaults to 32.
        output_size (int): The number of output classes. Defaults to 1.
        **kwargs: Additional keyword arguments passed to the base class.
    """

    def __init__(
        self,
        time_series_have_same_length=False,
        hidden_size=64,
        num_layers=6,
        num_features=32,
        output_size=1,
        **kwargs,
    ):
        # Add all arguments to init_args to enable model reconstruction in fit method
        n_ts_inputs = len(TIME_SERIES_PREDICTORS)
        n_static_inputs = len(STATIC_PREDICTORS)
        if not time_series_have_same_length:
            kwargs["interpolate_time_series"] = True
            kwargs["aggregate_time_series_to"] = "dekad"

        kwargs["hidden_size"] = hidden_size
        kwargs["num_layers"] = num_layers
        kwargs["num_features"] = num_features
        kwargs["output_size"] = output_size

        super().__init__(**kwargs)
        self._timeseries = InceptionTime(
            c_in=n_ts_inputs, c_out=hidden_size, nf=num_features, depth=num_layers
        )
        self._fc = nn.Linear(hidden_size + n_static_inputs, output_size)

    def fit(
        self,
        dataset: Dataset,
        optimize_hyperparameters: bool = False,
        param_space: dict = {},
        kfolds: int = 1,
        epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        **fit_params,
    ):
        """Fit or train the model.

        Args:
          dataset (Dataset): Training dataset.
          optimize_hyperparameters (bool): Flag to tune hyperparameters.
          param_space (dict): Each entry is a hyperparameter name and list or range of values.
          kfolds (int): k in k-fold cv.
          epochs (int): Number of epochs to train.
          seed (float): seed for random number generator.
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        if ("scheduler_fn" in fit_params) and (fit_params["scheduler_fn"] is not None):
            fit_params["sched_kwargs"] = {"step_size": 2, "gamma": 0.5}

        if optimize_hyperparameters and not param_space:
            param_space = {
                "lr": [0.0001, 0.00001],
                "weight_decay": [0.0001, 0.00001],
            }

        super().fit(
            dataset,
            optimize_hyperparameters=optimize_hyperparameters,
            param_space=param_space,
            kfolds=kfolds,
            epochs=epochs,
            device=device,
            seed=seed,
            **fit_params,
        )

    def forward(self, x):
        x_ts, x_static = separate_ts_static_inputs(x)
        x_ts = x_ts.permute(0, 2, 1)
        x_ts = self._timeseries(x_ts)
        x = torch.cat([x_ts, x_static], dim=1)
        output = self._fc(x)
        return output


class BaselineTransformer(BaseNNModel):
    """Transformer model.

    Args:
        seq_len (int): length of time series sequence (in days)
        time_series_have_same_length (bool): whether time series have the same length
        hidden_size (int): The number of resulting timeseries features.
        d_moodel (int): Total dimension of the model.
        n_head (int): Parallel attention heads.
        d_ffn (int): The dimension of the feedforward network model.
        output_size (int): The number of output classes. Defaults to 1.
        num_layers (1): The number of sub-encoder-layers in the encoder.
        **kwargs: Additional keyword arguments passed to the base class.
    """

    def __init__(
        self,
        seq_len,
        time_series_have_same_length=False,
        hidden_size=64,
        d_model=64,
        n_head=1,
        d_ff=256,
        output_size=1,
        num_layers=3,
        **kwargs,
    ):
        # Add all arguments to init_args to enable model reconstruction in fit method
        n_ts_inputs = len(TIME_SERIES_PREDICTORS)
        n_static_inputs = len(STATIC_PREDICTORS)
        if not time_series_have_same_length:
            kwargs["interpolate_time_series"] = True
            kwargs["aggregate_time_series_to"] = "dekad"
            # NOTE: Should match num_time_steps in TorchDataset __getitem__().
            seq_len = int(np.ceil(seq_len / 10))

        kwargs["hidden_size"] = hidden_size
        kwargs["d_model"] = d_model
        kwargs["n_head"] = n_head
        kwargs["d_ff"] = d_ff
        kwargs["num_layers"] = num_layers
        kwargs["output_size"] = output_size
        kwargs["seq_len"] = seq_len
        super().__init__(**kwargs)

        self._timeseries = TST(
            c_in=n_ts_inputs,
            c_out=hidden_size,
            seq_len=seq_len,
            n_layers=num_layers,
            d_model=d_model,
            n_heads=n_head,
            d_ff=d_ff,
        )
        self._fc = nn.Linear(hidden_size + n_static_inputs, output_size)

    def fit(
        self,
        dataset: Dataset,
        optimize_hyperparameters: bool = False,
        param_space: dict = {},
        kfolds: int = 1,
        epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        **fit_params,
    ):
        """Fit or train the model.

        Args:
          dataset (Dataset): Training dataset.
          optimize_hyperparameters (bool): Flag to tune hyperparameters.
          param_space (dict): Each entry is a hyperparameter name and list or range of values.
          kfolds (int): k in k-fold cv.
          epochs (int): Number of epochs to train.
          seed (float): seed for random number generator.
          **fit_params: Additional parameters.

        Returns:
          A tuple containing the fitted model and a dict with additional information.
        """
        if ("scheduler_fn" in fit_params) and (fit_params["scheduler_fn"] is not None):
            fit_params["sched_kwargs"] = {"step_size": 2, "gamma": 0.5}

        if optimize_hyperparameters and not param_space:
            param_space = {
                "lr": [0.0001, 0.00001],
                "weight_decay": [0.0001, 0.00001],
            }

        super().fit(
            dataset,
            optimize_hyperparameters=optimize_hyperparameters,
            param_space=param_space,
            kfolds=kfolds,
            epochs=epochs,
            device=device,
            seed=seed,
            **fit_params,
        )

    def forward(self, x):
        x_ts, x_static = separate_ts_static_inputs(x)
        x_ts = x_ts.permute(0, 2, 1)
        x_ts = self._timeseries(x_ts)
        x = torch.cat([x_ts, x_static], dim=1)
        output = self._fc(x)
        return output


# =============================================================================
# Flat-feature NN models (CNN1D / Transformer for ~100-dim engineered features)
# =============================================================================

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pos_embed[:, :seq_len, :]


class BaseFlatNNModel(BaseModel):
    """Base class for NN models that operate on flat engineered features.

    Handles feature engineering (same as BaseSklearnModel), StandardScaler
    normalization, and a PyTorch training loop.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._feature_cols = None
        self._predesigned_features = False
        if ("feature_cols" in kwargs) and kwargs["feature_cols"] is not None:
            self._feature_cols = kwargs["feature_cols"]
            self._predesigned_features = True

        self._scaler = StandardScaler()
        self._net = None
        self._init_args = kwargs
        self._logger = logging.getLogger(__name__)

    def _design_features(self, crop: str, data_items: Iterable):
        soil_df = data_to_pandas(data_items, data_cols=[KEY_LOC] + SOIL_PROPERTIES)
        soil_df = soil_df.drop_duplicates()
        dfs_x = {"soil": soil_df}
        if LOCATION_PROPERTIES:
            location_df = data_to_pandas(
                data_items, data_cols=[KEY_LOC] + LOCATION_PROPERTIES
            ).drop_duplicates()
            dfs_x["location"] = location_df
        for x, ts_cols in TIME_SERIES_INPUTS.items():
            df_ts = data_to_pandas(
                data_items, data_cols=[KEY_LOC, KEY_YEAR] + [KEY_DATES] + ts_cols
            )
            df_ts = unpack_time_series(df_ts, ts_cols)
            df_ts = df_ts.astype({k: "float" for k in ts_cols})
            df_ts = (
                df_ts.set_index([KEY_LOC, KEY_YEAR, "date"])
                .sort_index()
                .interpolate(method="linear")
            )
            dfs_x[x] = df_ts.reset_index()
        return design_features(crop, dfs_x)

    def _prepare_data(self, dataset):
        if self._predesigned_features:
            if isinstance(dataset, pd.DataFrame):
                df = dataset.copy()
            else:
                df = data_to_pandas(dataset)
        else:
            features = self._design_features(dataset.crop, dataset)
            labels = data_to_pandas(dataset, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET])
            self._feature_cols = [
                c for c in features.columns if c not in [KEY_LOC, KEY_YEAR]
            ]
            df = features.merge(labels, on=[KEY_LOC, KEY_YEAR])
        X = df[self._feature_cols].values.astype(np.float32)
        y = df[KEY_TARGET].values.astype(np.float32)
        return X, y

    def _prepare_predict_data(self, crop: str, data_items):
        if self._predesigned_features:
            if isinstance(data_items, pd.DataFrame):
                df = data_items.copy()
            else:
                df = data_to_pandas(data_items)
        else:
            features = self._design_features(crop, data_items)
            labels = data_to_pandas(
                data_items, data_cols=[KEY_LOC, KEY_YEAR, KEY_TARGET]
            )
            ft_cols = [c for c in features.columns if c not in [KEY_LOC, KEY_YEAR]]
            missing = [c for c in self._feature_cols if c not in ft_cols]
            for c in missing:
                features[c] = 0.0
            features = features[[KEY_LOC, KEY_YEAR] + self._feature_cols]
            df = features.merge(labels, on=[KEY_LOC, KEY_YEAR])
        return df[self._feature_cols].values.astype(np.float32)

    def fit(
        self,
        dataset: Dataset,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        **fit_params,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if self._net is None:
            self._build_net()

        X, y = self._prepare_data(dataset)

        n = len(X)
        n_val = max(1, int(n * 0.2))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        X_tr = self._scaler.fit_transform(X_tr)
        X_val = self._scaler.transform(X_val)

        ds_tr = torch.utils.data.TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr)
        )
        loader = torch.utils.data.DataLoader(
            ds_tr, batch_size=batch_size, shuffle=True
        )

        self._net = self._net.to(device)
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            self._net.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.forward(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            self._net.eval()
            with torch.no_grad():
                xv = torch.from_numpy(X_val).to(device)
                val_pred = self.forward(xv).squeeze(-1).cpu().numpy()
            val_loss = np.mean((val_pred - y_val) ** 2)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self._net.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs > patience:
                self._logger.debug(f"Early stopped at epoch {epoch}")
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self, {}

    def _predict_flat(self, X: np.ndarray, device: str):
        self._net.eval()
        X_scaled = self._scaler.transform(X)
        with torch.no_grad():
            xt = torch.from_numpy(X_scaled).to(device)
            preds = self.forward(xt).squeeze(-1).cpu().numpy()
        return preds

    def predict(self, dataset, device: str = "cuda" if torch.cuda.is_available() else "cpu", **predict_params):
        crop = dataset.crop if hasattr(dataset, 'crop') else None
        X = self._prepare_predict_data(crop, dataset)
        return self._predict_flat(X, device), {}

    def predict_items(self, X: list, crop=None, device: str = "cuda" if torch.cuda.is_available() else "cpu", **predict_params):
        assert crop is not None
        X_arr = self._prepare_predict_data(crop, X)
        return self._predict_flat(X_arr, device), {}

    def save(self, model_name):
        with open(model_name, "wb") as f:
            pickle.dump(self, f)

    def load(cls, model_name):
        with open(model_name, "rb") as f:
            return pickle.load(f)


class CNN1DRegressor(BaseFlatNNModel):
    """1D CNN regressor for flat engineered features (~100 dims)."""

    def __init__(self, feature_cols: list = None, hidden_dim: int = 128, dropout: float = 0.1, **kwargs):
        kwargs["feature_cols"] = feature_cols
        super().__init__(**kwargs)
        self._hidden_dim = hidden_dim
        self._dropout = dropout
        self._build_net()

    def _build_net(self):
        hd, dp = self._hidden_dim, self._dropout
        self._net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(128 * 4, hd),
            nn.ReLU(),
            nn.Dropout(dp),
            nn.Linear(hd, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self._net(x)


class _TransformerNet(nn.Module):
    """Internal nn.Module that holds all Transformer sub-components."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos = _PositionalEncoding(d_model, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


class TransformerRegressor(BaseFlatNNModel):
    """Transformer regressor for flat engineered features (~100 dims)."""

    def __init__(
        self,
        feature_cols: list = None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ):
        kwargs["feature_cols"] = feature_cols
        super().__init__(**kwargs)
        self._d_model = d_model
        self._nhead = nhead
        self._num_layers = num_layers
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._build_net()

    def _build_net(self):
        self._net = _TransformerNet(
            self._d_model, self._nhead, self._num_layers,
            self._dim_feedforward, self._dropout,
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self._net(x)
