"""\
models / helpers / torch_callbacks.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

"""

from __future__ import annotations

__all__ = []

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

from copy import deepcopy

import torch

if TYPE_CHECKING:
    from models.helpers.torch_model import TorchModel


# ================================= [ Base ] ================================= #


class TorchCallback(ABC):
    def __init__(
        self,
        after_batch: bool = False,
        after_epoch: bool = False,
        print_info: str = "",
        verbose: bool = True,
    ) -> None:
        """\
        Initialises the callback.

        Parameters
        ----------
        after_batch: bool
            Whether to run the callback after each batch. Defaults to False.
        
        after_epoch: bool
            Whether to run the callback after each epoch. Defaults to False.
        
        print_info: str
            The information to print when the callback is triggered. Defaults to
            `""`.

        verbose: bool
            Whether to enable verbose output. Defaults to True.
        """
        self._run_after_batch = bool(after_batch)
        self._run_after_epoch = bool(after_epoch)
        self._print_info = print_info
        self._verbose = bool(verbose)

    def _print(self, **print_format) -> None:
        """\
        Prints the current state of the callback.

        Parameters
        ----------
        **print_format: Any
            The values to format the print information with.
        """
        if self._verbose and self._print_info:
            print(
                f"[ {self.__class__.__name__} ] "
                f"{self._print_info.format(**print_format)}"
            )

    def is_triggered(
        self,
        model: TorchModel,
        epoch: int,
        info: dict[str, float],
    ) -> bool:
        """\
        Checks if the callback should be triggered.

        Parameters
        ----------
        model: TorchModel
            The model being trained.

        epoch: int
            The current epoch number.
        
        info: dict[str, float]
            The loss and metrics information for the batch or epoch.

        Returns
        -------
        bool
            Whether the callback should be triggered.

        Note
        ----
        This method should ideally be overriden by subclasses.
        """
        return True  # By default, always trigger the callback.

    @abstractmethod
    def run(
        self,
        model: TorchModel,
        epoch: int,
        info: dict[str, float],
    ) -> None:
        """\
        Runs the callback if the condition is met.

        Parameters
        ----------
        model: TorchModel
            The model being trained.

        epoch: int
            The current epoch number.
                
        info: dict[str, float]
            The loss and metrics information for the batch or epoch.
        """
        pass

    def __call__(
        self,
        model: TorchModel,
        epoch: int,
        info: dict[str, float],
    ) -> None:
        """\
        Calls the callback if the condition is met.

        Parameters
        ----------
        model: TorchModel
            The model being trained.

        epoch: int
            The current epoch number.
                
        info: dict[str, float]
            The loss and metrics information for the batch or epoch.
        """
        if self.is_triggered(
            model=model,
            epoch=epoch,
            info=info,
        ):
            self.run(
                model=model,
                epoch=epoch,
                info=info,
            )

    @property
    def after_batch(self) -> bool:
        """\
        Whether to run the callback after each batch.
        """
        return self._run_after_batch

    @property
    def after_epoch(self) -> bool:
        """\
        Whether to run the callback after each epoch.
        """
        return self._run_after_epoch


# ============================ [ Early Stopping ] ============================ #


class EarlyStopping(TorchCallback):
    def __init__(
        self,
        monitor: str = "ValLoss",
        min_delta: float = 0.1,
        patience: int = 0,
        restore_best_weights: bool = True,
        start_from_epoch: int = 1,
        verbose: bool = True,
    ) -> None:
        """\
        Initialises the early stopping callback.

        Parameters
        ----------
        monitor: str
            The metric to monitor for early stopping. Defaults to "ValLoss".

        min_delta: float
            The minimum change in the monitored quantity to qualify as an
            improvement. Defaults to 0.1.

        patience: int
            Epochs to wait until training is stopped. Defaults to 1.

        restore_best_weights: bool
            Whether to restore the model weights if early stopping is triggered.
            Defaults to True.

        start_from_epoch: int
            Epoch from which to start monitoring.

        verbose: bool
            Whether to enable verbose output. Defaults to True.

        Notes
        -----
        It is important to know that in this implementation, epochs start from 
        one and NOT from zero!
        """
        super().__init__(
            after_epoch=True,
            print_info=("Training stopped after {epochs} epochs."),
            verbose=verbose,
        )

        self._monitor = monitor
        self._min_delta = min_delta
        self._patience = patience
        self._restore = restore_best_weights
        self._start_epoch = start_from_epoch

        self._is_first_check = True
        self._patience_count = 0
        self._previous_value = 0.0
        self._best_weights: dict[str, torch.Tensor] | None = None

    def _save_best_weights(self, model: TorchModel) -> None:
        """\
        [ Internal ] Saves the best weights of the model.

        Parameters
        ----------
        model: TorchModel
            The model being trained.
        """
        self._best_weights = deepcopy(model.state_dict())

    def is_triggered(
        self,
        model: TorchModel,
        epoch: int,
        info: dict[str, float],
    ) -> bool:
        """\
        Checks if the monitor value has stopped improving.

        Parameters
        ----------
        model: TorchModel
            The model being trained.

        epoch: int
            The current epoch number.

        info: dict[str, float]
            The loss and metrics information for the batch or epoch.
        """
        # (1) Is the variable in the info dict?
        if self._monitor not in info:
            raise ValueError(
                self.__class__.__name__
                + f' - Monitor variable "{self._monitor}" not found!'
            )

        # (2) Are we at the starting epoch?
        if epoch < self._start_epoch:
            self._save_best_weights(model=model)
            return False  # Don't check at the starting epoch!

        # (3) Is this the first check?
        if self._is_first_check:
            self._is_first_check = False
            self._previous_value = info[self._monitor]
            return False  # Initialise the "previous value".

        # (4) Calculate the percentage change.
        percent_change = (
            abs(self._previous_value - info[self._monitor])
            / self._previous_value
        )

        # (5) Is the change more than the minimum delta?
        if percent_change >= self._min_delta:
            self._patience_count = 0
            self._previous_value = info[self._monitor]
            if self._restore:
                self._save_best_weights(model=model)
            return False  # Monitor value has changed enough, reset patience.

        self._patience_count += 1  # Monitor value did not change enough.

        # (6) Has the patience been exceeded?
        if self._patience_count >= self._patience:
            return True  # Triggered.

        # (7) Update previous value and best epoch.
        self._previous_value = info[self._monitor]

        return False

    def run(
        self,
        model: TorchModel,
        epoch: int,
        info: dict[str, float],
    ) -> None:
        """\
        Stops the training process.

        Parameters
        ----------
        model: TorchModel
            The model being trained.

        epoch: int
            The current epoch number.

        info: dict[str, float]
            The loss and metrics information for the batch or epoch.
        """
        # (1) Restore best weights, if enabled.
        if self._restore and (self._best_weights is not None):
            model.load_state_dict(self._best_weights)

        # (2) End training.
        model._stop_fitting()

        # (3) Verbosity.
        self._print(epochs=epoch)
