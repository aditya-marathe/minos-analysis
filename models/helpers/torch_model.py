"""\
models / helpers / torch_model.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

Wraps on `torch.nn.Module` to make Torch models behave like Keras models.
"""

from __future__ import annotations

__all__ = ["TorchModel"]

from typing import Final, Literal

from abc import ABC
from pathlib import Path

import torch
import torchmetrics as tm
from torchsummary import summary

from .torch_history import TorchHistory
from .torch_callbacks import TorchCallback
from .torch_progress import TrainingProgressBar

LOSS_LABEL: Final[str] = "Loss"
VALIDATION_PREFIX: Final[str] = "Val"


# =========================== [ Helper Functions ] =========================== #


def _validate_fit_inputs(
    optimiser: torch.optim.Optimizer | None,
    loss: torch.nn.Module | None,
    train_data: torch.utils.data.DataLoader,
    batch_size: int = 32,
    epochs: int = 1,
    class_weight: dict[int, float] | None = None,
    sample_weight: torch.Tensor | None = None,
    initial_epoch: int = 1,
    callbacks: list[TorchCallback] | None = None,
) -> None:
    # (1) Stuff that is not implemented yet!
    if sample_weight or class_weight:
        raise NotImplementedError("Work in progress for sample/class weights.")

    # (2) Is the model compiled?
    if not (optimiser and loss):
        raise ValueError("Model must be compiled before training!")

    # (3) General checks
    if batch_size <= 0:
        raise ValueError(
            f"Invalid batch size {batch_size}. Batch size must be in [1, +Inf)."
        )

    if batch_size != train_data.batch_size:
        # Note: Why don't I just get the batch size from the DataLoader?
        #       I guess, I just want to make sure that I know exactly which
        #       batch size is being used. There could be a mismatch when
        #       doing my experiments, so it would be good to be careful.

        raise ValueError(
            f"Batch size mismatch: {batch_size} != {train_data.batch_size}!"
        )

    if initial_epoch >= epochs or initial_epoch < 0:
        raise ValueError(
            f"Invalid value for the initial epoch: {initial_epoch}. "
            f"The initial epoch must be in [0, {epochs})!"
        )


def _separate_callbacks(
    callbacks: list[TorchCallback],
) -> tuple[list[TorchCallback], list[TorchCallback]]:
    """\
    [ Internal ] Seperate callbacks checked after each batch or epoch.

    Parameters
    ----------
    callbacks : list[TorchCallback]
        The list of callbacks to separate.

    Returns
    -------
    tuple[list[TorchCallback], list[TorchCallback]]
        A tuple containing the lists of callbacks to be called after each epoch 
        and after each batch.
    """
    after_epoch = []
    after_batch = []

    for callback in callbacks:
        if callback.after_batch:
            after_batch.append(callback)
        if callback.after_epoch:
            after_epoch.append(callback)

    return after_epoch, after_batch


def _move_data_to_device(
    inputs: torch.Tensor | list[torch.Tensor],
    labels: torch.Tensor | None,
    device: str,
) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | None]:
    """\
    [ Internal ] Moves the input and label tensors to the specified device.

    Parameters
    ----------
    inputs : torch.Tensor | list[torch.Tensor]
        The input tensors to move.

    labels : torch.Tensor | None
        The label tensors to move.

    device : str
        The device to move the tensors to (e.g., "cpu" or "cuda").

    Returns
    -------
    tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]
        The moved input and label tensors.
    """
    if isinstance(inputs, list):
        inputs = [input_tensor.to(device=device) for input_tensor in inputs]
    else:
        inputs = inputs.to(device=device)

    return inputs, labels.to(device=device) if labels is not None else None


def _reset_metrics(metrics: list[tm.Metric]) -> None:
    """\
    [ Internal ] Resets the metrics for the current batch.

    Parameters
    ----------
    metrics : list[tm.Metric]
        The list of metrics to reset.
    """
    for metric in metrics:
        metric.reset()


def _compute_metrics(metrics: list[tm.Metric]) -> dict[str, float]:
    """\
    [ Internal ] Computes the metrics for the current batch.

    Parameters
    ----------
    metrics : list[tm.Metric]
        The list of metrics to compute.

    Returns
    -------
    dict[str, float]
        A dictionary containing the computed metric names and their values.
    """
    return {
        metric.__class__.__name__: float(metric.compute()) for metric in metrics
    }


def _update_compute_metrics(
    metrics: list[tm.Metric],
    batch_labels: torch.Tensor,
    batch_output: torch.Tensor,
    activation: torch.nn.Module | None,
) -> dict[str, float]:
    """\
    [ Internal ] Calculates the metrics for the current batch.

    Parameters
    ----------
    metrics: list[tm.Metric]
        The list of metrics to calculate.

    batch_labels: torch.Tensor
        The label/truth data for the batch.

    batch_output: torch.Tensor
        The model's output predictions for the batch.

    activation: torch.nn.Module | None
        The activation function to use for the model's output.

    Returns
    -------
    dict[str, float]
        A dictionary containing the total metric names and their values.
    """
    for metric in metrics:
        if activation is not None:
            batch_output = activation(batch_output)

        metric.update(batch_output, batch_labels)

    return _compute_metrics(metrics=metrics)


def _train_batch(
    model: TorchModel,
    optimiser: torch.optim.Optimizer,
    loss: torch.nn.Module,
    batch_inputs: torch.Tensor | list[torch.Tensor],
    batch_labels: torch.Tensor,
    class_weight: dict[int, float] | None = None,
    sample_weight: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor]:
    """\
    [ Internal ] Train the model on a single batch of data.

    Parameters
    ----------
    batch_inputs : torch.Tensor | list[torch.Tensor]
        The input data for the batch.

    batch_labels : torch.Tensor
        The labels for the batch.

    class_weight : dict[int, float] | None
        Class weights for the loss function. Defaults to None.

    sample_weight : torch.Tensor | None
        Sample weights for the batch. Defaults to None.

    Returns
    -------
    tuple[float, torch.Tensor]
        The loss value for the batch and the model output.
    """
    optimiser.zero_grad()

    if not isinstance(batch_inputs, list):
        batch_inputs = [batch_inputs]

    batch_output: torch.Tensor = model(*batch_inputs)
    loss_value: torch.Tensor = loss(batch_output, batch_labels)

    loss_value.backward()
    optimiser.step()

    return float(loss_value.item()), batch_output.detach()


def _train_epoch(
    model: TorchModel,
    activation: torch.nn.Module | None,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    batch_size: int,
    loss: torch.nn.Module,
    metrics: list[tm.Metric],
    train_data: torch.utils.data.DataLoader,
    class_weight: dict[int, float] | None,
    sample_weight: torch.Tensor | None,
    callbacks: list[TorchCallback],
    progressbar: TrainingProgressBar,
    device: str,
) -> dict[str, float]:
    """\
    [ Internal ] Train the model for one epoch.
    
    Parameters
    ----------
    model : TorchModel
        The model to train.

    activation: torch.nn.Module | None
        The activation function to use for the model's output.

    optimiser : torch.optim.Optimizer
        The optimizer to use for training.

    epoch: int
        The current epoch number.

    batch_size: int
        The batch size to use for training.

    loss : torch.nn.Module
        The loss function to use for training.

    metrics : list[tm.Metric]
        The list of metrics to use for training.

    train_data : torch.utils.data.DataLoader
        The training data loader.

    callbacks: list[TorchCallback]
        The list of callbacks to use during training.

    progressbar : TrainingProgressBar
        The progress bar to use for training.

    class_weight : dict[int, float] | None
        Class weights for the loss function. Defaults to None.

    sample_weight : torch.Tensor | None
        Sample weights for the batch. Defaults to None.

    device : str
        The device to use for training (e.g., "cpu" or "cuda").

    Returns
    -------
    dict[str, float]
        A dictionary containing the training metrics for the epoch.
    """
    model.train()

    num_batches = len(train_data) * batch_size

    _reset_metrics(metrics=metrics)

    # (1) Batch loss.
    running_loss = 0.0

    for i, (batch_inputs, batch_labels) in enumerate(train_data):
        # (2) Move the data to the selected device.
        batch_inputs, batch_labels = _move_data_to_device(
            inputs=batch_inputs, labels=batch_labels, device=device
        )

        if batch_labels is None:
            raise ValueError("Un-reachable: No batch labels were provided!")

        # (3) Train the model on the current batch.
        current_loss, batch_output = _train_batch(
            model,
            optimiser,
            loss,
            batch_inputs,
            batch_labels,
            class_weight=class_weight,
            sample_weight=sample_weight,
        )

        # (4) Calculate this batch's loss and metrics.
        running_loss += current_loss
        _update_compute_metrics(
            metrics=metrics,
            batch_labels=batch_labels,
            batch_output=batch_output,
            activation=activation,
        )

        # (5) Update with the loss and metrics averaged over the batch size.
        this_batch_info = {LOSS_LABEL: current_loss / batch_size}
        this_batch_info.update(_compute_metrics(metrics=metrics))
        progressbar.update_progress(info=this_batch_info)

        # (6) Callbacks
        for callback in callbacks:
            callback(
                model=model,
                epoch=epoch,
                info=this_batch_info,
            )

    # (7) Return the averaged metrics for the epoch.
    return {
        LOSS_LABEL: running_loss / num_batches,
        **_compute_metrics(metrics=metrics),
    }


def _validate_epoch(
    model: TorchModel,
    activation: torch.nn.Module | None,
    loss: torch.nn.Module,
    batch_size: int,
    validation_data: torch.utils.data.DataLoader,
    metrics: list[tm.Metric],
    device: str,
) -> dict[str, float]:
    """\
    [ Internal ] Validate the model on the validation data.

    Parameters
    ----------
    model : TorchModel
        The model to validate.

    activation: torch.nn.Module | None
        The activation function to use for the model's output.

    loss : torch.nn.Module
        The loss function to use for validation.

    batch_size: int
        The batch size used for the validation dataset.

    validation_data : torch.utils.data.DataLoader
        The validation data loader.

    metrics : list[tm.Metric]
        The list of metrics to use for validation.

    device : str
        The device to use for validation (e.g., "cpu" or "cuda").

    Returns
    -------
    dict[str, float]
        The validation metrics, including the loss.
    """
    model.eval()

    _reset_metrics(metrics=metrics)

    num_batches = len(validation_data) * batch_size

    # (1) Set up a counter for the total metrics.
    running_loss = 0.0

    for batch_inputs, batch_labels in validation_data:
        # (2) Move the data to the selected device.
        batch_inputs, batch_labels = _move_data_to_device(
            inputs=batch_inputs, labels=batch_labels, device=device
        )

        if batch_labels is None:
            raise ValueError("Un-reachable: No batch labels were provided!")

        # (3) Forward pass and compute loss.
        if not isinstance(batch_inputs, list):
            batch_inputs = [batch_inputs]

        with torch.no_grad():
            batch_output = model(*batch_inputs)
            batch_loss: torch.Tensor = loss(batch_output, batch_labels)

            # (4) Update the running loss and the metrics.
            running_loss += float(batch_loss.item())
            _ = _update_compute_metrics(
                metrics=metrics,
                batch_labels=batch_labels,
                batch_output=batch_output,
                activation=activation,
            )

    this_epoch_info = {}
    this_epoch_info[LOSS_LABEL] = running_loss / num_batches
    this_epoch_info.update(_compute_metrics(metrics=metrics))

    return {
        VALIDATION_PREFIX + key: value for key, value in this_epoch_info.items()
    }


def _evaluate_model(
    model: TorchModel,
    test_data: torch.utils.data.DataLoader,
    output_activation: torch.nn.Module | None = None,
) -> dict[str, float]:
    raise NotImplementedError()


# ================================ [ Model  ] ================================ #


class TorchModel(torch.nn.Module, ABC):
    """\
    TorchModel
    ----------

    A simple wrapper around `torch.nn.Module` to make Torch models behave like 
    Keras models.
    """

    def __init__(self) -> None:
        super().__init__()

        self._optimiser: torch.optim.Optimizer | None = None
        self._loss: torch.nn.Module | None = None
        self._metrics: list[tm.Metric] = []

        # Note: I want to keep another copy of the history, so that even if the
        #       training is interrupted, I can still access the history of the
        #       partially trained model.
        self._previous_run_history: TorchHistory | None = None

        self._output_activation: torch.nn.Module | None = None

        self._device: str = "cpu"

        self._is_training: bool = False

    def jit_compile(self) -> None:
        """\
        Compiles the model for JIT (Just-In-Time) compilation.
        """
        torch.nn.Module.compile(self)

    def set_device(self, device: Literal["cpu", "cuda"]) -> None:
        """\
        Sets the device for the model.
        """
        self._device = device

    def compile(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        metrics: list[tm.Metric] | None = None,
    ) -> None:
        """\
        Compiles the model.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimiser to use for training the model.

        loss : torch.nn.Module
            The loss function to use for training the model.

        metrics : list[tm.Metric] | None
            A list of metrics to evaluate the model during training. Defaults 
            to None.

        Notes
        -----
        Overrides the `torch.nn.Module.compile` method. The same method is now
        renamed to `jit_compile`.

        I chose the American spelling for "optimiser" to make this `fit` method
        similar to Keras' `model.fit`.
        """
        # (1) Set the optimizer, loss, and metrics.
        self._optimiser = optimizer
        self._loss = loss
        self._metrics = metrics or []

        # (2) Set the device.
        self.to(device=self._device)

        for metric in self._metrics:
            metric.to(device=self._device)

    def _stop_fitting(self) -> None:
        """\
        [ Internal ] Stops the fitting mainloop.
        """
        self._is_training = False

    def fit(
        self,
        train_data: torch.utils.data.DataLoader,
        batch_size: int,
        epochs: int = 1,
        validation_data: torch.utils.data.DataLoader | None = None,
        class_weight: dict[int, float] | None = None,
        sample_weight: torch.Tensor | None = None,
        initial_epoch: int = 1,
        callbacks: list[TorchCallback] | None = None,
        verbose: bool = True,
    ) -> TorchHistory:
        """\
        Fits the model to the training data.

        Parameters
        ----------
        train_data : torch.utils.data.DataLoader
            The training data loader.

        batch_size : int
            The batch size to use for training.

        epochs : int
            The number of epochs to train the model for. Defaults to 1.

        validation_data : torch.utils.data.DataLoader | None
            The validation data loader. Defaults to None.

        class_weight : dict[int, float] | None
            Class weights for the loss function. Defaults to None.

        sample_weight : torch.Tensor | None
            Sample weights for the batch. Defaults to None.

        initial_epoch : int
            The initial epoch to start training from. Defaults to 1.

        callbacks : list[Callable] | None
            A list of callbacks to call during training. Defaults to None.

        verbose : bool
            Whether to enable verbose output during training. Defaults to True.

        Returns
        -------
        TorchHistory
            The history of the training run, including loss and metrics.
        """
        # (1) Validate inputs and setup callbacks.
        _validate_fit_inputs(
            train_data=train_data,
            optimiser=self._optimiser,
            loss=self._loss,
            batch_size=batch_size,
            initial_epoch=initial_epoch,
            epochs=epochs,
        )

        after_epoch_callbacks, after_batch_callbacks = _separate_callbacks(
            callbacks=(callbacks or [])
        )

        # (2) Training loop.
        history = TorchHistory()
        history.start_timer()

        self._is_training = True

        # (2.2) Epoch Loop
        for epoch in range(initial_epoch, epochs + 1):
            if not self._is_training:
                break

            this_epoch_bar = TrainingProgressBar(
                epoch=epoch,
                total_epochs=epochs,
                total_batches=len(train_data),
                enable_verbose=verbose,
            )

            # (2.3) Batch Loop
            this_epoch_info = _train_epoch(
                model=self,
                activation=self._output_activation,
                optimiser=self._optimiser,  # pyright: ignore[reportArgumentType]
                loss=self._loss,  # pyright: ignore[reportArgumentType]
                epoch=epoch,
                batch_size=batch_size,
                metrics=self._metrics,
                train_data=train_data,
                class_weight=class_weight,
                sample_weight=sample_weight,
                callbacks=after_batch_callbacks,
                progressbar=this_epoch_bar,
                device=self._device,
            )

            # (2.4) Validation Loop
            this_epoch_val = {}

            if validation_data is not None:
                this_epoch_val = _validate_epoch(
                    model=self,
                    activation=self._output_activation,
                    loss=self._loss,  # pyright: ignore[reportArgumentType]
                    batch_size=validation_data.batch_size,  # pyright: ignore[reportArgumentType]
                    validation_data=validation_data,
                    metrics=self._metrics,
                    device=self._device,
                )

            # (2.5) Merge dicts
            this_epoch_info = {**this_epoch_info, **this_epoch_val}

            # (2.6) Append to history and update progress bar.
            this_epoch_bar.update(info=this_epoch_info)
            this_epoch_bar.end()

            history.append(epoch=epoch, **this_epoch_info)
            self._previous_run_history = history

            # (2.7) Callbacks
            for callback in after_epoch_callbacks:
                callback(
                    model=self,
                    epoch=epoch,
                    info=this_epoch_info,
                )

        self.eval()
        history.end_timer()

        return history

    def evaluate(
        self,
        test_data: torch.utils.data.DataLoader,
        batch_size: int,
        verbose: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def predict(
        self,
        data: torch.utils.data.DataLoader,
        batch_size: int,
        verbose: bool = True,
    ) -> torch.Tensor:
        """\
        Predicts the output for the given data.

        Parameters
        ----------
        data : torch.utils.data.DataLoader
            The data loader containing the input data.

        batch_size : int
            The batch size to use for predictions.

        verbose : bool
            Whether to print progress messages.

        Returns
        -------
        torch.Tensor
            The predicted output for the input data.
        """
        raise NotImplementedError()

    def summary(self) -> None:
        """\
        Prints a summary of the PyTorch model.
        """
        summary(self, depth=3, verbose=1)

    def get_total_parameters(self) -> int:
        """\
        Gets the total number of parameters in the model.

        Returns
        -------
        int
            The total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    def get_total_trainable_parameters(self) -> int:
        """\
        Gets the total number of trainable parameters in the model.

        Returns
        -------
        int
            The total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path) -> None:
        """\
        Saves the model to the specified path.

        Parameters
        ----------
        path : str | Path
            The path to save the model to.
        """
        path = Path(path).expanduser().resolve()

        if path.is_dir():
            path /= f"{self.__class__.__name__.lower()}.pt"

        if (path.suffix != ".pt") or (path.suffix != ".pth"):
            # Note: Not that it matters but it is good to stick to the
            #       conventions.
            raise ValueError(
                f'Invalid file extension "{path.suffix}". '
                'The model must be saved with a ".pt" or ".pth" extension!'
            )

        torch.save(self.state_dict(), path)

    @property
    def optimiser(self) -> torch.optim.Optimizer:
        """\
        The optimiser used by the model.
        """
        if not self._optimiser:
            raise ValueError("Model is not compiled yet!")

        return self._optimiser

    @property
    def loss(self) -> torch.nn.Module:
        """\
        The loss function used by the model.
        """
        if not self._loss:
            raise ValueError("Model is not compiled yet!")

        return self._loss

    @property
    def last_training_history(self) -> TorchHistory | None:
        """\
        The history of the last training run.
        """
        return self._previous_run_history

    def __str__(self) -> str:
        is_compiled = False

        if (self._optimiser is not None) or (self._loss is not None):
            is_compiled = True

        return f"{self.__class__.__name__}(is_compiled={is_compiled})"

    def __repr__(self) -> str:
        return str(self)
