"""
models / helpers / torch_progress.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

"""

from __future__ import annotations

__all__ = ["TrainingProgressBar"]

from typing import Final

from tqdm import tqdm


TQDM_FORMAT: Final[str] = (
    "Epoch {{desc:>{max_desc_width}}} - {{n_fmt:>{max_n_width}}}/"
    "{{total_fmt:<{max_n_width}}} |{{bar}}| {{rate_fmt:>15}} - ETA "
    "{{remaining:>6}} {{postfix}}"
)
EPOCH_INFO: Final[str] = "{current_epoch}/{total_epochs}"


# =========================== [ Helper Functions ] =========================== #


def _format_postfix(info: dict[str, float]) -> str:
    """\
    [ Internal ] Formats the postfix for the progress bar.

    Parameters
    ----------
    info : dict[str, float]
        Information about the current batch: loss and metrics.

    Returns
    -------
    str
        The formatted postfix string.
    """
    return " - ".join(f"{key}: {value:.4f}" for key, value in info.items())


# ============================== [ Verbosity  ] ============================== #


class TrainingProgressBar:
    """\
    TrainingProgressBar
    -------------------

    A simple class to handle the verbosity of the training process.
    """

    def __init__(
        self,
        epoch: int,
        total_epochs: int,
        total_batches: int,
        enable_verbose: bool,
        leave: bool = True,
    ) -> None:
        """\
        Initialises the verbosity for the training process.
        
        Parameters
        ----------
        epoch : int
            The current epoch number.

        total_epochs : int
            The total number of epochs.

        total_batches : int
            The total number of training steps (batches) to complete.

        enable_verbose : bool
            Whether to enable verbose output during training.

        leave : bool
            Whether to leave the progress bar after training is complete.
            Defaults to False.
        """
        self._bar = tqdm(
            desc=EPOCH_INFO.format(
                current_epoch=epoch, total_epochs=total_epochs
            ),
            total=total_batches,
            bar_format=TQDM_FORMAT.format(
                max_desc_width=(2 * len(str(total_epochs)) + 1),
                max_n_width=len(str(total_batches)),
            ),
            leave=leave,
            unit="batch",
            disable=(not enable_verbose),
        )

    def update(self, info: dict[str, float]) -> None:
        """\
        Updates the progress bar with the current batch information.
        
        Parameters
        ----------
        info : dict[str, float] | None
            Information about the current batch: loss and metrics.
            Defaults to None.
        """
        self._bar.set_postfix_str(s=_format_postfix(info), refresh=True)

    def update_progress(self, info: dict[str, float]) -> None:
        """\
        Updates the progress bar with the information for the current batch.
        
        Parameters
        ----------
        info : dict[str, float]
            Information about the current batch: loss and metrics.
        """
        self._bar.update()
        self.update(info=info)

    def end(self) -> None:
        """\
        Ends the progress bar.
        """
        self._bar.close()
