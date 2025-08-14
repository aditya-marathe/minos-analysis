"""\
models / helpers / torch_history.py
--------------------------------------------------------------------------------

Author - Aditya Marathe
Email  - aditya.marathe.20@ucl.ac.uk

--------------------------------------------------------------------------------

My version of the Keras History class for PyTorch.
"""

from __future__ import annotations

__all__ = ["TorchHistory"]

from collections import defaultdict
import time


class TorchHistory:
    """\
    TorchHistory
    ------------

    A simple class to store the training history of a PyTorch model.
    """

    def __init__(self):
        self._history: dict[str, list[float]] = defaultdict(list)

        self._start_time: float | None = None
        self._end_time: float | None = None

    def start_timer(self) -> None:
        """\
        Start the timer.
        """
        self._start_time = time.time()

    def end_timer(self) -> None:
        """\
        End the timer.
        """
        self._end_time = time.time()

    def append(self, epoch: int, **other_stuff: float):
        """\
        Append a new entry to the history.
        
        Parameters
        ----------
        epoch: int
            The current epoch number.

        **other_stuff: float
            Any other metrics to log.
        """
        self._history["Epochs"].append(epoch)

        for key, value in other_stuff.items():
            self._history[key].append(value)

    @property
    def history(self) -> dict[str, list[float]]:
        """\
        The training history.
        """
        return dict(self._history).copy()

    @property
    def duration(self) -> float | None:
        """\
        The duration of the training in seconds.
        """
        if (self._start_time is not None) and (self._end_time is not None):
            return self._end_time - self._start_time

        return None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._history})"

    def __repr__(self) -> str:
        return str(self)
