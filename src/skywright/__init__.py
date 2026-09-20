"""Deploy PyTorch training jobs with fewer moving parts."""

from skywright.accelerator import Accelerator, inspect_accelerator
from skywright.datasets import Dataset
from skywright.events import (
    DatasetOpened,
    DatasetOpenFailed,
    DatasetOpenStarted,
    EventListeners,
    RunOutcome,
    Start,
    Stop,
)
from skywright.training import (
    RunContext,
    Training,
    TrainingState,
    run,
)

__all__ = [
    "Accelerator",
    "Dataset",
    "DatasetOpenFailed",
    "DatasetOpenStarted",
    "DatasetOpened",
    "EventListeners",
    "RunContext",
    "RunOutcome",
    "Start",
    "Stop",
    "Training",
    "TrainingState",
    "inspect_accelerator",
    "run",
]
__version__ = "0.1.0"
