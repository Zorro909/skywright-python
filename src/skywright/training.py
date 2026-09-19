"""Skywright-owned training lifecycle."""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from types import FrameType
from typing import Never

from skywright.accelerator import Accelerator, inspect_accelerator
from skywright.events import EventListeners, RunOutcome, Start, Stop


@dataclass(frozen=True, slots=True)
class RunContext:
    """Runtime information supplied to project setup."""

    argv: tuple[str, ...]
    accelerator: Accelerator


@dataclass(frozen=True, slots=True)
class Training[Batch]:
    """Project training behavior executed by Skywright."""

    epochs: int
    batches: Callable[[int], Iterable[Batch]]
    step: Callable[[Batch], None]
    listeners: EventListeners = field(default_factory=EventListeners)


type Setup[Batch] = Callable[[RunContext], Training[Batch]]


def run[Batch](setup: Setup[Batch], argv: Sequence[str] = ()) -> int:
    """Set up and execute a training run."""
    if not callable(setup):
        raise TypeError("setup must be callable")

    arguments = tuple(argv)
    if not all(isinstance(argument, str) for argument in arguments):
        raise TypeError("every argv item must be a string")

    context = RunContext(argv=arguments, accelerator=inspect_accelerator())
    training = setup(context)
    _validate_training(training)

    previous_sigterm_handler = signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        try:
            training.listeners._dispatch(Start(context=context))
            for epoch in range(training.epochs):
                for batch in training.batches(epoch):
                    training.step(batch)
        except BaseException as error:
            outcome = (
                RunOutcome.INTERRUPTED
                if isinstance(error, (KeyboardInterrupt, SystemExit))
                else RunOutcome.FAILED
            )
            stop_errors = training.listeners._dispatch_all(
                Stop(context=context, outcome=outcome, error=error)
            )
            for stop_error in stop_errors:
                error.add_note(f"Stop listener failed: {stop_error!r}")
            raise

        stop_errors = training.listeners._dispatch_all(
            Stop(context=context, outcome=RunOutcome.COMPLETED, error=None)
        )
        if stop_errors:
            first_error, *additional_errors = stop_errors
            for error in additional_errors:
                first_error.add_note(f"Another Stop listener failed: {error!r}")
            raise first_error
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


def _interrupt_on_sigterm(signum: int, frame: FrameType | None) -> Never:
    raise SystemExit(128 + signum)


def _validate_training(training: object) -> None:
    if not isinstance(training, Training):
        raise TypeError("setup must return Training")
    if isinstance(training.epochs, bool) or not isinstance(training.epochs, int):
        raise TypeError("epochs must be an integer")
    if training.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not callable(training.batches):
        raise TypeError("batches must be callable")
    if not callable(training.step):
        raise TypeError("step must be callable")
    if not isinstance(training.listeners, EventListeners):
        raise TypeError("listeners must be EventListeners")
