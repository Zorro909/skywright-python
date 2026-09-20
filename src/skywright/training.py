"""Skywright-owned training lifecycle."""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import FrameType
from typing import Never, cast

import torch

from skywright.accelerator import Accelerator, inspect_accelerator
from skywright.checkpointing import CheckpointStore, seed_random_generators, setup_identity
from skywright.events import EventListeners, RunOutcome, Start, Stop


@dataclass(frozen=True, slots=True)
class RunContext:
    """Runtime information supplied to project setup."""

    argv: tuple[str, ...]
    accelerator: Accelerator


@dataclass(frozen=True, slots=True)
class TrainingState:
    """PyTorch objects restored together when a run resumes."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    scaler: torch.amp.GradScaler | None = None


@dataclass(frozen=True, slots=True)
class Training[Batch]:
    """Project training behavior executed by Skywright."""

    epochs: int
    batches: Callable[[int], Iterable[Batch]]
    step: Callable[[Batch], None]
    listeners: EventListeners = field(default_factory=EventListeners)
    state: TrainingState | None = None


type Setup[Batch] = Callable[[RunContext], Training[Batch]]


def run[Batch](
    setup: Setup[Batch],
    argv: Sequence[str] = (),
    *,
    checkpoint_dir: str | PathLike[str] | None = None,
) -> int:
    """Set up and execute a training run."""
    if not callable(setup):
        raise TypeError("setup must be callable")

    arguments = tuple(argv)
    if not all(isinstance(argument, str) for argument in arguments):
        raise TypeError("every argv item must be a string")

    context = RunContext(argv=arguments, accelerator=inspect_accelerator())
    if checkpoint_dir is None:
        training = setup(context)
        _validate_training(training)
        return _execute(context, training)

    store = CheckpointStore(
        Path(checkpoint_dir),
        setup_identity=setup_identity(setup),
        arguments=arguments,
        accelerator=context.accelerator,
    )
    with store:
        seed_random_generators(store.seed, context.accelerator)
        training = setup(context)
        _validate_training(training)
        if training.state is None:
            raise ValueError("checkpoint_dir requires Training.state")
        next_epoch, completed_steps, rng = store.load(
            training.state, epochs=training.epochs
        )
        if next_epoch < 0 or next_epoch > training.epochs:
            raise ValueError("checkpoint next epoch is outside the configured epoch range")
        return _execute(
            context,
            training,
            checkpoint_store=store,
            next_epoch=next_epoch,
            completed_steps=completed_steps,
            resume_rng=rng,
        )


def _execute[Batch](
    context: RunContext,
    training: Training[Batch],
    *,
    checkpoint_store: CheckpointStore | None = None,
    next_epoch: int = 0,
    completed_steps: int = 0,
    resume_rng: Mapping[str, object] | None = None,
) -> int:
    """Execute validated training and preserve lifecycle event behavior."""

    previous_sigterm_handler = signal.signal(signal.SIGTERM, _interrupt_on_sigterm)
    try:
        try:
            training.listeners._dispatch(Start(context=context))
            if resume_rng is not None:
                assert checkpoint_store is not None
                checkpoint_store.restore_rng(resume_rng)
            for epoch in range(next_epoch, training.epochs):
                for batch in training.batches(epoch):
                    training.step(batch)
                    completed_steps += 1
                if checkpoint_store is not None:
                    checkpoint_store.save(
                        cast("TrainingState", training.state),
                        epochs=training.epochs,
                        next_epoch=epoch + 1,
                        completed_steps=completed_steps,
                    )
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
    if training.state is not None:
        if not isinstance(training.state, TrainingState):
            raise TypeError("state must be TrainingState")
        if not isinstance(training.state.model, torch.nn.Module):
            raise TypeError("state.model must be torch.nn.Module")
        if not isinstance(training.state.optimizer, torch.optim.Optimizer):
            raise TypeError("state.optimizer must be torch.optim.Optimizer")
        if training.state.scheduler is not None and not isinstance(
            training.state.scheduler, torch.optim.lr_scheduler.LRScheduler
        ):
            raise TypeError("state.scheduler must be torch.optim.lr_scheduler.LRScheduler")
        if training.state.scaler is not None and not isinstance(
            training.state.scaler, torch.amp.GradScaler
        ):
            raise TypeError("state.scaler must be torch.amp.GradScaler")
