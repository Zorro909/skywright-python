from __future__ import annotations

import random
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pytest
import torch

from skywright import (
    EventListeners,
    RunContext,
    RunOutcome,
    Start,
    Stop,
    Training,
    TrainingState,
    run,
)


def test_run_controls_training_lifecycle() -> None:
    observed: list[object] = []
    listeners = EventListeners()
    listeners.add(Start, lambda event: observed.append(("start:first", event.context.argv)))
    listeners.add(Start, lambda event: observed.append(("start:second", event.context.argv)))
    listeners.add(
        Stop,
        lambda event: observed.append(("stop", event.outcome, event.error)),
    )

    def setup(context: RunContext) -> Training[str]:
        observed.append(("setup", context.argv))
        return Training(
            epochs=2,
            batches=lambda epoch: (f"{epoch}:first", f"{epoch}:second"),
            step=lambda batch: observed.append(("step", batch)),
            listeners=listeners,
        )

    assert run(setup, ["--learning-rate", "0.01"]) == 0
    assert observed == [
        ("setup", ("--learning-rate", "0.01")),
        ("start:first", ("--learning-rate", "0.01")),
        ("start:second", ("--learning-rate", "0.01")),
        ("step", "0:first"),
        ("step", "0:second"),
        ("step", "1:first"),
        ("step", "1:second"),
        ("stop", RunOutcome.COMPLETED, None),
    ]


def test_setup_failure_emits_no_events() -> None:
    observed: list[str] = []
    listeners = EventListeners()
    listeners.add(Start, lambda event: observed.append("start"))
    listeners.add(Stop, lambda event: observed.append("stop"))

    def setup(context: RunContext) -> Training[object]:
        raise RuntimeError("setup failed")

    with pytest.raises(RuntimeError, match="setup failed"):
        run(setup)

    assert observed == []


def test_invalid_training_is_rejected_before_start() -> None:
    observed: list[str] = []
    listeners = EventListeners()
    listeners.add(Start, lambda event: observed.append("start"))

    def setup(context: RunContext) -> Training[object]:
        return Training(
            epochs=0,
            batches=lambda epoch: (),
            step=lambda batch: None,
            listeners=listeners,
        )

    with pytest.raises(ValueError, match="epochs must be positive"):
        run(setup)

    assert observed == []


def test_stop_reaches_every_listener_without_hiding_step_failure() -> None:
    original_error = RuntimeError("step failed")
    secondary_error = RuntimeError("stop failed")
    observed: list[Stop] = []
    listener_order: list[str] = []
    listeners = EventListeners()

    def fail_on_stop(event: Stop) -> None:
        listener_order.append("first")
        observed.append(event)
        raise secondary_error

    def observe_stop(event: Stop) -> None:
        listener_order.append("second")
        observed.append(event)

    listeners.add(Stop, fail_on_stop)
    listeners.add(Stop, observe_stop)

    def fail_step(batch: object) -> None:
        raise original_error

    def setup(context: RunContext) -> Training[object]:
        return Training(
            epochs=1,
            batches=lambda epoch: (object(),),
            step=fail_step,
            listeners=listeners,
        )

    with pytest.raises(RuntimeError) as caught:
        run(setup)

    assert caught.value is original_error
    assert listener_order == ["first", "second"]
    assert len(observed) == 2
    assert all(event.outcome is RunOutcome.FAILED for event in observed)
    assert all(event.error is original_error for event in observed)


def test_start_failure_still_emits_stop() -> None:
    original_error = RuntimeError("start failed")
    observed: list[Stop] = []
    listeners = EventListeners()

    def fail_on_start(event: Start) -> None:
        raise original_error

    listeners.add(Start, fail_on_start)
    listeners.add(Stop, observed.append)

    def setup(context: RunContext) -> Training[object]:
        return Training(
            epochs=1,
            batches=lambda epoch: (),
            step=lambda batch: None,
            listeners=listeners,
        )

    with pytest.raises(RuntimeError) as caught:
        run(setup)

    assert caught.value is original_error
    assert len(observed) == 1
    assert observed[0].outcome is RunOutcome.FAILED
    assert observed[0].error is original_error


def test_interruption_is_reported_to_stop_listeners() -> None:
    interruption = KeyboardInterrupt()
    observed: list[Stop] = []
    listeners = EventListeners()
    listeners.add(Stop, observed.append)

    def interrupt_step(batch: object) -> None:
        raise interruption

    def setup(context: RunContext) -> Training[object]:
        return Training(
            epochs=1,
            batches=lambda epoch: (object(),),
            step=interrupt_step,
            listeners=listeners,
        )

    with pytest.raises(KeyboardInterrupt) as caught:
        run(setup)

    assert caught.value is interruption
    assert len(observed) == 1
    assert observed[0].outcome is RunOutcome.INTERRUPTED
    assert observed[0].error is interruption


def test_run_resumes_registered_state_at_the_next_epoch(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    visited_epochs: list[int] = []
    attempts = 0
    final_model: torch.nn.Linear | None = None

    def setup(context: RunContext) -> Training[int]:
        nonlocal attempts, final_model
        attempts += 1
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25, momentum=0.5)
        final_model = model

        def batches(epoch: int) -> tuple[int, ...]:
            visited_epochs.append(epoch)
            return (epoch,)

        def step(epoch: int) -> None:
            if attempts == 1 and epoch == 1:
                raise KeyboardInterrupt
            optimizer.zero_grad()
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()

        return Training(
            epochs=2,
            batches=batches,
            step=step,
            state=TrainingState(model=model, optimizer=optimizer),
        )

    with pytest.raises(KeyboardInterrupt):
        run(setup, checkpoint_dir=checkpoint_dir)

    assert final_model is not None
    interrupted_weight = final_model.weight.detach().clone()
    assert run(setup, checkpoint_dir=checkpoint_dir) == 0

    assert visited_epochs == [0, 1, 1]
    assert final_model is not None
    assert not torch.equal(final_model.weight, interrupted_weight)


def test_resumed_run_matches_uninterrupted_training_after_replaying_epoch(
    tmp_path: Path,
) -> None:
    def execute(
        checkpoint_dir: Path | None, *, interrupt_once: bool
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, int, float, float, float, torch.Tensor]:
        attempts = 0
        interrupted = False
        final_state: TrainingState | None = None

        def setup(context: RunContext) -> Training[tuple[int, int]]:
            nonlocal attempts, final_state, interrupted
            attempts += 1
            random.seed(120)
            np.random.seed(121)
            torch.manual_seed(122)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
            scaler = torch.amp.GradScaler("cpu")
            final_state = TrainingState(model, optimizer, scheduler, scaler)
            listeners = EventListeners()

            def consume_randomness(event: Start) -> None:
                random.random()
                np.random.random()
                torch.rand(1)

            listeners.add(Start, consume_randomness)

            def step(batch: tuple[int, int]) -> None:
                nonlocal interrupted
                optimizer.zero_grad()
                multiplier = random.random() + float(np.random.random())
                inputs = torch.rand(2, 2) * multiplier
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = model(inputs).square().sum()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if interrupt_once and not interrupted and batch == (1, 0):
                    interrupted = True
                    raise KeyboardInterrupt

            return Training(
                epochs=3,
                batches=lambda epoch: ((epoch, 0), (epoch, 1)),
                step=step,
                state=final_state,
                listeners=listeners,
            )

        if interrupt_once:
            with pytest.raises(KeyboardInterrupt):
                run(setup, checkpoint_dir=checkpoint_dir)
            assert attempts == 1
            assert run(setup, checkpoint_dir=checkpoint_dir) == 0
        else:
            assert run(setup) == 0

        assert final_state is not None
        assert final_state.scheduler is not None
        assert final_state.scaler is not None
        momentum = final_state.optimizer.state[next(iter(final_state.model.parameters()))]
        return (
            tuple(parameter.detach().clone() for parameter in final_state.model.parameters()),
            momentum["momentum_buffer"].detach().clone(),
            final_state.scheduler.last_epoch,
            final_state.scaler.get_scale(),
            random.random(),
            float(np.random.random()),
            torch.rand(3),
        )

    uninterrupted = execute(None, interrupt_once=False)
    resumed = execute(tmp_path / "checkpoints", interrupt_once=True)

    for actual_tensor, expected_tensor in zip(resumed[0], uninterrupted[0], strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    torch.testing.assert_close(resumed[1], uninterrupted[1])
    assert resumed[2:6] == uninterrupted[2:6]
    torch.testing.assert_close(resumed[6], uninterrupted[6])


def test_checkpoint_arguments_must_match_before_setup_runs_again(tmp_path: Path) -> None:
    setup_calls = 0

    def setup(context: RunContext) -> Training[object]:
        nonlocal setup_calls
        setup_calls += 1
        model = torch.nn.Linear(1, 1)
        return Training(
            epochs=1,
            batches=lambda epoch: (),
            step=lambda batch: None,
            state=TrainingState(model, torch.optim.SGD(model.parameters(), lr=0.1)),
        )

    checkpoint_dir = tmp_path / "checkpoints"
    assert run(setup, ["--rate", "0.1"], checkpoint_dir=checkpoint_dir) == 0

    with pytest.raises(RuntimeError, match="arguments digest does not match"):
        run(setup, ["--rate", "0.2"], checkpoint_dir=checkpoint_dir)

    assert setup_calls == 1


def test_checkpoint_directory_requires_declared_training_state(tmp_path: Path) -> None:
    def setup(context: RunContext) -> Training[object]:
        return Training(epochs=1, batches=lambda epoch: (), step=lambda batch: None)

    with pytest.raises(ValueError, match=r"requires Training\.state"):
        run(setup, checkpoint_dir=tmp_path / "checkpoints")


def test_changed_optimizer_parameter_order_is_rejected_before_start(tmp_path: Path) -> None:
    reverse_parameters = False
    starts = 0

    def setup(context: RunContext) -> Training[object]:
        nonlocal starts
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
        parameters = list(model.parameters())
        if reverse_parameters:
            parameters.reverse()
        optimizer = torch.optim.SGD(parameters, lr=0.1)
        listeners = EventListeners()

        def observe_start(event: Start) -> None:
            nonlocal starts
            starts += 1

        listeners.add(Start, observe_start)
        return Training(
            epochs=1,
            batches=lambda epoch: (),
            step=lambda batch: None,
            state=TrainingState(model, optimizer),
            listeners=listeners,
        )

    checkpoint_dir = tmp_path / "checkpoints"
    assert run(setup, checkpoint_dir=checkpoint_dir) == 0
    reverse_parameters = True

    with pytest.raises(RuntimeError, match="optimizer parameters does not match"):
        run(setup, checkpoint_dir=checkpoint_dir)

    assert starts == 1


def test_changed_model_shape_and_optional_state_are_rejected_before_start(
    tmp_path: Path,
) -> None:
    configuration = "original"
    starts = 0

    def setup(context: RunContext) -> Training[object]:
        nonlocal starts
        output_features = 2 if configuration == "shape" else 1
        model = torch.nn.Linear(1, output_features)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = (
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            if configuration == "scheduler"
            else None
        )
        listeners = EventListeners()

        def observe_start(event: Start) -> None:
            nonlocal starts
            starts += 1

        listeners.add(Start, observe_start)
        return Training(
            epochs=1,
            batches=lambda epoch: (),
            step=lambda batch: None,
            listeners=listeners,
            state=TrainingState(model, optimizer, scheduler),
        )

    checkpoint_dir = tmp_path / "checkpoints"
    assert run(setup, checkpoint_dir=checkpoint_dir) == 0

    configuration = "shape"
    with pytest.raises(RuntimeError, match="model shapes does not match"):
        run(setup, checkpoint_dir=checkpoint_dir)

    configuration = "scheduler"
    with pytest.raises(RuntimeError, match="scheduler type does not match"):
        run(setup, checkpoint_dir=checkpoint_dir)

    assert starts == 1


@pytest.mark.parametrize("field", ["skywright_version", "torch_version", "accelerator"])
def test_changed_runtime_metadata_is_rejected(tmp_path: Path, field: str) -> None:
    def setup(context: RunContext) -> Training[object]:
        model = torch.nn.Linear(1, 1)
        return Training(
            epochs=1,
            batches=lambda epoch: (),
            step=lambda batch: None,
            state=TrainingState(model, torch.optim.SGD(model.parameters(), lr=0.1)),
        )

    checkpoint_dir = tmp_path / "checkpoints"
    assert run(setup, checkpoint_dir=checkpoint_dir) == 0
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["metadata"][field] = "changed"
    torch.save(payload, checkpoint_path)

    with pytest.raises(RuntimeError, match=field.replace("_", " ")):
        run(setup, checkpoint_dir=checkpoint_dir)


def test_checkpoint_without_rng_state_is_rejected(tmp_path: Path) -> None:
    def setup(context: RunContext) -> Training[object]:
        model = torch.nn.Linear(1, 1)
        return Training(
            epochs=1,
            batches=lambda epoch: (),
            step=lambda batch: None,
            state=TrainingState(model, torch.optim.SGD(model.parameters(), lr=0.1)),
        )

    checkpoint_dir = tmp_path / "checkpoints"
    assert run(setup, checkpoint_dir=checkpoint_dir) == 0
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    payload = torch.load(checkpoint_path, weights_only=True)
    del payload["rng"]
    torch.save(payload, checkpoint_path)

    with pytest.raises(RuntimeError, match="RNG state is missing"):
        run(setup, checkpoint_dir=checkpoint_dir)


@pytest.mark.parametrize("failure_stage", ["serialization", "publication"])
def test_failed_checkpoint_write_preserves_previous_completed_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    attempts = 0
    visited_epochs: list[int] = []
    final_model: torch.nn.Linear | None = None

    def setup(context: RunContext) -> Training[int]:
        nonlocal attempts, final_model
        attempts += 1
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        final_model = model

        def step(epoch: int) -> None:
            visited_epochs.append(epoch)
            if attempts == 1 and epoch == 1:
                raise KeyboardInterrupt
            optimizer.zero_grad()
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()

        return Training(
            epochs=2,
            batches=lambda epoch: (epoch,),
            step=step,
            state=TrainingState(model, optimizer),
        )

    checkpoint_dir = tmp_path / "checkpoints"
    with pytest.raises(KeyboardInterrupt):
        run(setup, checkpoint_dir=checkpoint_dir)

    def fail_save(value: object, file: BinaryIO) -> None:
        file.write(b"partial")
        raise OSError("disk failed")

    def fail_fsync(directory: Path) -> None:
        raise OSError("disk failed")

    with monkeypatch.context() as patch:
        if failure_stage == "serialization":
            patch.setattr(torch, "save", fail_save)
        else:
            patch.setattr("skywright.checkpointing._fsync_directory", fail_fsync)
        with pytest.raises(OSError, match="disk failed"):
            run(setup, checkpoint_dir=checkpoint_dir)

    assert run(setup, checkpoint_dir=checkpoint_dir) == 0
    assert visited_epochs == [0, 1, 1, 1]
    assert final_model is not None
    torch.testing.assert_close(final_model.weight, torch.tensor([[-2.0]]))
