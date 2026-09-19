from __future__ import annotations

import pytest

from skywright import EventListeners, RunContext, RunOutcome, Start, Stop, Training, run


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
