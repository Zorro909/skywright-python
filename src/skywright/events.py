"""Synchronous lifecycle events emitted by Skywright."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from skywright.datasets import Dataset
    from skywright.training import RunContext


@dataclass(frozen=True, slots=True)
class Start:
    """Emitted immediately before Skywright starts the training loop."""

    context: RunContext


class RunOutcome(StrEnum):
    """The reason a training run stopped."""

    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class Stop:
    """Emitted when a started run leaves the training loop."""

    context: RunContext
    outcome: RunOutcome
    error: BaseException | None


@dataclass(frozen=True, slots=True)
class DatasetOpenStarted:
    """Emitted before Skywright resolves and opens a dataset."""

    dataset: Dataset
    operation_id: str


@dataclass(frozen=True, slots=True)
class DatasetOpened:
    """Emitted after Skywright constructs a streaming dataset reader."""

    dataset: Dataset
    operation_id: str
    source_file_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class DatasetOpenFailed:
    """Emitted when Skywright cannot construct a dataset reader."""

    dataset: Dataset
    operation_id: str
    error: BaseException
    elapsed_seconds: float


type Event = Start | Stop | DatasetOpenStarted | DatasetOpened | DatasetOpenFailed
type Listener[EventType: Event] = Callable[[EventType], object]
type EventClass = (
    type[Start]
    | type[Stop]
    | type[DatasetOpenStarted]
    | type[DatasetOpened]
    | type[DatasetOpenFailed]
)


class EventListeners:
    """Listeners registered for Skywright lifecycle events."""

    def __init__(self) -> None:
        self._listeners: dict[EventClass, list[Callable[..., object]]] = {
            Start: [],
            Stop: [],
            DatasetOpenStarted: [],
            DatasetOpened: [],
            DatasetOpenFailed: [],
        }

    @overload
    def add(self, event_type: type[Start], listener: Listener[Start], /) -> None: ...

    @overload
    def add(self, event_type: type[Stop], listener: Listener[Stop], /) -> None: ...

    @overload
    def add(
        self,
        event_type: type[DatasetOpenStarted],
        listener: Listener[DatasetOpenStarted],
        /,
    ) -> None: ...

    @overload
    def add(
        self,
        event_type: type[DatasetOpened],
        listener: Listener[DatasetOpened],
        /,
    ) -> None: ...

    @overload
    def add(
        self,
        event_type: type[DatasetOpenFailed],
        listener: Listener[DatasetOpenFailed],
        /,
    ) -> None: ...

    def add(
        self,
        event_type: EventClass,
        listener: Callable[..., object],
        /,
    ) -> None:
        """Register a listener for an event type."""
        if event_type not in self._listeners:
            raise ValueError(f"Unsupported event type: {event_type!r}")
        if not callable(listener):
            raise TypeError("listener must be callable")
        self._listeners[event_type].append(listener)

    def _dispatch(self, event: Event) -> None:
        for listener in tuple(self._listeners[type(event)]):
            listener(event)

    def _dispatch_all(self, event: Stop | DatasetOpenFailed) -> tuple[BaseException, ...]:
        errors: list[BaseException] = []
        for listener in tuple(self._listeners[type(event)]):
            try:
                listener(event)
            except BaseException as error:
                errors.append(error)
        return tuple(errors)
