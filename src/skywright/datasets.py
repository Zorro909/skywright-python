"""Dataset definitions resolved against deployment configuration."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import uuid4

import torch.distributed as distributed

if TYPE_CHECKING:
    from datasets import IterableDataset

    from skywright.events import EventListeners

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CONFIG_ENVIRONMENT_VARIABLE = "SKYWRIGHT_DATASETS_CONFIG"
_DEFAULT_CONFIG_PATH = Path("/etc/skywright/datasets.toml")
_ROOT_KEYS = frozenset({"storage", "datasets"})
_STORAGE_KEYS = frozenset({"endpoint_url", "region", "profile"})
_DATASET_KEYS = frozenset({"name", "revision", "split", "files"})


@dataclass(frozen=True, slots=True)
class _Binding:
    name: str
    revision: str
    split: str
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Configuration:
    storage_options: dict[str, object]
    bindings: tuple[_Binding, ...]


@dataclass(frozen=True, slots=True)
class Dataset:
    """A logical dataset resolved to Parquet files when opened."""

    name: str
    revision: str
    split: str = "train"

    def __post_init__(self) -> None:
        _validate_identifier("name", self.name)
        _validate_identifier("revision", self.revision)
        _validate_identifier("split", self.split)

    def open(
        self,
        *,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        shuffle_buffer_size: int = 1000,
        listeners: EventListeners | None = None,
    ) -> IterableDataset:
        """Open this dataset as a streaming Hugging Face iterable."""
        from skywright.events import (
            DatasetOpened,
            DatasetOpenFailed,
            DatasetOpenStarted,
            EventListeners,
        )

        if listeners is None:
            listeners = EventListeners()
        elif not isinstance(listeners, EventListeners):
            raise TypeError("listeners must be EventListeners")

        operation_id = str(uuid4())
        started_at = monotonic()
        try:
            listeners._dispatch(DatasetOpenStarted(dataset=self, operation_id=operation_id))
            _validate_open_options(shuffle, shuffle_seed, shuffle_buffer_size)
            configuration = _load_configuration()
            binding = _resolve_binding(configuration, self)
            world_size, rank = _distributed_context(len(binding.files))
            reader = _open_parquet(binding.files, configuration.storage_options)
            if shuffle:
                reader = reader.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
            if world_size > 1:
                reader = _split_by_rank(reader, rank=rank, world_size=world_size)
        except BaseException as error:
            failed = DatasetOpenFailed(
                dataset=self,
                operation_id=operation_id,
                error=error,
                elapsed_seconds=monotonic() - started_at,
            )
            for listener_error in listeners._dispatch_all(failed):
                error.add_note(f"DatasetOpenFailed listener failed: {listener_error!r}")
            raise

        listeners._dispatch(
            DatasetOpened(
                dataset=self,
                operation_id=operation_id,
                source_file_count=len(binding.files),
                elapsed_seconds=monotonic() - started_at,
            )
        )
        return reader


def _validate_identifier(field_name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"dataset {field_name} must be a string")
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"dataset {field_name} must start with an ASCII letter or digit and contain only "
            "ASCII letters, digits, '.', '_', or '-' (maximum 128 characters)"
        )


def _validate_open_options(shuffle: object, seed: object, buffer_size: object) -> None:
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("shuffle_seed must be an integer")
    if seed < 0:
        raise ValueError("shuffle_seed must not be negative")
    if isinstance(buffer_size, bool) or not isinstance(buffer_size, int):
        raise TypeError("shuffle_buffer_size must be an integer")
    if buffer_size <= 0:
        raise ValueError("shuffle_buffer_size must be positive")


def _load_configuration() -> _Configuration:
    filename = os.environ.get(_CONFIG_ENVIRONMENT_VARIABLE)
    if filename is not None and not filename.strip():
        raise RuntimeError(f"{_CONFIG_ENVIRONMENT_VARIABLE} must not be empty")

    path = Path(filename) if filename is not None else _DEFAULT_CONFIG_PATH
    with path.open("rb") as file:
        document = tomllib.load(file)

    _reject_unknown_keys(document, _ROOT_KEYS, "dataset configuration")
    storage_options = _parse_storage(document.get("storage", {}))
    raw_bindings = document.get("datasets")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise ValueError("dataset configuration must contain at least one [[datasets]] binding")

    bindings = tuple(_parse_binding(value, index) for index, value in enumerate(raw_bindings))
    identities: set[tuple[str, str, str]] = set()
    for binding in bindings:
        identity = (binding.name, binding.revision, binding.split)
        if identity in identities:
            raise ValueError(
                "dataset configuration contains duplicate binding "
                f"{binding.name!r}/{binding.revision!r}/{binding.split!r}"
            )
        identities.add(identity)
    return _Configuration(storage_options=storage_options, bindings=bindings)


def _parse_storage(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("storage must be a TOML table")
    _reject_unknown_keys(value, _STORAGE_KEYS, "storage")

    options: dict[str, object] = {}
    endpoint = value.get("endpoint_url")
    if endpoint is not None:
        endpoint = _nonempty_string("storage.endpoint_url", endpoint)
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("storage.endpoint_url must be an HTTP or HTTPS URL with a host")
        options["endpoint_url"] = endpoint

    region = value.get("region")
    if region is not None:
        options["client_kwargs"] = {"region_name": _nonempty_string("storage.region", region)}

    profile = value.get("profile")
    if profile is not None:
        options["profile"] = _nonempty_string("storage.profile", profile)
    return options


def _parse_binding(value: object, index: int) -> _Binding:
    label = f"datasets[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a TOML table")
    _reject_unknown_keys(value, _DATASET_KEYS, label)

    fields: dict[str, str] = {}
    for field_name in ("name", "revision", "split"):
        if field_name not in value:
            raise ValueError(f"{label}.{field_name} is required")
        field_value = value[field_name]
        if not isinstance(field_value, str):
            raise ValueError(f"{label}.{field_name} must be a string")
        _validate_identifier(field_name, field_value)
        fields[field_name] = field_value

    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError(f"{label}.files must be a nonempty array")
    files = tuple(_validate_file_uri(file, label) for file in raw_files)
    if len(files) != len(set(files)):
        raise ValueError(f"{label}.files must not contain duplicates")
    return _Binding(
        name=fields["name"],
        revision=fields["revision"],
        split=fields["split"],
        files=files,
    )


def _validate_file_uri(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}.files entries must be strings")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "s3"
        or not parsed.hostname
        or parsed.netloc != parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.lstrip("/")
        or not parsed.path.endswith(".parquet")
        or any(character in value for character in "*?[")
    ):
        raise ValueError(f"{label}.files entry must be an explicit s3:// URI ending in .parquet")
    return value


def _nonempty_string(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _reject_unknown_keys(
    value: Mapping[str, object], allowed: frozenset[str], label: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        rendered = ", ".join(sorted(repr(key) for key in unknown))
        raise ValueError(f"{label} contains unknown key(s): {rendered}")


def _resolve_binding(configuration: _Configuration, dataset: Dataset) -> _Binding:
    for binding in configuration.bindings:
        if (binding.name, binding.revision, binding.split) == (
            dataset.name,
            dataset.revision,
            dataset.split,
        ):
            return binding
    raise ValueError(
        "dataset configuration has no binding for "
        f"{dataset.name!r}/{dataset.revision!r}/{dataset.split!r}"
    )


def _distributed_context(source_file_count: int) -> tuple[int, int]:
    if distributed.is_available() and distributed.is_initialized():
        world_size = distributed.get_world_size()
        rank = distributed.get_rank()
        if source_file_count % world_size:
            raise ValueError(
                f"dataset has {source_file_count} source files, which is not divisible by "
                f"distributed world size {world_size}"
            )
        return world_size, rank

    raw_world_size = os.environ.get("WORLD_SIZE")
    if raw_world_size is not None:
        try:
            world_size = int(raw_world_size)
        except ValueError:
            raise RuntimeError("WORLD_SIZE must be an integer") from None
        if world_size <= 0:
            raise RuntimeError("WORLD_SIZE must be positive")
        if world_size > 1:
            raise RuntimeError(
                "WORLD_SIZE is greater than one but torch.distributed is not initialized"
            )
    return 1, 0


def _open_parquet(files: tuple[str, ...], storage_options: dict[str, object]) -> IterableDataset:
    import datasets
    import s3fs  # noqa: F401

    return datasets.load_dataset(
        "parquet",
        data_files={"train": list(files)},
        split="train",
        streaming=True,
        storage_options=storage_options,
        on_bad_files="error",
    )


def _split_by_rank(reader: IterableDataset, *, rank: int, world_size: int) -> IterableDataset:
    from datasets.distributed import split_dataset_by_node

    return split_dataset_by_node(reader, rank=rank, world_size=world_size)
