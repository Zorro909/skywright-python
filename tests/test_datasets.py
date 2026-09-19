from __future__ import annotations

from pathlib import Path
from typing import cast

import datasets
import pytest

from skywright import (
    Dataset,
    DatasetOpened,
    DatasetOpenFailed,
    DatasetOpenStarted,
    EventListeners,
)
from skywright import datasets as dataset_module


class FakeReader:
    def __init__(self) -> None:
        self.shuffle_calls: list[tuple[int, int]] = []

    def shuffle(self, *, seed: int, buffer_size: int) -> FakeReader:
        self.shuffle_calls.append((seed, buffer_size))
        return self


def write_config(path: Path, text: str) -> None:
    path.write_text(text)


def configure(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("SKYWRIGHT_DATASETS_CONFIG", str(path))
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def test_definition_validates_identity_without_loading_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SKYWRIGHT_DATASETS_CONFIG", raising=False)

    assert Dataset("images", "2026-09-01") == Dataset("images", "2026-09-01", "train")

    with pytest.raises(ValueError, match="dataset name"):
        Dataset("bad/name", "2026-09-01")
    with pytest.raises(TypeError, match="dataset revision"):
        Dataset("images", cast("str", 1))


def test_open_uses_standard_configuration_path_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "datasets.toml"
    write_config(
        config,
        """
        [[datasets]]
        name = "images"
        revision = "v1"
        split = "train"
        files = ["s3://bucket/train.parquet"]
        """,
    )
    monkeypatch.delenv("SKYWRIGHT_DATASETS_CONFIG", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(dataset_module, "_DEFAULT_CONFIG_PATH", config)
    reader = FakeReader()
    monkeypatch.setattr(dataset_module, "_open_parquet", lambda files, options: reader)

    assert Dataset("images", "v1").open() is reader


def test_open_resolves_files_storage_shuffle_and_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "datasets.toml"
    write_config(
        config,
        """
        [storage]
        endpoint_url = "https://objects.example.com"
        region = "eu-central-1"
        profile = "training"

        [[datasets]]
        name = "images"
        revision = "2026-09-01"
        split = "train"
        files = [
            "s3://training-data/images/train-000.parquet",
            "s3://training-data/images/train-001.parquet",
        ]

        [[datasets]]
        name = "images"
        revision = "2026-09-01"
        split = "validation"
        files = ["s3://training-data/images/validation.parquet"]
        """,
    )
    configure(monkeypatch, config)
    reader = FakeReader()
    opened_with: list[object] = []

    def open_parquet(files: tuple[str, ...], storage_options: dict[str, object]) -> FakeReader:
        opened_with.extend((files, storage_options))
        return reader

    monkeypatch.setattr(dataset_module, "_open_parquet", open_parquet)
    observed: list[object] = []
    listeners = EventListeners()
    listeners.add(DatasetOpenStarted, observed.append)
    listeners.add(DatasetOpened, observed.append)
    dataset = Dataset("images", "2026-09-01")

    assert dataset.open(
        shuffle=True,
        shuffle_seed=9,
        shuffle_buffer_size=256,
        listeners=listeners,
    ) is reader

    assert opened_with == [
        (
            "s3://training-data/images/train-000.parquet",
            "s3://training-data/images/train-001.parquet",
        ),
        {
            "endpoint_url": "https://objects.example.com",
            "client_kwargs": {"region_name": "eu-central-1"},
            "profile": "training",
        },
    ]
    assert reader.shuffle_calls == [(9, 256)]
    assert len(observed) == 2
    assert isinstance(observed[0], DatasetOpenStarted)
    assert isinstance(observed[1], DatasetOpened)
    assert observed[0].dataset is dataset
    assert observed[0].operation_id == observed[1].operation_id
    assert observed[1].source_file_count == 2
    assert observed[1].elapsed_seconds >= 0


def test_hugging_face_adapter_opens_parquet_as_a_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = FakeReader()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load_dataset(*args: object, **kwargs: object) -> FakeReader:
        calls.append((args, kwargs))
        return reader

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    assert dataset_module._open_parquet(
        ("s3://bucket/0.parquet", "s3://bucket/1.parquet"),
        {"endpoint_url": "https://objects.example.com"},
    ) is reader
    assert calls == [
        (
            ("parquet",),
            {
                "data_files": {
                    "train": ["s3://bucket/0.parquet", "s3://bucket/1.parquet"]
                },
                "split": "train",
                "streaming": True,
                "storage_options": {"endpoint_url": "https://objects.example.com"},
                "on_bad_files": "error",
            },
        )
    ]


def test_open_failure_emits_failure_and_preserves_the_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "datasets.toml"
    write_config(
        config,
        """
        [[datasets]]
        name = "images"
        revision = "v1"
        split = "train"
        files = ["s3://bucket/train.parquet"]
        """,
    )
    configure(monkeypatch, config)
    original_error = OSError("storage unavailable")

    def fail_open(files: tuple[str, ...], storage_options: dict[str, object]) -> FakeReader:
        raise original_error

    monkeypatch.setattr(dataset_module, "_open_parquet", fail_open)
    observed: list[object] = []
    listeners = EventListeners()
    listeners.add(DatasetOpenStarted, observed.append)
    listeners.add(DatasetOpenFailed, observed.append)

    with pytest.raises(OSError) as caught:
        Dataset("images", "v1").open(listeners=listeners)

    assert caught.value is original_error
    assert len(observed) == 2
    assert isinstance(observed[0], DatasetOpenStarted)
    assert isinstance(observed[1], DatasetOpenFailed)
    assert observed[1].error is original_error
    assert observed[0].operation_id == observed[1].operation_id


def test_success_listener_failure_does_not_emit_open_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "datasets.toml"
    write_config(
        config,
        """
        [[datasets]]
        name = "images"
        revision = "v1"
        split = "train"
        files = ["s3://bucket/train.parquet"]
        """,
    )
    configure(monkeypatch, config)
    monkeypatch.setattr(dataset_module, "_open_parquet", lambda files, options: FakeReader())
    observed: list[str] = []
    listeners = EventListeners()

    def fail_success(event: DatasetOpened) -> None:
        raise RuntimeError("listener failed")

    listeners.add(DatasetOpened, fail_success)
    listeners.add(DatasetOpenFailed, lambda event: observed.append("failed"))

    with pytest.raises(RuntimeError, match="listener failed"):
        Dataset("images", "v1").open(listeners=listeners)

    assert observed == []


@pytest.mark.parametrize(
    ("config_text", "message"),
    [
        (
            """
            unexpected = true
            [[datasets]]
            name = "images"
            revision = "v1"
            split = "train"
            files = ["s3://bucket/train.parquet"]
            """,
            "unknown key",
        ),
        (
            """
            [[datasets]]
            name = "images"
            revision = "v1"
            split = "train"
            files = ["s3://bucket/*.parquet"]
            """,
            "explicit s3:// URI",
        ),
        (
            """
            [[datasets]]
            name = "images"
            revision = "v1"
            split = "train"
            files = ["s3://bucket/train.parquet"]
            [[datasets]]
            name = "images"
            revision = "v1"
            split = "train"
            files = ["s3://bucket/other.parquet"]
            """,
            "duplicate binding",
        ),
    ],
)
def test_invalid_configuration_fails_before_reader_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_text: str,
    message: str,
) -> None:
    config = tmp_path / "datasets.toml"
    write_config(config, config_text)
    configure(monkeypatch, config)
    called = False

    def open_parquet(files: tuple[str, ...], storage_options: dict[str, object]) -> FakeReader:
        nonlocal called
        called = True
        return FakeReader()

    monkeypatch.setattr(dataset_module, "_open_parquet", open_parquet)

    with pytest.raises(ValueError, match=message):
        Dataset("images", "v1").open()

    assert not called


def test_distributed_reader_is_split_after_shuffle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "datasets.toml"
    write_config(
        config,
        """
        [[datasets]]
        name = "images"
        revision = "v1"
        split = "train"
        files = ["s3://bucket/0.parquet", "s3://bucket/1.parquet"]
        """,
    )
    configure(monkeypatch, config)
    reader = FakeReader()
    split_calls: list[tuple[object, int, int]] = []
    monkeypatch.setattr(dataset_module, "_open_parquet", lambda files, options: reader)
    monkeypatch.setattr(dataset_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(dataset_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(dataset_module.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(dataset_module.distributed, "get_rank", lambda: 1)

    def split_by_rank(value: object, *, rank: int, world_size: int) -> object:
        split_calls.append((value, rank, world_size))
        return "rank reader"

    monkeypatch.setattr(dataset_module, "_split_by_rank", split_by_rank)

    result = Dataset("images", "v1").open(shuffle=True)

    assert result == "rank reader"
    assert reader.shuffle_calls == [(42, 1000)]
    assert split_calls == [(reader, 1, 2)]


def test_distributed_file_count_must_divide_world_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "datasets.toml"
    write_config(
        config,
        """
        [[datasets]]
        name = "images"
        revision = "v1"
        split = "train"
        files = ["s3://bucket/0.parquet"]
        """,
    )
    configure(monkeypatch, config)
    monkeypatch.setattr(dataset_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(dataset_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(dataset_module.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(dataset_module.distributed, "get_rank", lambda: 0)

    with pytest.raises(ValueError, match="not divisible"):
        Dataset("images", "v1").open()


def test_declared_distributed_run_requires_initialized_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "datasets.toml"
    write_config(
        config,
        """
        [[datasets]]
        name = "images"
        revision = "v1"
        split = "train"
        files = ["s3://bucket/0.parquet"]
        """,
    )
    configure(monkeypatch, config)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(dataset_module.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="not initialized"):
        Dataset("images", "v1").open()
