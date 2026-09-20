"""Local checkpoint persistence for the managed training loop."""

from __future__ import annotations

import hashlib
import json
import os
import random
import secrets
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from types import TracebackType

    from skywright.accelerator import Accelerator
    from skywright.training import TrainingState

_FORMAT_VERSION = 1
_RUN_RECORD = "run.json"
_CHECKPOINT = "checkpoint.pt"
_PREVIOUS_CHECKPOINT = f".{_CHECKPOINT}.previous.tmp"
_LOCK = ".lock"


class CheckpointError(RuntimeError):
    """A checkpoint directory cannot safely start or resume this run."""


class CheckpointStore(AbstractContextManager["CheckpointStore"]):
    """Own one locked local run directory."""

    def __init__(
        self,
        directory: Path,
        *,
        setup_identity: str,
        arguments: Sequence[str],
        accelerator: Accelerator,
    ) -> None:
        self._directory = directory
        self._setup_identity = setup_identity
        self._arguments_digest = _arguments_digest(arguments)
        self._accelerator = accelerator
        self._lock_file: BinaryIO | None = None
        self.seed = 0

    def __enter__(self) -> CheckpointStore:
        if self._accelerator.kind == "mps":
            raise CheckpointError("checkpointing does not support the MPS backend")

        self._directory.mkdir(parents=True, exist_ok=True)
        lock_path = self._directory / _LOCK
        lock_file = lock_path.open("a+b")
        try:
            _lock_exclusive(lock_file, self._directory)
        except BaseException:
            lock_file.close()
            raise
        self._lock_file = lock_file

        try:
            self.seed = self._open_run_record()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._lock_file is not None:
            _unlock(self._lock_file)
            self._lock_file.close()
            self._lock_file = None

    def load(
        self, state: TrainingState, *, epochs: int
    ) -> tuple[int, int, Mapping[str, object] | None]:
        """Load and validate the last completed epoch, if one exists."""
        expected = self._metadata(state, epochs=epochs)
        checkpoint_path = self._directory / _CHECKPOINT
        if not checkpoint_path.exists():
            return 0, 0, None
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise CheckpointError(f"could not load checkpoint: {error}") from error
        if not isinstance(payload, dict):
            raise CheckpointError("checkpoint payload must be a dictionary")
        checkpoint = cast("dict[str, object]", payload)

        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            raise CheckpointError("checkpoint metadata is missing or malformed")
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise CheckpointError(
                    f"checkpoint {key.replace('_', ' ')} does not match this run"
                )

        next_epoch = checkpoint.get("next_epoch")
        completed_steps = checkpoint.get("completed_steps")
        if not isinstance(next_epoch, int) or isinstance(next_epoch, bool):
            raise CheckpointError("checkpoint next epoch is missing or malformed")
        if not isinstance(completed_steps, int) or isinstance(completed_steps, bool):
            raise CheckpointError("checkpoint completed step count is missing or malformed")
        if next_epoch < 0 or completed_steps < 0:
            raise CheckpointError("checkpoint progress cannot be negative")

        try:
            state.model.load_state_dict(cast("dict[str, object]", checkpoint["model"]))
            if state.scheduler is not None:
                state.scheduler.load_state_dict(
                    cast("dict[str, object]", checkpoint["scheduler"])
                )
            state.optimizer.load_state_dict(
                cast("dict[str, object]", checkpoint["optimizer"])
            )
            if state.scaler is not None:
                state.scaler.load_state_dict(cast("dict[str, object]", checkpoint["scaler"]))
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise CheckpointError(f"checkpoint training state is incompatible: {error}") from error

        rng = checkpoint.get("rng")
        if not isinstance(rng, dict):
            raise CheckpointError("checkpoint RNG state is missing or malformed")
        return next_epoch, completed_steps, cast("Mapping[str, object]", rng)

    def restore_rng(self, rng: Mapping[str, object]) -> None:
        """Restore random generators after Start listeners have run."""
        try:
            random.setstate(cast("tuple[object, ...]", rng["python"]))
            numpy_state = cast("Mapping[str, object]", rng["numpy"])
            np.random.set_state(
                (
                    cast("str", numpy_state["algorithm"]),
                    cast("torch.Tensor", numpy_state["keys"]).cpu().numpy(),
                    cast("int", numpy_state["position"]),
                    cast("int", numpy_state["has_gauss"]),
                    cast("float", numpy_state["cached_gaussian"]),
                )
            )
            torch.set_rng_state(cast("torch.Tensor", rng["torch_cpu"]).cpu())
            device_rng = rng.get("device")
            if device_rng is not None:
                torch.cuda.set_rng_state(
                    cast("torch.Tensor", device_rng).cpu(), self._accelerator.device
                )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise CheckpointError(f"checkpoint RNG state is incompatible: {error}") from error

    def save(
        self,
        state: TrainingState,
        *,
        epochs: int,
        next_epoch: int,
        completed_steps: int,
    ) -> None:
        """Atomically publish state after a completed epoch."""
        model_state = state.model.state_dict()
        payload: dict[str, object] = {
            "metadata": self._metadata(
                state,
                epochs=epochs,
                model_state=model_state,
            ),
            "next_epoch": next_epoch,
            "completed_steps": completed_steps,
            "model": model_state,
            "optimizer": state.optimizer.state_dict(),
            "scheduler": None if state.scheduler is None else state.scheduler.state_dict(),
            "scaler": None if state.scaler is None else state.scaler.state_dict(),
            "rng": self._capture_rng(),
        }
        _validate_payload(payload)
        temporary_path = self._directory / f".{_CHECKPOINT}.{secrets.token_hex(8)}.tmp"
        checkpoint_path = self._directory / _CHECKPOINT
        previous_path = self._directory / _PREVIOUS_CHECKPOINT
        try:
            with temporary_path.open("xb") as checkpoint_file:
                torch.save(payload, checkpoint_file)
                checkpoint_file.flush()
                os.fsync(checkpoint_file.fileno())
            had_previous = checkpoint_path.exists()
            if had_previous:
                os.replace(checkpoint_path, previous_path)
            try:
                os.replace(temporary_path, checkpoint_path)
                _fsync_directory(self._directory)
            except BaseException:
                if had_previous and previous_path.exists():
                    os.replace(previous_path, checkpoint_path)
                else:
                    checkpoint_path.unlink(missing_ok=True)
                with suppress(OSError):
                    _fsync_directory(self._directory)
                raise
            with suppress(OSError):
                previous_path.unlink(missing_ok=True)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _open_run_record(self) -> int:
        record_path = self._directory / _RUN_RECORD
        checkpoint_path = self._directory / _CHECKPOINT
        previous_path = self._directory / _PREVIOUS_CHECKPOINT
        if previous_path.exists():
            if checkpoint_path.exists():
                previous_path.unlink()
            else:
                os.replace(previous_path, checkpoint_path)
            _fsync_directory(self._directory)
        for pattern in (f".{_RUN_RECORD}.*.tmp", f".{_CHECKPOINT}.*.tmp"):
            for temporary_path in self._directory.glob(pattern):
                temporary_path.unlink(missing_ok=True)
        allowed_paths = {_LOCK, _RUN_RECORD, _CHECKPOINT}
        unexpected = sorted(
            path.name for path in self._directory.iterdir() if path.name not in allowed_paths
        )
        if unexpected:
            raise CheckpointError(
                f"checkpoint directory contains unexpected entry {unexpected[0]!r}"
            )

        if not record_path.exists():
            if checkpoint_path.exists():
                raise CheckpointError("checkpoint exists without a run record")
            seed = secrets.randbits(63)
            record = {
                "format_version": _FORMAT_VERSION,
                "setup": self._setup_identity,
                "arguments_digest": self._arguments_digest,
                "seed": seed,
            }
            _atomic_json(record_path, record)
            return seed

        try:
            record = json.loads(record_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CheckpointError(f"could not read run record: {error}") from error
        if not isinstance(record, dict):
            raise CheckpointError("run record must be a JSON object")
        expected = {
            "format_version": _FORMAT_VERSION,
            "setup": self._setup_identity,
            "arguments_digest": self._arguments_digest,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise CheckpointError(f"run record {key.replace('_', ' ')} does not match")
        seed = record.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise CheckpointError("run record seed is missing or malformed")
        return seed

    def _metadata(
        self,
        state: TrainingState,
        *,
        epochs: int,
        model_state: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if model_state is None:
            model_state = state.model.state_dict()
        return {
            "format_version": _FORMAT_VERSION,
            "setup": self._setup_identity,
            "arguments_digest": self._arguments_digest,
            "epochs": epochs,
            "model_type": _type_name(state.model),
            "optimizer_type": _type_name(state.optimizer),
            "scheduler_type": None if state.scheduler is None else _type_name(state.scheduler),
            "scaler_type": None if state.scaler is None else _type_name(state.scaler),
            "optimizer_parameters": _optimizer_parameter_names(state),
            "model_shapes": _model_shapes(model_state),
            "skywright_version": _skywright_version(),
            "torch_version": str(torch.__version__),
            "accelerator": self._accelerator.kind,
        }

    def _capture_rng(self) -> dict[str, object]:
        numpy_state = np.random.get_state()
        device_rng = None
        if self._accelerator.kind in {"cuda", "rocm"} and self._accelerator.available:
            device_rng = torch.cuda.get_rng_state(self._accelerator.device).cpu()
        return {
            "python": random.getstate(),
            "numpy": {
                "algorithm": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].copy()),
                "position": numpy_state[2],
                "has_gauss": numpy_state[3],
                "cached_gaussian": numpy_state[4],
            },
            "torch_cpu": torch.get_rng_state(),
            "device": device_rng,
        }


def seed_random_generators(seed: int, accelerator: Accelerator) -> None:
    """Seed generators before project setup constructs training objects."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if accelerator.kind in {"cuda", "rocm"} and accelerator.available:
        torch.cuda.manual_seed_all(seed)


def setup_identity(setup: object) -> str:
    module = getattr(setup, "__module__", type(setup).__module__)
    qualified_name = getattr(setup, "__qualname__", type(setup).__qualname__)
    return f"{module}:{qualified_name}"


def _arguments_digest(arguments: Sequence[str]) -> str:
    encoded = json.dumps(list(arguments), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _optimizer_parameter_names(state: TrainingState) -> list[list[str]]:
    names_by_identity = {id(parameter): name for name, parameter in state.model.named_parameters()}
    groups: list[list[str]] = []
    for group in state.optimizer.param_groups:
        group_names: list[str] = []
        for parameter in group["params"]:
            name = names_by_identity.get(id(parameter))
            if name is None:
                raise CheckpointError(
                    "every optimizer parameter must be a named parameter of the model"
                )
            group_names.append(name)
        groups.append(group_names)
    return groups


def _model_shapes(model_state: Mapping[str, object]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for name, value in model_state.items():
        if not isinstance(value, torch.Tensor):
            raise CheckpointError("model state must contain tensors only")
        shapes[name] = list(value.shape)
    return shapes


def _skywright_version() -> str:
    from skywright import __version__

    return __version__


def _validate_payload(value: object, path: str = "checkpoint") -> None:
    if value is None or isinstance(value, bool | int | float | str | bytes | torch.Tensor):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, bool | int | float | str | bytes):
                raise CheckpointError(f"{path} contains unsupported key type {type(key).__name__}")
            _validate_payload(item, f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_payload(item, f"{path}[{index}]")
        return
    raise CheckpointError(f"{path} contains unsupported value type {type(value).__name__}")


def _lock_exclusive(lock_file: BinaryIO, directory: Path) -> None:
    try:
        import fcntl
    except ModuleNotFoundError:
        raise CheckpointError("checkpoint directory locking requires a POSIX system") from None
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise CheckpointError(f"checkpoint directory is already in use: {directory}") from None


def _unlock(lock_file: BinaryIO) -> None:
    import fcntl

    fcntl.flock(lock_file, fcntl.LOCK_UN)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as record_file:
            json.dump(value, record_file, sort_keys=True)
            record_file.write("\n")
            record_file.flush()
            os.fsync(record_file.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
