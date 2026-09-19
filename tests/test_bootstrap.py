from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

BOOTSTRAP_PATH = Path(__file__).parents[1] / "scripts" / "bootstrap.py"
SPEC = importlib.util.spec_from_file_location("bootstrap", BOOTSTRAP_PATH)
assert SPEC is not None
assert SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


@pytest.mark.parametrize("backend", ["auto", "cpu", "xpu", "cu130", "rocm7.2", "rocm6.2.4"])
def test_backend_accepts_uv_names(backend: str) -> None:
    assert bootstrap._backend(backend) == backend


@pytest.mark.parametrize("backend", ["cuda", "rocm", "cu13.0", "", "shell command"])
def test_backend_rejects_ambiguous_or_invalid_names(backend: str) -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        bootstrap._backend(backend)


@pytest.mark.parametrize(
    ("output", "expected"),
    [("uv 0.11.3\n", (0, 11, 3)), ("uv 0.12.17 (abcdef 2026-09-18)\n", (0, 12, 17))],
)
def test_uv_version(
    monkeypatch: pytest.MonkeyPatch, output: str, expected: tuple[int, ...]
) -> None:
    class Result:
        stdout = output

    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: Result())

    assert bootstrap._uv_version("uv") == expected
