#!/usr/bin/env python3
"""Create a development environment with the right PyTorch build."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MINIMUM_UV_VERSION = (0, 12, 17)
BACKEND_PATTERN = re.compile(r"(?:auto|cpu|xpu|cu\d+|rocm\d+(?:\.\d+)*)\Z")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create .venv and install Skywright with a GPU-aware PyTorch build."
    )
    parser.add_argument(
        "--backend",
        help="uv PyTorch backend such as auto, cpu, cu132, rocm7.2, or xpu",
    )
    parser.add_argument(
        "--python",
        default="3.14",
        help="Python interpreter request passed to uv venv (default: 3.14)",
    )
    parser.add_argument(
        "--no-dev",
        action="store_true",
        help="skip test, lint, and type-check dependencies",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and display the install plan without installing packages",
    )
    return parser


def _uv_version(uv: str) -> tuple[int, ...]:
    result = subprocess.run(
        [uv, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", result.stdout)
    if match is None:
        raise RuntimeError(f"Could not parse uv version from {result.stdout.strip()!r}")
    return tuple(int(part) for part in match.groups())


def _backend(argument: str | None) -> str:
    backend = argument if argument is not None else os.environ.get("UV_TORCH_BACKEND", "auto")
    if BACKEND_PATTERN.fullmatch(backend) is None:
        raise ValueError(
            f"Unsupported backend name {backend!r}. Expected auto, cpu, xpu, cuNNN, or rocmN.N."
        )
    return backend


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    """Create the environment and return a process exit code."""
    args = _parser().parse_args(argv)
    uv = shutil.which("uv")
    if uv is None:
        message = "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
        print(message, file=sys.stderr)
        return 1

    if _uv_version(uv) < MINIMUM_UV_VERSION:
        minimum = ".".join(str(part) for part in MINIMUM_UV_VERSION)
        print(f"uv {minimum} or newer is required for current ROCm indexes.", file=sys.stderr)
        return 1

    try:
        backend = _backend(args.backend)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    environment = PROJECT_ROOT / ".venv"
    if not (environment / "pyvenv.cfg").is_file():
        _run([uv, "venv", "--python", args.python, str(environment)])
    else:
        print(f"Using existing environment at {environment}", flush=True)

    install = [
        uv,
        "pip",
        "install",
        "--python",
        str(environment),
        "--editable",
        ".",
        "--torch-backend",
        backend,
        "--reinstall-package",
        "torch",
        "--upgrade-package",
        "torch",
    ]
    if not args.no_dev:
        install.extend(["--group", "dev"])
    if args.dry_run:
        install.append("--dry-run")
    _run(install)

    if args.dry_run:
        return 0

    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run([str(python), "-m", "skywright.cli", "doctor"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
