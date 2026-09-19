# Skywright

Skywright is an opinionated Python library for deploying PyTorch training and fine-tuning jobs.
The public API is still taking shape. This repository currently provides the development base: a
packaged `src` layout, strict linting and type checks, tests, and GPU-aware PyTorch installation.

## Set up the workspace

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.7.14 or newer, then run:

```console
python scripts/bootstrap.py
```

The bootstrap command creates `.venv`, asks uv to detect the local accelerator, and installs the
matching PyTorch build. uv checks for NVIDIA CUDA, AMD ROCm, and Intel XPU support. It falls back to
the CPU build when it finds no supported accelerator.

Hardware discovery can be unreliable inside a container or on a login node. Override it when
needed:

```console
python scripts/bootstrap.py --backend cpu
python scripts/bootstrap.py --backend cu130
python scripts/bootstrap.py --backend rocm7.2
python scripts/bootstrap.py --backend xpu
```

Backend names track uv and PyTorch. Run `uv pip install --help` to see the versions supported by
your installed uv release.

The backend selector is part of uv's pip-compatible interface, not its project sync interface.
Running a plain `uv sync` may replace the selected build. Use the environment created by the
bootstrap command directly, or add `--no-sync` when using `uv run`:

```console
uv run --no-sync skywright doctor
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ty check
```

Set `UV_TORCH_BACKEND` instead of passing `--backend` if that works better in automation:

```console
UV_TORCH_BACKEND=rocm7.2 python scripts/bootstrap.py
```

## Package layout

Library code belongs in `src/skywright`. Tests mirror it under `tests`. `skywright doctor` reports
the installed PyTorch build and the accelerator visible at runtime, which is useful as a first
check on a new training host.

## License

MIT
