# Skywright

Skywright is a Python library being built to run PyTorch training and fine-tuning jobs inside
Docker containers. Skywright will be the container's entrypoint: it prepares the workspace,
bootstraps your AI project, and runs an epoch → step training loop using your project's code.

The goal is to remove repeated deployment work from training projects. A project supplies its
models, data, and training logic. Skywright handles the surrounding runtime so deploying that
project takes less setup, starts predictably, and protects the work a training run produces.

## Current status

Skywright is pre-alpha. Today it provides a packaged Python library, a development environment
installer with PyTorch backend selection, and `skywright doctor` for accelerator diagnostics.
The container entrypoint for training, runtime workspace management, project bootstrap API, and
epoch/step runner are not implemented yet. The lifecycle below describes the intended design,
not an API you can call today.

## Intended runtime lifecycle

The primary deployment unit is an image containing Skywright, your AI project, and its
dependencies. Install dependencies while building the image and choose the PyTorch backend for
the deployment target explicitly. Container startup should use those installed dependencies,
without resolving or upgrading packages. Local development should use the same training
lifecycle without requiring Docker.

```text
Container starts Skywright
  → Prepare the run workspace and inspect the runtime
  → Load the AI project and call its bootstrap code
  → Validate the resulting run options and initialize training
  → For each epoch
      → For each step, call the project's training code
  → Finalize outputs, release resources, and exit
```

1. Skywright prepares a workspace for the run, with explicit locations for inputs, temporary
   files, and outputs. Durable outputs belong on mounted storage so they survive container
   removal. Startup checks configuration, required devices, and writable output paths before
   expensive training work begins. A requested GPU that is unavailable should fail startup.
2. The project's bootstrap code receives the workspace and runtime information. It can supply
   project-specific options and configure the run, such as its epoch count. Skywright validates
   the final options before entering the loop. Model and data initialization happens after those
   checks. This runtime bootstrap is separate from the repository's development installer.
3. Skywright advances the epochs and steps. The project supplies the steps for each epoch and
   implements the work of a step, including forward passes, loss computation, and optimization.
   A step does not have to equal an optimizer update, so gradient accumulation stays under the
   project's control. Keep the integration small enough that a project can use ordinary Python
   functions without adopting a class hierarchy or callback framework.
4. Skywright owns the run's exit status, cleanup, and handling of stop requests. Training errors
   must remain visible and produce a failed exit. Graceful shutdown should allow the project to
   save at a safe step boundary within the container's shutdown deadline. Abrupt termination
   still requires recovery from an earlier checkpoint.

Skywright should coordinate checkpoint storage and resume when those features are added; the
project defines the training state to serialize and restore. Resume must be explicit, and a new
run must not silently overwrite existing outputs. A reusable workspace does not by itself mean
training has resumed.

## Keep the boundary small

Skywright owns the infrastructure around a run. The AI project owns the model architecture,
dataset interpretation, numerical training behavior, and evaluation. Docker and the deployment
platform supply device access, mounts, and process scheduling. The first runner should serve a
single training process; distributed execution and external services need a concrete use case
before they become part of the design.

KISS is a design constraint from day one. Start with one explicit lifecycle, a small set of run
options, and direct calls into project code. Add an abstraction, dependency, or configuration
layer only when it solves a current problem for library users or maintainers. Describe that
benefit and why the simpler approach is insufficient. Speculative flexibility is not enough.
Keep backend-specific details at the boundary, and keep orchestration easy to follow.

## Set up a development checkout

These commands prepare this repository for development. They do not launch a training job or
create a runtime workspace.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.11.3 or newer, then run:

```console
python scripts/bootstrap.py
```

The bootstrap command creates a Python 3.14 environment in `.venv`, asks uv to detect the local
accelerator, and upgrades PyTorch within the bounds in `pyproject.toml`. You can also select a
backend explicitly. For a CPU development environment, use:

```console
python scripts/bootstrap.py --backend cpu
```

Hardware discovery can be unreliable inside a container or on a login node. Override it when
needed:

```console
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

## Development checks

Library code belongs in `src/skywright`, with tests under `tests`. CI uses a CPU environment and
runs Ruff, ty, and pytest with coverage. The commands above run the same tools locally.
`skywright doctor` reports the installed PyTorch build and runtime availability for CUDA, ROCm,
and MPS, with CPU as the fallback. XPU installation can be requested, but the diagnostic does
not yet identify XPU devices. The doctor command is informational; it does not enforce a
training job's device requirements.

## License

MIT
