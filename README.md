# Skywright

Skywright aims to make PyTorch training jobs simpler, faster, and safer to deploy,
mainly inside Docker containers.

The idea is that Skywright is the container's entrypoint. It sets up the workspace,
lets your AI project configure options during bootstrap, then runs an epoch → step
training loop. Your project supplies the training code. Skywright handles the
infrastructure around it.

The project is in early development. Skywright currently provides accelerator
diagnostics and a small training runtime with setup, epoch/step execution, and
start/stop events.

We follow KISS. Added complexity must have a clear benefit for library users or
maintainers.

## Docker base images

The base image packages Skywright and PyTorch with CPU, CUDA 13.2, or ROCm 10.0.
A project adds its files and training setup target to create a runnable image.
For a first CPU trial:

```console
docker build --build-arg PROFILE=cpu -t skywright:0.1.0-cpu .
docker build -t skywright-trial examples/trial
docker run --rm skywright-trial
```

See [base images](docs/docker.md) for GPU profiles, project dependencies, and host
requirements. These commands build local images; no registry release is published.

Skywright can also stream Parquet datasets from S3-compatible storage. See
[datasets](docs/datasets.md) for configuration and training-loop usage.

## Training entrypoint

A project exposes a setup function that returns its training definition. Skywright
then owns the epoch and step loops.

```python
# project/training.py
import argparse

from skywright import EventListeners, RunContext, Start, Stop, Training, TrainingState


def setup(context: RunContext) -> Training[object]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    options = parser.parse_args(context.argv)

    model, optimizer, loader = build_training(context.accelerator.device)

    def step(batch: object) -> None:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

    listeners = EventListeners()
    listeners.add(Start, lambda event: print("training started"))
    listeners.add(Stop, lambda event: print(f"training {event.outcome}"))
    return Training(
        epochs=options.epochs,
        batches=lambda epoch: loader,
        step=step,
        listeners=listeners,
        state=TrainingState(model, optimizer),
    )
```

Run the same entrypoint locally or in Docker. Arguments after the target belong to
the project.

```console
skywright run project.training:setup --epochs 20
```

Pass a local directory to save after each completed epoch and resume the same run
automatically:

```console
skywright run --checkpoint-dir /checkpoints/trial project.training:setup --epochs 20
```

The directory identifies the run. Reuse it with the same project arguments, epoch
count, PyTorch version, and accelerator type. In Docker, mount it from the host or a
named volume. Skywright restores the model, optimizer, optional scheduler and AMP
scaler, and random generator state. An interrupted epoch starts again from its first
batch.

Each successful `step` call must finish one logical optimizer update, including any
gradient accumulation and scheduler or scaler work. `batches(epoch)` must create a
fresh iterable that reproduces that epoch when setup arguments, seed, and input data
are unchanged.

```dockerfile
ENTRYPOINT ["skywright", "run", "project.training:setup"]
CMD ["--epochs", "20"]
```

Listeners run synchronously in registration order. `Start` is emitted after setup
and validation. Once Start dispatch begins, Skywright emits `Stop` when the run
completes, fails, or is interrupted.

## Development setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12.17 or newer,
then run:

```console
python scripts/bootstrap.py
```

This creates a Python 3.14 environment in `.venv` and installs PyTorch with automatic
backend selection. To choose a backend explicitly, use `--backend`, for example:

```console
python scripts/bootstrap.py --backend cpu
```

You can also set `UV_TORCH_BACKEND`. Run `uv pip install --help` for supported backend
names.

Use `--no-sync` to keep the selected PyTorch build. Plain `uv sync` may replace it.

```console
uv run --no-sync skywright doctor
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ty check
```

## License

MIT
