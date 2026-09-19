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

## Training entrypoint

A project exposes a setup function that returns its training definition. Skywright
then owns the epoch and step loops.

```python
# project/training.py
import argparse

from skywright import EventListeners, RunContext, Start, Stop, Training


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
    )
```

Run the same entrypoint locally or in Docker. Arguments after the target belong to
the project.

```console
skywright run project.training:setup --epochs 20
```

```dockerfile
ENTRYPOINT ["skywright", "run", "project.training:setup"]
CMD ["--epochs", "20"]
```

Listeners run synchronously in registration order. `Start` is emitted after setup
and validation. Once Start dispatch begins, Skywright emits `Stop` when the run
completes, fails, or is interrupted.

## Development setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.11.3 or newer,
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
