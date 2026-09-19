# Skywright

Skywright aims to make PyTorch training jobs simpler, faster, and safer to deploy,
mainly inside Docker containers.

The idea is that Skywright is the container's entrypoint. It sets up the workspace,
lets your AI project configure options during bootstrap, then runs an epoch → step
training loop. Your project supplies the training code. Skywright handles the
infrastructure around it.

The project is in early development. Only development setup and accelerator
diagnostics exist today. The training runtime is not implemented yet.

We follow KISS. Added complexity must have a clear benefit for library users or
maintainers.

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
