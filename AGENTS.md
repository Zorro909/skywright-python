# Working on Skywright

Read [README.md](README.md) before changing the runtime or public API. It defines the intended
container lifecycle and distinguishes planned behavior from what exists today.

## Design constraints

- Apply KISS. Choose the smallest implementation that handles the current use case correctly.
  For added complexity, explain the concrete benefit to a library user or maintainer and why
  a simpler approach falls short. Future flexibility alone is insufficient.
- Keep ownership clear. Skywright owns workspace setup and the epoch/step lifecycle. Projects
  own models, data, and training semantics. Keep deployment and backend details out of the
  training loop.
- Prefer direct functions and explicit run options. Introduce shared abstractions when actual
  callers need them. Keep dependencies tied to a demonstrated need.
- Keep image construction separate from runtime startup. Resolve dependencies during the build;
  validate the installed environment when the container starts.
- Consider startup cost and work added per training step. Measure performance claims against
  the affected path before adding an optimization.

## Changes and verification

- Check affected entrypoints and supported backends when changing shared behavior. State gaps
  in hardware verification; CPU tests do not establish GPU compatibility.
- Test observable behavior and failure cases with the smallest useful checks. Use temporary
  workspaces and small fixtures; preserve datasets, credentials, checkpoints, and live runs.
- Run the relevant pytest tests for behavior changes, plus Ruff and ty for Python changes.
  Use the development commands in README.md, including `uv run --no-sync` to preserve the
  selected PyTorch build. Broaden checks when the change warrants it.
- For lifecycle changes, cover startup failures, training exceptions, and interruption where
  applicable. Preserve the original error and a failed process exit when a run fails.

## Documentation

- Update guidance when behavior changes. Label proposals as planned until they work, and use
  runnable examples only for implemented APIs.
- Keep user instructions in README.md while the project is small. Add separate documents for
  durable decisions or procedures that need them. Record local reasoning beside the code.
- Keep temporary plans and investigation notes outside the repository.

The simplicity, verification, and documentation guidance adapts ideas from
[T3 Code's AGENTS.md](https://github.com/pingdotgg/t3code/blob/main/AGENTS.md) to this library.
