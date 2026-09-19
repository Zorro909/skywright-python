# Skywright

Skywright aims to handle deployment infrastructure around PyTorch training jobs,
mainly in Docker. It is still early in development.

These are defaults. The maintainer's instructions take priority.

## Taste

Keep it simple. Understand the problem before choosing the design.

- Apply KISS. Extra complexity needs a clear benefit for users or maintainers.
- Build for the task at hand. Leave future features for when they are needed.
- Simplify awkward code when you touch it. Existing complexity does not justify more.
- Keep infrastructure details out of project training code.
- Type public interfaces. Use a known type instead of reaching for `Any`.
- Care about startup time and overhead during training. Measure before optimizing.

## Drafting

- Do all transient design and specification work in `.scratch/`. Complete the draft
  before starting implementation, and keep the two phases separate.

## Verification

- Consider library use both inside and outside Docker, with CPU, CUDA, and ROCm.
  CPU is mainly for tests. Say which paths you checked and which remain unverified.
- Write focused tests that catch wrong behavior. Repeating the implementation or
  checking that one function calls another proves little.
- Run Ruff and ty for Python changes.
- Use `uv run --no-sync` so checks keep the selected PyTorch build.

## Documentation

- Keep docs short and useful. Explain what works today; avoid speculative API designs.
- Record reasons the code cannot explain. Skip catalogs of functions and fields.
- Rewrite or remove stale guidance. Don't append a second account of the same thing.
- Keep comments close to the code they explain. Skip narration of obvious code.

## Pull requests

- Keep each PR about one problem. If unrelated work creeps in, push back, even when
  the maintainer asks for it. Explain why it belongs in a separate PR.
- Write a plain title and a short description: the problem, the fix, and how you
  checked it.
