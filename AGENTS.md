# Skywright

Skywright aims to handle deployment infrastructure around PyTorch training jobs,
mainly in Docker. It is still early in development.

## Taste

Keep it simple. Understand the problem before choosing the design.

- Apply KISS. Extra complexity needs a clear benefit for users or maintainers.
- Build for the task at hand. Leave future features for when they are needed.
- Simplify awkward code when you touch it. Existing complexity does not justify more.
- Keep infrastructure details out of project training code.
- Care about startup time and overhead during training. Measure before optimizing.

## Working on changes

- Test behavior with focused tests. Run Ruff and ty for Python changes.
- Use `uv run --no-sync` so checks keep the selected PyTorch build.
- Keep docs short and useful. Explain what works today; avoid speculative API designs.
- Keep comments close to the code they explain. Skip narration of obvious code.
- Treat these as defaults. The maintainer's instructions take priority.
