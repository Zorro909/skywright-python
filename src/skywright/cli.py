"""Command-line entry point for Skywright."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import cast

from skywright.accelerator import inspect_accelerator
from skywright.checkpointing import CheckpointError
from skywright.training import Setup, run


class _TargetError(ValueError):
    pass


def _doctor() -> int:
    accelerator = inspect_accelerator()
    print(f"PyTorch: {accelerator.torch_version}")
    print(f"Backend: {accelerator.kind}")
    print(f"Device: {accelerator.device}")
    print(f"Available: {'yes' if accelerator.available else 'no'}")
    if accelerator.runtime_version is not None:
        print(f"Runtime: {accelerator.runtime_version}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skywright")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="show the installed PyTorch accelerator")
    run_parser = subcommands.add_parser("run", help="run a project's training setup")
    run_parser.add_argument(
        "--checkpoint-dir",
        help="save and resume training state in this local directory",
    )
    run_parser.add_argument("target", help="project setup as package.module:function")
    run_parser.add_argument("project_arguments", nargs=argparse.REMAINDER)
    return parser


def _load_setup(target: str) -> Setup[object]:
    module_name, separator, function_name = target.partition(":")
    valid_module = module_name and all(part.isidentifier() for part in module_name.split("."))
    valid_target = (
        separator and ":" not in function_name and valid_module and function_name.isidentifier()
    )
    if not valid_target:
        raise _TargetError(f"invalid target {target!r}; expected package.module:function")

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            raise _TargetError(f"could not import target module {module_name!r}") from None
        raise

    try:
        setup = getattr(module, function_name)
    except AttributeError:
        raise _TargetError(
            f"target module {module_name!r} has no function {function_name!r}"
        ) from None
    if not callable(setup):
        raise _TargetError(f"target {target!r} is not callable")
    return cast("Setup[object]", setup)


def main(argv: list[str] | None = None) -> int:
    """Run the Skywright command-line interface."""
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    if args.command == "run":
        try:
            setup = _load_setup(args.target)
        except _TargetError as error:
            print(f"skywright: {error}", file=sys.stderr)
            return 2
        try:
            return run(
                setup,
                args.project_arguments,
                checkpoint_dir=args.checkpoint_dir,
            )
        except CheckpointError as error:
            print(f"skywright: {error}", file=sys.stderr)
            return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
