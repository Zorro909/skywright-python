"""Command-line entry point for Skywright."""

from __future__ import annotations

import argparse

from skywright.accelerator import inspect_accelerator


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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Skywright command-line interface."""
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
