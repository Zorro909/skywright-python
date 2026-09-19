"""Runtime accelerator inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

AcceleratorKind = Literal["cuda", "rocm", "mps", "cpu"]


@dataclass(frozen=True, slots=True)
class Accelerator:
    """The PyTorch build and compute device visible to this process."""

    kind: AcceleratorKind
    device: torch.device
    available: bool
    torch_version: str
    runtime_version: str | None


def inspect_accelerator() -> Accelerator:
    """Return the accelerator selected by the installed PyTorch build."""
    torch_version = torch.__version__

    if torch.version.hip is not None:
        return Accelerator(
            kind="rocm",
            device=torch.device("cuda"),
            available=torch.cuda.is_available(),
            torch_version=torch_version,
            runtime_version=torch.version.hip,
        )

    if torch.version.cuda is not None:
        return Accelerator(
            kind="cuda",
            device=torch.device("cuda"),
            available=torch.cuda.is_available(),
            torch_version=torch_version,
            runtime_version=torch.version.cuda,
        )

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_built():
        return Accelerator(
            kind="mps",
            device=torch.device("mps"),
            available=mps.is_available(),
            torch_version=torch_version,
            runtime_version=None,
        )

    return Accelerator(
        kind="cpu",
        device=torch.device("cpu"),
        available=True,
        torch_version=torch_version,
        runtime_version=None,
    )
