from __future__ import annotations

import torch

from skywright import inspect_accelerator


def test_inspect_accelerator_matches_installed_torch() -> None:
    accelerator = inspect_accelerator()

    assert accelerator.torch_version == torch.__version__
    assert accelerator.kind in {"cuda", "rocm", "mps", "cpu"}
    assert accelerator.device.type in {"cuda", "mps", "cpu"}


def test_accelerator_is_available_on_cpu_build() -> None:
    accelerator = inspect_accelerator()

    if accelerator.kind == "cpu":
        assert accelerator.available
