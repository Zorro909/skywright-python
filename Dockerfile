# syntax=docker/dockerfile:1
ARG PROFILE=cpu
ARG PYTHON_IMAGE=docker.io/library/python:3.14-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.17

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS base
COPY --from=uv /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    UV_PYTHON=/opt/venv/bin/python \
    UV_NO_CACHE=1 \
    PATH="/opt/venv/bin:${PATH}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends libatomic1 libgomp1 libnuma1 libelf1 libdrm2 \
    && rm -rf /var/lib/apt/lists/* \
    && uv venv --python /usr/local/bin/python /opt/venv

FROM base AS wheel
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN uv build --wheel --out-dir /dist

FROM base AS profile-cpu
ENV UV_TORCH_BACKEND=cpu

FROM base AS profile-cuda
ENV UV_TORCH_BACKEND=cu132 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM base AS profile-rocm
# uv has no ROCm 10 backend selector. Keep project auto-detection from overriding AMD's index.
ENV UV_INDEX=https://stable.repo.amd.com/rocm/whl-next/ \
    UV_NO_CONFIG=1

FROM profile-${PROFILE} AS runtime
ARG PROFILE
ARG TORCH_VERSION=2.13.0
RUN if [ "${PROFILE}" = rocm ]; then \
        uv pip install "torch[device-all]==${TORCH_VERSION}+rocm10.0.0" \
            "rocm[libraries,device-all]==10.0.0"; \
    else \
        uv pip install "torch==${TORCH_VERSION}"; \
    fi
COPY --from=wheel /dist/ /tmp/skywright-wheel/
RUN uv pip install "torch==${TORCH_VERSION}" /tmp/skywright-wheel/*.whl \
    && rm -rf /tmp/skywright-wheel \
    && uv pip check
# Verify the installed build without requiring a GPU on the build host.
RUN python - <<'PY'
import os
from importlib.metadata import version
from pathlib import Path

import torch

profile = os.environ["PROFILE"]
if profile == "cpu":
    assert torch.version.cuda is None and torch.version.hip is None
elif profile == "cuda":
    assert torch.version.cuda == "13.2" and torch.version.hip is None
else:
    # ROCm 10 ships HIP 7.x, so HIP's version does not identify the SDK release.
    assert torch.version.hip is not None and version("rocm") == "10.0.0"

constraints = Path("/opt/skywright/constraints.txt")
constraints.parent.mkdir(parents=True, exist_ok=True)
constraints.write_text("".join(f"{name}=={version(name)}\n" for name in ("skywright", "torch")))
PY
ENV UV_CONSTRAINT=/opt/skywright/constraints.txt \
    PIP_CONSTRAINT=/opt/skywright/constraints.txt
RUN useradd --create-home --uid 1000 skywright \
    && mkdir /workspace \
    && chown skywright:skywright /workspace
WORKDIR /workspace
USER skywright
# -m makes copied project modules importable from the working directory.
ENTRYPOINT ["python", "-m", "skywright.cli", "run"]
CMD []
