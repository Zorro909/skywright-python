# Base images

Build Skywright once, then extend that image for each training project. The base
contains Python 3.14, uv 0.12.17, NumPy, PyTorch 2.13.0, and a wheel built from this
checkout. It has no project code or default training target. Nothing is installed
at container startup.

## Build a profile

Run from the Skywright repository root:

```console
docker build --build-arg PROFILE=cpu -t skywright:0.1.0-cpu .
docker build --build-arg PROFILE=cuda -t skywright:0.1.0-cu132 .
docker build --build-arg PROFILE=rocm -t skywright:0.1.0-rocm10.0 .
```

`PROFILE` defaults to `cpu`. Use Linux x86-64 for these profiles. Tags above are
local names, not published images. CUDA and ROCm tags include their runtime version
because host compatibility depends on it.

| Profile | PyTorch build | Host requirement for training |
| --- | --- | --- |
| `cpu` | `cpu` | CPU, useful for tests and a first trial |
| `cuda` | `cu132` | Supported NVIDIA GPU, CUDA 13.2 compatible driver and NVIDIA Container Toolkit |
| `rocm` | `rocm10.0.0` | Supported AMD GPU and ROCm 10.0 compatible host driver |

All profiles use Debian Bookworm and install the selected PyTorch wheels and their
runtime dependencies. They omit optional development packages. Projects that
compile native extensions need to add matching build dependencies themselves.
Builds select the backend explicitly and check the installed runtime version;
building a GPU image does not require a GPU.

The CPU and CUDA wheels come from PyTorch's
[CPU](https://download.pytorch.org/whl/cpu/torch/) and
[CUDA 13.2](https://download.pytorch.org/whl/cu132/torch/) indexes. ROCm uses
[AMD's stable index](https://stable.repo.amd.com/rocm/whl-next/torch/), including
both PyTorch and ROCm device packages for all architectures AMD ships. This keeps
GPU selection out of the image build, at the cost of a larger image.

The project uses PyTorch 2.13 across all profiles because
[ROCm 10.0 supports PyTorch through 2.13](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
CUDA 13.2 is the newest stable CUDA wheel available for that PyTorch release.

## Add a project

For a project with `training.py` exposing `setup(context)`, the Dockerfile is:

```dockerfile
FROM skywright:0.1.0-cpu
COPY --chown=skywright:skywright . /workspace/
CMD ["training:setup", "--epochs", "2"]
```

The inherited entrypoint runs `python -m skywright.cli run`. The working directory
is `/workspace`, and training runs as the `skywright` user with UID 1000. Python and
Skywright live in `/opt/venv`, outside the project directory. Exclude `.venv`, Git
metadata, credentials, and datasets in the project's `.dockerignore`.

Try the complete example, which trains a linear model on synthetic data:

```console
docker build -t skywright-trial examples/trial
docker run --rm skywright-trial
docker run --rm skywright-trial training:setup --epochs 5
```

Arguments to `docker run` replace `CMD`, so include the setup target when overriding
arguments. A project can fix the target in its own `ENTRYPOINT` if it prefers to
accept only training arguments.

For extra Python dependencies, install them while building the project image:

```dockerfile
FROM skywright:0.1.0-cu132
USER root
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install -r /tmp/requirements.txt && uv pip check
USER skywright
COPY --chown=skywright:skywright . /workspace/
CMD ["training:setup"]
```

For an installable project, `RUN uv pip install .` also works after copying its
source as root. uv installs into `/opt/venv` and inherits the profile's PyTorch
backend. Both uv and pip inherit constraints for the installed Skywright and exact
PyTorch build, including its backend suffix. Incompatible requirements fail
instead of replacing these packages. Use `uv pip install` for additions; `uv sync`
creates or synchronizes a project environment and can discard the base's setup.
The [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
describes backend selection.

The ROCm image uses AMD's index directly because uv has no ROCm 10 backend
selector. It disables uv configuration-file discovery so project settings cannot
restore automatic backend selection. A command-line `--index` replaces the
inherited AMD index, so repeat the AMD URL when adding another index:

```dockerfile
RUN uv pip install \
    --index https://stable.repo.amd.com/rocm/whl-next/ \
    --index https://packages.example.com/simple \
    -r requirements.txt
```

## Run on a GPU

Build the example against the desired base:

```console
docker build --build-arg SKYWRIGHT_IMAGE=skywright:0.1.0-cu132 -t skywright-trial:cuda examples/trial
docker run --rm --gpus all --shm-size=1g skywright-trial:cuda

docker build --build-arg SKYWRIGHT_IMAGE=skywright:0.1.0-rocm10.0 -t skywright-trial:rocm examples/trial
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --shm-size=1g skywright-trial:rocm
```

For ROCm, select the render device on your host and grant its numeric group to the
container user. Host drivers and GPU access remain host setup steps. See the
[NVIDIA container instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)
and [AMD container instructions](https://rocmdocs.amd.com/en/latest/install/docker-containers.html).
GPU profiles keep their GPU device selection when hardware is unavailable;
training then fails instead of silently switching to CPU.

To inspect an image without starting a project:

```console
docker run --rm --entrypoint skywright skywright:0.1.0-cu132 doctor
```

`doctor` reports the installed backend and whether a device is available. Its exit
code does not assert GPU availability. For ROCm, its runtime version is the HIP
component version, which differs from the ROCm SDK release. A successful example
training run verifies allocation, forward/backward execution, and an optimizer
step on the selected device.

## Rebuilds

`TORCH_VERSION` can select another compatible PyTorch release when building the
base. `PYTHON_IMAGE` and `UV_IMAGE` can pin upstream image digests. The defaults
pin the PyTorch and uv releases but do not lock every dependency or OS package.
For repeatable deployments, build and validate a base once, publish it to your
registry, and pin its digest in project Dockerfiles.
