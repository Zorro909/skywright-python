# Datasets

Skywright reads Parquet files from S3-compatible storage through Hugging Face
Datasets. S3 support is part of the standard Skywright installation.

Skywright reads bindings from `/etc/skywright/datasets.toml`. Set
`SKYWRIGHT_DATASETS_CONFIG` to use another path, which is useful for local runs.
The file maps each logical dataset to an explicit list of objects:

```toml
[storage]
endpoint_url = "https://objects.example.com" # omit for AWS S3
region = "us-east-1"

[[datasets]]
name = "images"
revision = "2026-09-01"
split = "train"
files = [
    "s3://training-data/images/2026-09-01/train-000.parquet",
    "s3://training-data/images/2026-09-01/train-001.parquet",
]
```

Credentials come from the normal AWS environment, profile, or workload role.
Keep credentials out of this file. Revisions are immutable: publish changed data
under a new revision and update the binding.

Declare and open the dataset during project setup. In this example,
`train_step` is the project's training function:

```python
from collections.abc import Iterable

from torch.utils.data import DataLoader

from skywright import Dataset, RunContext, Training

TRAIN = Dataset("images", revision="2026-09-01")


def setup(context: RunContext) -> Training[dict[str, object]]:
    reader = TRAIN.open(shuffle=True)
    loader = DataLoader(
        reader,
        batch_size=64,
        num_workers=4,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )

    def batches(epoch: int) -> Iterable[dict[str, object]]:
        reader.set_epoch(epoch)
        return loader

    return Training(epochs=10, batches=batches, step=train_step)
```

Do not add a sampler or DataLoader shuffle. Hugging Face handles worker sharding
and Skywright splits files across initialized distributed ranks. The number of
Parquet files must divide the distributed world size. Files assigned to different
ranks should contain enough rows to produce the same number of batches.

Use `spawn` or `forkserver` with multiple workers because s3fs does not support
`fork`. For stable file order, open with `shuffle=False` and call
`reader.set_epoch(0)` before every pass. Streaming does not keep a bounded local
copy of read Parquet data, so later epochs may read objects again.

`DatasetOpenStarted`, `DatasetOpened`, and `DatasetOpenFailed` use the same
`EventListeners` interface as training events. `DatasetOpened` means the reader
was created; later files can still fail during iteration, which causes the normal
`Stop(FAILED)` event during a training run.
