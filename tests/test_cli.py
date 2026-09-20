from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


def test_run_command_loads_setup_and_forwards_project_arguments(tmp_path: Path) -> None:
    project = tmp_path / "sample_training.py"
    project.write_text(
        textwrap.dedent(
            """
            from skywright import EventListeners, Start, Stop, Training


            def setup(context):
                print(f"setup {context.argv}")
                listeners = EventListeners()
                listeners.add(Start, lambda event: print("start"))
                listeners.add(Stop, lambda event: print(f"stop {event.outcome}"))
                return Training(
                    epochs=2,
                    batches=lambda epoch: (epoch,),
                    step=lambda batch: print(f"step {batch}"),
                    listeners=listeners,
                )
            """
        )
    )
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(SOURCE_ROOT), str(tmp_path)))
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "skywright.cli",
            "run",
            "sample_training:setup",
            "--epochs",
            "2",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "setup ('--epochs', '2')",
        "start",
        "step 0",
        "step 1",
        "stop completed",
    ]


def test_run_command_dispatches_stop_when_sigterm_interrupts_training(tmp_path: Path) -> None:
    project = tmp_path / "terminating_training.py"
    project.write_text(
        textwrap.dedent(
            """
            import os
            import signal

            from skywright import EventListeners, Start, Stop, Training


            def setup(context):
                listeners = EventListeners()
                listeners.add(Start, lambda event: print("start", flush=True))
                listeners.add(
                    Stop,
                    lambda event: print(f"stop {event.outcome}", flush=True),
                )
                return Training(
                    epochs=1,
                    batches=lambda epoch: (None,),
                    step=lambda batch: os.kill(os.getpid(), signal.SIGTERM),
                    listeners=listeners,
                )
            """
        )
    )
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(SOURCE_ROOT), str(tmp_path)))
    }

    result = subprocess.run(
        [sys.executable, "-m", "skywright.cli", "run", "terminating_training:setup"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 128 + signal.SIGTERM
    assert result.stdout.splitlines() == ["start", "stop interrupted"]


def test_run_command_uses_checkpoint_directory_and_completed_run_is_idempotent(
    tmp_path: Path,
) -> None:
    project = tmp_path / "checkpointed_training.py"
    project.write_text(
        textwrap.dedent(
            """
            import torch

            from skywright import EventListeners, Start, Stop, Training, TrainingState


            def setup(context):
                print(f"setup {context.argv}")
                model = torch.nn.Linear(1, 1)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                listeners = EventListeners()
                listeners.add(Start, lambda event: print("start"))
                listeners.add(Stop, lambda event: print(f"stop {event.outcome}"))
                return Training(
                    epochs=1,
                    batches=lambda epoch: (epoch,),
                    step=lambda batch: print(f"step {batch}"),
                    listeners=listeners,
                    state=TrainingState(model, optimizer),
                )
            """
        )
    )
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(SOURCE_ROOT), str(tmp_path)))
    }
    command = [
        sys.executable,
        "-m",
        "skywright.cli",
        "run",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "checkpointed_training:setup",
        "--learning-rate",
        "0.1",
    ]

    first = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0
    assert first.stdout.splitlines() == [
        "setup ('--learning-rate', '0.1')",
        "start",
        "step 0",
        "stop completed",
    ]
    assert second.returncode == 0
    assert second.stdout.splitlines() == [
        "setup ('--learning-rate', '0.1')",
        "start",
        "stop completed",
    ]
