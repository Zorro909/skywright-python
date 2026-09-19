"""A small training run with synthetic data and no downloads."""

import argparse

import torch

from skywright import EventListeners, RunContext, Stop, Training


def setup(context: RunContext) -> Training[torch.Tensor]:
    """Fit y = 2x on the image's selected device."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    options = parser.parse_args(context.argv)
    device = context.accelerator.device
    model = torch.nn.Linear(1, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.linspace(-1, 1, 32, device=device).reshape(-1, 1)

    def step(batch: torch.Tensor) -> None:
        optimizer.zero_grad()
        loss = (model(batch) - 2 * batch).square().mean()
        loss.backward()
        optimizer.step()
        print(f"loss={loss.item():.6f}")

    listeners = EventListeners()
    listeners.add(Stop, lambda event: print(f"training {event.outcome}"))
    return Training(
        epochs=options.epochs,
        batches=lambda epoch: (inputs,),
        step=step,
        listeners=listeners,
    )
