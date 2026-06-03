"""Baseline: a plain coordinate MLP.

This is the canonical template for a solution module. Copy it to start a new approach.
It is intentionally simple so it's a fair, easy-to-beat floor on the leaderboard.

A solution module must expose `SOLUTION` (an instance) or `build()` (a factory).
"""

import torch
import torch.nn as nn

from harness.interface import FitContext, TorchSolution


class MLP(nn.Module):
    def __init__(self, hidden=256, layers=6):
        super().__init__()
        dims = [2] + [hidden] * layers + [1]
        blocks = []
        for i in range(len(dims) - 2):
            blocks += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        blocks += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return (torch.tanh(self.net(x)) + 1) / 2  # -> [0, 1]


class BaselineMLP(TorchSolution):
    name = "baseline_mlp"
    description = "plain GELU MLP (256x6), uniform sampling, Adam"

    def __init__(self):
        self.model = MLP(hidden=256, layers=6).to(self.device)

    def fit(self, ctx: FitContext) -> None:
        opt = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        self.model.train()
        batch = 65_536
        step = 0
        while not ctx.expired():
            coords, targets = ctx.sample(batch)
            pred = self.model(coords).reshape(-1)
            loss = torch.mean((pred - targets) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                ctx.log(f"step {step} loss {loss.item():.5f} (left {ctx.time_left():.0f}s)")


SOLUTION = BaselineMLP()
