"""Sanity-check solution: predict uniform-random noise. No training, no torch modules.

Establishes a floor on the leaderboard and exercises the harness with a solution that
is not an nn.Module. Subclasses Solution directly (save/load left unimplemented — the
evaluator tolerates that).
"""

import torch

from harness.interface import FitContext, Solution


class RandomGuess(Solution):
    name = "random"
    description = "uniform random noise in [0,1], no training (sanity floor)"

    def fit(self, ctx: FitContext) -> None:
        self.device = ctx.device
        self.gen = torch.Generator(device=ctx.device).manual_seed(0)

    def predict(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.rand(coords.shape[0], generator=self.gen, device=coords.device)


SOLUTION = RandomGuess()
