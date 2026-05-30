"""'Cheating' solution: return the exact ground-truth value. No model, no training.

This is the theoretical ceiling — it should score MSE ~0 (PSNR inf), because nothing can
fit the target better than the target itself. Useful to confirm the harness and metric
are wired correctly. Not a real research entry (it just calls the ground truth).
"""

import torch

from harness import groundtruth as gt
from harness.interface import FitContext, Solution


class Cheat(Solution):
    name = "cheat_groundtruth"
    description = "returns exact ground truth (upper bound; not a real solution)"

    def fit(self, ctx: FitContext) -> None:
        pass

    def predict(self, coords: torch.Tensor) -> torch.Tensor:
        return gt.mandelbrot(coords)


SOLUTION = Cheat()
