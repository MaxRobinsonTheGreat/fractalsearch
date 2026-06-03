"""Fourier-feature MLP.

The target (periodic log-distance) is dominated by HIGH-FREQUENCY content near the
set boundary. Plain coordinate MLPs have a strong spectral bias and cannot represent
those frequencies. Random Fourier features (Tancik et al. 2020, "Fourier Features Let
Networks Learn High Frequency Functions in Low Dimensional Domains") lift the 2D input
into sin/cos of random projections, letting a standard MLP fit high frequencies.

Design knobs that matter:
- mapping_size: number of random frequencies (feature dim = 2*mapping_size).
- sigma: scale of the gaussian frequency matrix B ~ N(0, sigma^2). This is the single
  most important hyperparameter — too low underfits detail, too high overfits noise.
- coords are normalized to ~[-1, 1] before the Fourier lift so sigma is interpretable.
"""

import torch
import torch.nn as nn

from harness import groundtruth as gt
from harness.interface import FitContext, TorchSolution

# Normalization constants for mapping the view window to ~[-1, 1].
_CX = (gt.XMIN + gt.XMAX) / 2.0
_CY = (gt.YMIN + gt.YMAX) / 2.0
_SX = (gt.XMAX - gt.XMIN) / 2.0
_SY = (gt.YMAX - gt.YMIN) / 2.0


class FourierMLP(nn.Module):
    def __init__(self, mapping_size=256, sigma=8.0, hidden=512, layers=6):
        super().__init__()
        B = torch.randn(2, mapping_size) * sigma
        self.register_buffer("B", B)
        in_dim = 2 * mapping_size
        dims = [in_dim] + [hidden] * layers + [1]
        blocks = []
        for i in range(len(dims) - 2):
            blocks += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        blocks += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*blocks)

    def _norm(self, x):
        nx = (x[:, 0:1] - _CX) / _SX
        ny = (x[:, 1:2] - _CY) / _SY
        return torch.cat([nx, ny], dim=1)

    def forward(self, x):
        x = self._norm(x)
        proj = 2.0 * torch.pi * (x @ self.B)
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        return (torch.tanh(self.net(feats)) + 1) / 2  # -> [0, 1]


class FourierSolution(TorchSolution):
    name = "fourier_mlp"
    description = "random Fourier features (256, sigma=8) + GELU MLP 512x6, uniform sampling, Adam"

    def __init__(self):
        self.model = FourierMLP(mapping_size=256, sigma=8.0, hidden=512, layers=6).to(self.device)

    def fit(self, ctx: FitContext) -> None:
        opt = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20000, eta_min=1e-5)
        self.model.train()
        batch = 131_072
        step = 0
        while not ctx.expired():
            coords, targets = ctx.sample(batch)
            pred = self.model(coords).reshape(-1)
            loss = torch.mean((pred - targets) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                ctx.log(f"step {step} loss {loss.item():.5f} (left {ctx.time_left():.0f}s)")


SOLUTION = FourierSolution()
