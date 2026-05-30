"""Fourier-feature network with skip connections.

Port of the strongest family from mandelbrotnn: lift (real, imag) into Fourier
features [sin(k x), cos(k x), x] and feed a SkipConn MLP (input + all-prior-layer
skips concatenated at every layer). The Mandelbrot target is quasi-periodic, so the
Fourier basis is a strong fit. Device-agnostic (no hardcoded .cuda()).
"""

import torch
import torch.nn as nn

from harness import groundtruth as gt
from harness.interface import FitContext, TorchSolution


class CenteredLinearMap(nn.Module):
    """Affine remap of (real, imag) from the view window into ~[-pi, pi]^2,
    a good range for Fourier features."""

    def __init__(self):
        super().__init__()
        import math
        sx = 2 * math.pi / (gt.XMAX - gt.XMIN)
        sy = 2 * math.pi / (gt.YMAX - gt.YMIN)
        self.register_buffer("scale", torch.tensor([sx, sy]))
        self.register_buffer("shift", torch.tensor([
            -math.pi - gt.XMIN * sx, -math.pi - gt.YMIN * sy]))

    def forward(self, x):
        return x * self.scale + self.shift


class SkipConn(nn.Module):
    def __init__(self, hidden, layers, in_size, out_size=1, activation=nn.GELU):
        super().__init__()
        self.inLayer = nn.Linear(in_size, hidden)
        self.act = activation()
        hid = []
        for i in range(layers):
            cin = hidden * 2 + in_size if i > 0 else hidden + in_size
            hid.append(nn.Linear(cin, hidden))
        self.hidden = nn.ModuleList(hid)
        self.outLayer = nn.Linear(hidden * 2 + in_size, out_size)

    def forward(self, x):
        cur = self.act(self.inLayer(x))
        prev = torch.zeros_like(cur[:, :0])
        for layer in self.hidden:
            combined = torch.cat([cur, prev, x], dim=1)
            prev = cur
            cur = self.act(layer(combined))
        y = self.outLayer(torch.cat([cur, prev, x], dim=1))
        return (torch.tanh(y) + 1) / 2


class FourierNet(nn.Module):
    def __init__(self, order=16, hidden=256, layers=8):
        super().__init__()
        self.order = order
        self.linmap = CenteredLinearMap()
        in_size = order * 2 * 2 + 2  # (sin,cos) x order x 2 coords + raw 2
        self.inner = SkipConn(hidden, layers, in_size)
        self.register_buffer("orders", torch.arange(1, order + 1).float())

    def forward(self, x):
        x = self.linmap(x)
        xe = x.unsqueeze(-1)  # (N, 2, 1)
        feats = torch.cat([torch.sin(self.orders * xe),
                           torch.cos(self.orders * xe), xe], dim=-1)
        feats = feats.view(x.shape[0], -1)
        return self.inner(feats)


class Fourier(TorchSolution):
    name = "fourier"
    description = "Fourier features (order 16) + SkipConn MLP (256x8), Adam, uniform sampling"

    def __init__(self):
        self.model = FourierNet(order=16, hidden=256, layers=8).to(self.device)

    def fit(self, ctx: FitContext) -> None:
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
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
            if step % 100 == 0:
                ctx.log(f"step {step} loss {loss.item():.5f} (left {ctx.time_left():.0f}s)")


SOLUTION = Fourier()
