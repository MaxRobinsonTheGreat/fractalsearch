"""Multiresolution hash-grid encoding (Instant-NGP style) + small MLP.

The target is a dense 2D spatial field with structure at every scale. Instant-NGP
(Mueller et al. 2022, "Instant Neural Graphics Primitives with a Multiresolution Hash
Encoding") is the SOTA way to fit exactly this: a stack of L learnable feature grids at
geometrically-spaced resolutions, each entry looked up via (optionally hashed) bilinear
interpolation, concatenated and fed to a tiny MLP. The grids carry the spatial detail;
the MLP just decodes. It's a universal approximator (learnable interpolant + MLP).

2D specifics:
- Each level l has resolution N_l = N_min * b^l, b = (N_max/N_min)^(1/(L-1)).
- Feature table size per level = min(N_l^2, T). Coarse levels index directly; fine
  levels collide through a spatial hash (xor of per-axis primes).
- Bilinear interp of the 4 cell corners. coords normalized to [0,1].
"""

import torch
import torch.nn as nn

from harness import groundtruth as gt
from harness.interface import FitContext, TorchSolution

_PRIMES = (1, 2654435761)


class HashGridEncoding(nn.Module):
    def __init__(self, n_levels=16, n_features=2, log2_T=20,
                 n_min=16, n_max=4096):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.T = 1 << log2_T
        b = (n_max / n_min) ** (1.0 / (n_levels - 1))
        res = [int(round(n_min * b ** l)) for l in range(n_levels)]
        self.register_buffer("res", torch.tensor(res, dtype=torch.long))
        # one embedding table per level
        self.tables = nn.ParameterList()
        self.sizes = []
        for l in range(n_levels):
            N = res[l]
            size = min(N * N, self.T)
            self.sizes.append(size)
            t = nn.Parameter(torch.empty(size, n_features).uniform_(-1e-4, 1e-4))
            self.tables.append(t)
        self.out_dim = n_levels * n_features

    def _hash(self, ix, iy, size, N):
        # direct index when the grid fits in the table, else spatial hash
        if N * N <= size:
            return (iy * N + ix) % size
        h = (ix * _PRIMES[0]) ^ (iy * _PRIMES[1])
        return h % size

    def forward(self, x):
        # x in [0,1]^2
        feats = []
        for l in range(self.n_levels):
            N = int(self.res[l].item())
            size = self.sizes[l]
            xs = x * (N - 1)
            x0 = torch.floor(xs).long()
            xf = xs - x0.float()
            ix0, iy0 = x0[:, 0], x0[:, 1]
            ix1 = (ix0 + 1).clamp(max=N - 1)
            iy1 = (iy0 + 1).clamp(max=N - 1)
            ix0 = ix0.clamp(0, N - 1)
            iy0 = iy0.clamp(0, N - 1)
            fx = xf[:, 0:1]
            fy = xf[:, 1:2]
            tbl = self.tables[l]
            c00 = tbl[self._hash(ix0, iy0, size, N)]
            c10 = tbl[self._hash(ix1, iy0, size, N)]
            c01 = tbl[self._hash(ix0, iy1, size, N)]
            c11 = tbl[self._hash(ix1, iy1, size, N)]
            c0 = c00 * (1 - fx) + c10 * fx
            c1 = c01 * (1 - fx) + c11 * fx
            feats.append(c0 * (1 - fy) + c1 * fy)
        return torch.cat(feats, dim=1)


# Normalization to [0,1] over the view window.
_CX, _CY = gt.XMIN, gt.YMIN
_W, _H = (gt.XMAX - gt.XMIN), (gt.YMAX - gt.YMIN)


class HashGridNet(nn.Module):
    def __init__(self, n_levels=16, n_features=2, log2_T=20,
                 n_min=16, n_max=4096, hidden=128, mlp_layers=3):
        super().__init__()
        self.enc = HashGridEncoding(n_levels, n_features, log2_T, n_min, n_max)
        dims = [self.enc.out_dim] + [hidden] * (mlp_layers - 1) + [1]
        blocks = []
        for i in range(len(dims) - 2):
            blocks += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        blocks += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*blocks)

    def _norm(self, x):
        nx = (x[:, 0:1] - _CX) / _W
        ny = (x[:, 1:2] - _CY) / _H
        return torch.cat([nx, ny], dim=1).clamp(0.0, 1.0)

    def forward(self, x):
        e = self.enc(self._norm(x))
        return (torch.tanh(self.net(e)) + 1) / 2


class HashGridSolution(TorchSolution):
    name = "hashgrid_bank"
    description = ("hash grid (24 lvl, F=2, T=2^21, Nmax=8192) + MLP 256x4, "
                  "precomputed 32M-point data bank (no per-step ground-truth recompute)")

    def __init__(self):
        self.model = HashGridNet(n_levels=24, n_features=2, log2_T=21,
                                 n_min=16, n_max=8192, hidden=256, mlp_layers=4).to(self.device)

    def _build_bank(self, ctx, n_total, chunk=4_000_000):
        """Precompute a large uniform (coords, targets) bank once. The per-step
        Mandelbrot recompute in ctx.sample() is the throughput bottleneck; banking it
        turns each training step into a pure GPU gather + forward/backward."""
        cs, ts = [], []
        done = 0
        while done < n_total:
            n = min(chunk, n_total - done)
            c, t = ctx.sample(n)
            cs.append(c)
            ts.append(t)
            done += n
        coords = torch.cat(cs)
        targets = torch.cat(ts)
        ctx.log(f"bank built: {coords.shape[0]} pts in {ctx.elapsed():.1f}s")
        return coords, targets

    def fit(self, ctx: FitContext) -> None:
        coords_bank, targets_bank = self._build_bank(ctx, n_total=32_000_000)
        N = coords_bank.shape[0]
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-2, betas=(0.9, 0.99), eps=1e-15)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30000, eta_min=1e-4)
        self.model.train()
        batch = 262_144
        step = 0
        while not ctx.expired():
            idx = torch.randint(0, N, (batch,), device=coords_bank.device)
            coords, targets = coords_bank[idx], targets_bank[idx]
            pred = self.model(coords).reshape(-1)
            loss = torch.mean((pred - targets) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                ctx.log(f"step {step} loss {loss.item():.5f} (left {ctx.time_left():.0f}s)")


SOLUTION = HashGridSolution()
