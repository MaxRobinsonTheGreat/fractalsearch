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
import torch.nn.functional as F

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
        self._res = res                      # plain python ints -> no per-forward CUDA sync
        self.register_buffer("res", torch.tensor(res, dtype=torch.long))
        # One packed embedding table for all levels. Each level keeps a static offset into
        # this table; the four bilinear corner lookups are reduced with embedding_bag.
        self.sizes = []
        offsets = []
        self._direct = []                    # whether level indexes directly (no hash)
        total = 0
        for l in range(n_levels):
            N = res[l]
            size = min(N * N, self.T)
            offsets.append(total)
            self.sizes.append(size)
            self._direct.append(N * N <= size)
            total += size
        self.table = nn.Parameter(torch.empty(total, n_features).uniform_(-1e-4, 1e-4))
        self._offsets = offsets
        self.out_dim = n_levels * n_features

    def _idx(self, ix, iy, size, N, direct, offset):
        # direct index when the grid fits in the table, else spatial hash
        if direct:
            return (iy * N + ix) % size + offset
        h = (ix * _PRIMES[0]) ^ (iy * _PRIMES[1])
        return h % size + offset

    def forward(self, x):
        # x in [0,1]^2.  Uses python-int res (self._res) to avoid a CUDA sync per level.
        feats = []
        for l in range(self.n_levels):
            N = self._res[l]
            size = self.sizes[l]
            direct = self._direct[l]
            offset = self._offsets[l]
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
            corner_idx = torch.stack([
                self._idx(ix0, iy0, size, N, direct, offset),
                self._idx(ix1, iy0, size, N, direct, offset),
                self._idx(ix0, iy1, size, N, direct, offset),
                self._idx(ix1, iy1, size, N, direct, offset),
            ], dim=1)
            weights = torch.cat([
                (1 - fx) * (1 - fy),
                fx * (1 - fy),
                (1 - fx) * fy,
                fx * fy,
            ], dim=1)
            feats.append(F.embedding_bag(corner_idx, self.table, per_sample_weights=weights, mode="sum"))
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
    name = "hashgrid_bag"
    description = ("champion config with packed hash table + embedding_bag bilinear lookup "
                   "(one fused weighted reduction per level instead of four fancy-index gathers)")

    TBL_LR0 = 6e-1
    MLP_LR0 = 5e-3
    LR_MIN_FRAC = 0.01
    WARMUP = 0.08

    def __init__(self):
        self.model = HashGridNet(n_levels=12, n_features=2, log2_T=24,
                                 n_min=16, n_max=32768, hidden=256, mlp_layers=4).to(self.device)

    def _set_lr(self, opt, frac):
        import math
        if frac < self.WARMUP:
            mult = frac / self.WARMUP
        else:
            f2 = (frac - self.WARMUP) / (1 - self.WARMUP)
            mult = self.LR_MIN_FRAC + 0.5 * (1 - self.LR_MIN_FRAC) * (1 + math.cos(math.pi * f2))
        for g in opt.param_groups:
            g["lr"] = g["peak_lr"] * mult

    def fit(self, ctx: FitContext) -> None:
        opt = torch.optim.Adam([
            {"params": self.model.enc.parameters(), "peak_lr": self.TBL_LR0, "lr": self.TBL_LR0},
            {"params": self.model.net.parameters(), "peak_lr": self.MLP_LR0, "lr": self.MLP_LR0},
        ], betas=(0.9, 0.99), eps=1e-15)
        self.model.train()
        batch = 786_432
        pool_mult = 6
        n_hard = 3 * batch // 4
        budget = ctx.time_budget_s
        step = 0
        # bf16 autocast: fp32 exponent range so no GradScaler, grid gather/interp stay safe.
        ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        while not ctx.expired():
            self._set_lr(opt, min(1.0, ctx.elapsed() / budget))
            with torch.no_grad(), ac:
                pcoords, ptargets = ctx.sample(batch * pool_mult)
                ppred = self.model(pcoords).reshape(-1)
                perr = (ppred.float() - ptargets).abs() + 1e-6
            hard_idx = torch.multinomial(perr, n_hard, replacement=False)
            unif_idx = torch.randint(0, pcoords.shape[0], (batch - n_hard,), device=pcoords.device)
            idx = torch.cat([hard_idx, unif_idx])
            coords, targets = pcoords[idx], ptargets[idx]

            with ac:
                pred = self.model(coords).reshape(-1)
                loss = torch.mean((pred.float() - targets) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                ctx.log(f"step {step} loss {loss.item():.5f} (left {ctx.time_left():.0f}s)")


SOLUTION = HashGridSolution()
