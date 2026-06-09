"""Champion recipe + PACKED single-gather hash-grid encoder.

Hypothesis (notebook bet #1): the champion encoder is a python loop over 12 levels with
4 fancy-index lookups each = ~48 tiny gather kernels per forward (x3 forwards/step).
That is launch-overhead-bound, not purely bandwidth-bound. Pack all level tables into
ONE contiguous parameter and compute every (level, corner) index vectorized, so the
whole encoding is a single [B, L, 4] gather + one weighted sum. If this raises steps
meaningfully (~600 -> 1000+), the undertrained-grid story is confirmed and the same
recipe should score below 0.000335.

(hashgrid_bag tried embedding_bag for this and lost; this is a plain gather instead.)
Everything else is identical to champion.py.
"""

import torch
import torch.nn as nn

from harness import groundtruth as gt
from harness.interface import FitContext, TorchSolution

_PRIMES = (1, 2654435761)


class PackedHashGridEncoding(nn.Module):
    def __init__(self, n_levels=16, n_features=2, log2_T=20,
                 n_min=16, n_max=4096):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.T = 1 << log2_T
        b = (n_max / n_min) ** (1.0 / (n_levels - 1))
        res = [int(round(n_min * b ** l)) for l in range(n_levels)]
        sizes = [min(N * N, self.T) for N in res]
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)
        # one packed table for every level
        self.table = nn.Parameter(
            torch.empty(offsets[-1], n_features).uniform_(-1e-4, 1e-4))
        self.register_buffer("res", torch.tensor(res, dtype=torch.long))        # [L]
        self.register_buffer("sizes", torch.tensor(sizes, dtype=torch.long))    # [L]
        self.register_buffer("offsets", torch.tensor(offsets[:-1], dtype=torch.long))
        self.register_buffer("direct", torch.tensor(
            [N * N <= s for N, s in zip(res, sizes)], dtype=torch.bool))         # [L]
        self.out_dim = n_levels * n_features

    def forward(self, x):
        # x in [0,1]^2 -> [B, L*F]; ONE gather for all levels and corners.
        L = self.n_levels
        res1 = (self.res - 1).view(1, L)                       # [1,L]
        xs = x.unsqueeze(2) * res1.unsqueeze(1)                # [B,2,L] (xs[:,0]=x, [:,1]=y)
        x0 = torch.floor(xs)
        xf = (xs - x0)                                         # [B,2,L] fractional
        x0 = x0.long()
        ix0 = x0[:, 0, :].clamp_(min=0).minimum(res1)          # [B,L]
        iy0 = x0[:, 1, :].clamp_(min=0).minimum(res1)
        ix1 = (ix0 + 1).minimum(res1)
        iy1 = (iy0 + 1).minimum(res1)
        # corner index tensors [B,L,4] in order (00,10,01,11)
        ix = torch.stack([ix0, ix1, ix0, ix1], dim=2)
        iy = torch.stack([iy0, iy0, iy1, iy1], dim=2)
        N = self.res.view(1, L, 1)
        size = self.sizes.view(1, L, 1)
        idx_direct = iy * N + ix                               # valid where grid fits
        idx_hash = ((ix * _PRIMES[0]) ^ (iy * _PRIMES[1])) % size
        idx = torch.where(self.direct.view(1, L, 1), idx_direct, idx_hash)
        idx = idx + self.offsets.view(1, L, 1)
        vals = self.table[idx.reshape(-1)].view(-1, L, 4, self.n_features)
        fx = xf[:, 0, :].unsqueeze(2)                          # [B,L,1]
        fy = xf[:, 1, :].unsqueeze(2)
        wx = torch.cat([1 - fx, fx, 1 - fx, fx], dim=2)        # [B,L,4]
        wy = torch.cat([1 - fy, 1 - fy, fy, fy], dim=2)
        w = (wx * wy).unsqueeze(3)                             # [B,L,4,1]
        out = (vals * w).sum(dim=2)                            # [B,L,F]
        return out.reshape(out.shape[0], -1)


_CX, _CY = gt.XMIN, gt.YMIN
_W, _H = (gt.XMAX - gt.XMIN), (gt.YMAX - gt.YMIN)


class HashGridNet(nn.Module):
    def __init__(self, n_levels=16, n_features=2, log2_T=20,
                 n_min=16, n_max=4096, hidden=128, mlp_layers=3):
        super().__init__()
        self.enc = PackedHashGridEncoding(n_levels, n_features, log2_T, n_min, n_max)
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


class PackedSolution(TorchSolution):
    name = "hashgrid_packed"
    description = ("champion recipe with packed single-gather encoder "
                   "(one [B,L,4] gather instead of ~48 per-level lookups)")

    TBL_LR0 = 6e-1
    MLP_LR0 = 5e-3
    LR_MIN_FRAC = 0.01
    WARMUP = 0.08

    def __init__(self):
        self.model = HashGridNet(n_levels=12, n_features=2, log2_T=24,
                                 n_min=16, n_max=32768, hidden=128, mlp_layers=4).to(self.device)

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
        pool_mult = 4
        n_hard = 17 * batch // 20
        budget = ctx.time_budget_s
        step = 0
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


SOLUTION = PackedSolution()
