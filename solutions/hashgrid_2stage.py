"""Champion recipe + FUSED Triton hash-grid encoder (tiny-cuda-nn style, in-scope:
triton ships with torch).

Why: the champion encoder is ~48 tiny gather kernels per forward; the packed pure-torch
variant is 1 gather but must materialize a [B,L,4] int64 index tensor (~1.2 GB at the
3.1M-point mining pool) plus weight tensors — huge extra DRAM traffic. This kernel fuses
index computation + 4-corner gather + bilinear interp into ONE pass with zero
intermediates, and the backward recomputes indices and atomic-adds straight into the
table gradient. If the gather was the throughput cap, steps should jump (~600 -> 1500+)
and the same recipe should beat 0.000335.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from harness import groundtruth as gt
from harness.interface import FitContext, TorchSolution

P1 = tl.constexpr(2654435761)


@triton.jit
def _enc_fwd(x_ptr, tbl_ptr, out_ptr, meta_ptr, B,
             L: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    # meta layout per level: [N, size, offset, direct]  (int64 x 4)
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < B
    x = tl.load(x_ptr + o * 2, mask=m, other=0.0).to(tl.float32)
    y = tl.load(x_ptr + o * 2 + 1, mask=m, other=0.0).to(tl.float32)
    for l in tl.static_range(L):
        N = tl.load(meta_ptr + l * 4 + 0)
        size = tl.load(meta_ptr + l * 4 + 1)
        off = tl.load(meta_ptr + l * 4 + 2)
        direct = tl.load(meta_ptr + l * 4 + 3)
        xs = x * (N - 1).to(tl.float32)
        ys = y * (N - 1).to(tl.float32)
        ix0 = xs.to(tl.int64)            # x,y in [0,1] -> floor == trunc
        iy0 = ys.to(tl.int64)
        fx = xs - ix0.to(tl.float32)
        fy = ys - iy0.to(tl.float32)
        ix1 = tl.minimum(ix0 + 1, N - 1)
        iy1 = tl.minimum(iy0 + 1, N - 1)
        for f in tl.static_range(F):
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            # corners: (00),(10),(01),(11)
            for c in tl.static_range(4):
                ix = ix0 if (c % 2 == 0) else ix1
                iy = iy0 if (c // 2 == 0) else iy1
                wx = (1.0 - fx) if (c % 2 == 0) else fx
                wy = (1.0 - fy) if (c // 2 == 0) else fy
                idx_d = iy * N + ix
                idx_h = ((ix * 1) ^ (iy * P1)) % size
                idx = tl.where(direct != 0, idx_d, idx_h) + off
                v = tl.load(tbl_ptr + idx * F + f, mask=m, other=0.0)
                acc += v * wx * wy
            tl.store(out_ptr + o * (L * F) + l * F + f, acc, mask=m)


@triton.jit
def _enc_bwd(x_ptr, go_ptr, gtbl_ptr, meta_ptr, B,
             L: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < B
    x = tl.load(x_ptr + o * 2, mask=m, other=0.0).to(tl.float32)
    y = tl.load(x_ptr + o * 2 + 1, mask=m, other=0.0).to(tl.float32)
    for l in tl.static_range(L):
        N = tl.load(meta_ptr + l * 4 + 0)
        size = tl.load(meta_ptr + l * 4 + 1)
        off = tl.load(meta_ptr + l * 4 + 2)
        direct = tl.load(meta_ptr + l * 4 + 3)
        xs = x * (N - 1).to(tl.float32)
        ys = y * (N - 1).to(tl.float32)
        ix0 = xs.to(tl.int64)
        iy0 = ys.to(tl.int64)
        fx = xs - ix0.to(tl.float32)
        fy = ys - iy0.to(tl.float32)
        ix1 = tl.minimum(ix0 + 1, N - 1)
        iy1 = tl.minimum(iy0 + 1, N - 1)
        for f in tl.static_range(F):
            g = tl.load(go_ptr + o * (L * F) + l * F + f, mask=m, other=0.0)
            for c in tl.static_range(4):
                ix = ix0 if (c % 2 == 0) else ix1
                iy = iy0 if (c // 2 == 0) else iy1
                wx = (1.0 - fx) if (c % 2 == 0) else fx
                wy = (1.0 - fy) if (c // 2 == 0) else fy
                idx_d = iy * N + ix
                idx_h = ((ix * 1) ^ (iy * P1)) % size
                idx = tl.where(direct != 0, idx_d, idx_h) + off
                tl.atomic_add(gtbl_ptr + idx * F + f, g * wx * wy, mask=m)


class _EncFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, table, meta, L, F):
        x = x.contiguous().float()
        B = x.shape[0]
        out = torch.empty(B, L * F, device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(B, 256),)
        _enc_fwd[grid](x, table, out, meta, B, L=L, F=F, BLOCK=256)
        ctx.save_for_backward(x, meta)
        ctx.shape = (table.shape, L, F)
        return out

    @staticmethod
    def backward(ctx, go):
        x, meta = ctx.saved_tensors
        tshape, L, F = ctx.shape
        B = x.shape[0]
        gtbl = torch.zeros(tshape, device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(B, 256),)
        _enc_bwd[grid](x, go.contiguous().float(), gtbl, meta, B, L=L, F=F, BLOCK=256)
        return None, gtbl, None, None, None


class TritonHashGridEncoding(nn.Module):
    def __init__(self, n_levels=16, n_features=2, log2_T=20, n_min=16, n_max=4096):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        T = 1 << log2_T
        b = (n_max / n_min) ** (1.0 / (n_levels - 1))
        res = [int(round(n_min * b ** l)) for l in range(n_levels)]
        sizes = [min(N * N, T) for N in res]
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)
        meta = []
        for l in range(n_levels):
            meta += [res[l], sizes[l], offsets[l], 1 if res[l] * res[l] <= sizes[l] else 0]
        self.register_buffer("meta", torch.tensor(meta, dtype=torch.long))
        self.table = nn.Parameter(
            torch.empty(offsets[-1], n_features).uniform_(-1e-4, 1e-4))
        self.out_dim = n_levels * n_features

    def forward(self, x):
        return _EncFn.apply(x, self.table, self.meta, self.n_levels, self.n_features)


_CX, _CY = gt.XMIN, gt.YMIN
_W, _H = (gt.XMAX - gt.XMIN), (gt.YMAX - gt.YMIN)


class HashGridNet(nn.Module):
    def __init__(self, n_levels=16, n_features=2, log2_T=20,
                 n_min=16, n_max=4096, hidden=128, mlp_layers=3):
        super().__init__()
        self.enc = TritonHashGridEncoding(n_levels, n_features, log2_T, n_min, n_max)
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


class TwoStageSolution(TorchSolution):
    name = "hashgrid_2stage"
    description = ("two-stage mining: FD-proxy preselect 2x batch from a 12x GT-free "
                   "pool, compute GT on those, then TRUE-error-proportional select the "
                   "final batch. True-error selectivity at ~2x (not 12x) GT cost.")

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
        pool_mult = 12
        presel_mult = 2          # stage-1 keeps presel_mult * batch candidates
        n_hard = 17 * batch // 20
        budget = ctx.time_budget_s
        step = 0
        ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        # finite-difference probe: well below one eval-grid pixel (3.5/3840 ~ 9e-4)
        delta = 2e-4
        while not ctx.expired():
            self._set_lr(opt, min(1.0, ctx.elapsed() / budget))
            with torch.no_grad(), ac:
                pcoords = gt.sample_uniform(batch * pool_mult,
                                            generator=ctx.generator, device=ctx.device)
                ppred = self.model(pcoords).reshape(-1)
                off = torch.randn_like(pcoords)
                off = off * (delta / off.norm(dim=1, keepdim=True).clamp_min(1e-12))
                ppred2 = self.model(pcoords + off).reshape(-1)
                # HF proxy: local variation of the model's own prediction. Where the
                # model has fit HF structure this is large; where it is flat (well-fit
                # smooth regions AND unlearned-but-smooth regions) it is small, so keep
                # a uniform floor for coverage.
                score = (ppred.float() - ppred2.float()).abs() + 1e-4
            # stage 1: FD-proxy preselect (GT-free), keep a 15% uniform slice for coverage
            n_pre = presel_mult * batch
            n_pre_hard = 17 * n_pre // 20
            pre_hard = torch.multinomial(score, n_pre_hard, replacement=False)
            pre_unif = torch.randint(0, pcoords.shape[0], (n_pre - n_pre_hard,),
                                     device=pcoords.device)
            pre_idx = torch.cat([pre_hard, pre_unif])
            cand_c = pcoords[pre_idx]
            cand_t = ctx.ground_truth(cand_c)
            # stage 2: TRUE-error-proportional selection among GT'd candidates
            cand_err = (ppred[pre_idx].float() - cand_t).abs() + 1e-6
            sel_hard = torch.multinomial(cand_err, n_hard, replacement=False)
            sel_unif = torch.randint(0, cand_c.shape[0], (batch - n_hard,),
                                     device=pcoords.device)
            sel = torch.cat([sel_hard, sel_unif])
            coords, targets = cand_c[sel], cand_t[sel]

            with ac:
                pred = self.model(coords).reshape(-1)
                loss = torch.mean((pred.float() - targets) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                ctx.log(f"step {step} loss {loss.item():.5f} (left {ctx.time_left():.0f}s)")


SOLUTION = TwoStageSolution()
