Don't replace this text. Below, write your current notes for ideas and research threads pursued. Try not to bloat, prefer updating existing text rather than appending.

# Research Notes

## Target characteristics
- Periodic log-distance encoding: phase = 0.05*log(dist), target = 0.5+0.5*sin(2pi*phase).
- HIGH-FREQUENCY content near the boundary, detail at every scale (band freq -> inf as dist -> 0).
- In-set points (never escape) -> exactly 1.0.
- View window [-2.65,1.15]x[-1.2,1.2], aspect ~1.59. Eval grid 3840x2414 (~9.3M pts).
- Hardware: RTX 3090 Ti, 16 CPU. 5-min train budget, 10-min hard kill.

## Leaderboard (MSE, lower better)
- baseline_mlp (GELU 256x6, uniform):        0.00413279  (psnr 23.84)
- fourier_mlp (RFF 256 sigma=8, 512x6):       0.00246713  (psnr 26.08)
- hashgrid    (16 lvl, T=2^20, Nmax=4096):    0.00067412  (psnr 31.71)
- hashgrid_v2 (24 lvl, T=2^21, Nmax=8192):    0.00054886  (psnr 32.61)
- hashgrid_adaptive (v2 + err-weighted samp): 0.00051100  (psnr 32.92)  ** best
- hashgrid_bank (v2 + 32M fixed bank):        0.00130063  (psnr 28.86)  OVERFIT
- hashgrid_F4 (v2 + F=4 features):            0.00057509  (psnr 32.40)  worse (slower)
- hashgrid_bigT (v2 + T=2^23):                0.00553924  (psnr 22.57)  BAD (too sparse)
- hashgrid_best (v2 + adaptive + time-LR):    0.00048605  (psnr 33.13)  ** best
- hashgrid_replay (8M churn bank + bank-mine): 0.00065260  (psnr 31.85)  worse (staleness)

## CHAMPION: hashgrid_l12 = 0.00045894 (psnr 33.38). ~9x better than baseline (0.00413).
(hashgrid_l16 = 0.00045908 essentially tied.) Deterministic, reproducible harness.

## Champion config detail
12-16 levels (equiv), F=2, T=2^23, Nmin=16 Nmax=8192, MLP 256x4 GELU, bf16 autocast,
adaptive mining (3x uniform pool, 75% hard + 25% uniform), split LRs (table 5e-2 / MLP 5e-3)
cosine-decayed over budget. ~1400 steps.

## Level / feature sweep
- n_levels: 24->0.000462, 16->0.000459, 12->0.000459. Fewer levels = more steps AND
  slightly better (coarse levels were redundant). 12-16 optimal, GT-bound past that.
- F=4 at 16 levels: 0.000468 WORSE (slower). F=2 definitively optimal.
- smoothstep C1 interp: 0.000463, marginally worse (boundary isn't smooth, bilinear fine).

## (earlier best config note) hashgrid_bigT2 = 0.00046660, psnr 33.31
24 levels, F=2, T=2^23, Nmin=16 Nmax=8192, MLP 256x4 GELU, bf16 autocast,
adaptive mining (3x pool, 75% hard + 25% uniform), time-based cosine LR 1e-2->1e-4.

## Tuning results around best (all ~0.00047, deep diminishing returns)
- T: 2^21=0.000478, 2^22=0.000470, 2^23=0.000467 (mining trains the fine entries that
  plain uniform left cold -> bigger T finally helps WITH mining, not without).
- progressive coarse-to-fine unlocking: 0.000574 WORSE (only ~1200 steps; curriculum
  starves the fine levels of training time).
- EMA: WORSE (weight-avg blurs high-freq grid).

## More tuning (all within noise of 0.000462, hash grid is squeezed)
- hashgrid_fast (sync-free encoder): 0.000465. Step count unchanged (~1200) -> gather is
  memory-bandwidth-capped, not sync/launch-bound. ~1200 steps seems hard for this grid.
- hashgrid_lr (split table LR 5e-2 / MLP LR 5e-3): 0.00046177 ** best. Marginal win.
- hashgrid_dec (decoder 512x3): 0.000465. Decoder not the bottleneck; 256x4 fine.

- hashgrid_siren (sine decoder omega0=30): 0.084 FAILED. Grid features ~1e-4 -> sin(30*tiny)
  ~0, dead gradients. SIREN+grid needs careful scaling; not worth chasing.
- Eval grid is 3840 wide; Nmax=8192 already gives sub-eval-pixel cells -> raising Nmax is
  pointless (detail finer than eval is invisible). The limit is table-entry training
  coverage (mining) + finite steps + irreducible boundary aliasing. Floor ~0.00046.

- hashgrid_ens (K=2 averaged, half budget each): 0.000494 WORSE. Each member undertrained
  (~600 steps); bias dominates, averaging two weak models < one strong. No ensembling.

- hashgrid_compile (torch.compile): 0.000465, ~1200 steps unchanged. Gather randomly
  accesses 128MB tables >> 6MB L2 -> every lookup a DRAM read. Memory-bandwidth wall,
  compile can't fuse it. ~1200 steps is a HARD ceiling for this architecture.

## ERROR MAP (champion) — decisive
All error is on the THIN BOUNDARY CURVE; interior+exterior ~perfect. Boundary is
measure-zero, so uniform sampling lands too few points on it. -> densify sampling near
the boundary by jittering "hard anchors" (model-error-driven, general). Cheaper per step
AND far denser boundary coverage. This is the key lever, attacking the actual error.

- hashgrid_densify (jitter hard anchors, 1800 steps): 0.000505 WORSE. KEY LESSON: eval is
  UNIFORM MSE. Oversampling the measure-zero boundary creates train/eval mismatch — fits
  boundary but easy-area weight dominates uniform MSE and punishes any drift. Don't
  over-focus; mining from a UNIFORM pool (lr) is near the optimal allocation.
- Practical floor ~0.00046 confirmed: residual = boundary aliasing + finite capacity, and
  uniform metric forbids over-focusing on the boundary.

## Throughput diagnosis
- replay cut per-step GT from 786k->131k but steps only went 1200->1500. So GT is NOT
  the bottleneck — MODEL COMPUTE is (big mining-pool forward + 256x4 MLP fwd/bwd).
  -> to get more steps: mixed precision (AMP/fp16 tensor cores), smaller pool, or torch.compile.
- mining (best) beats no-mining (v2) even at fewer steps: 0.00049 vs 0.00055.
- hashgrid_amp (bf16 autocast): 0.00047836 psnr33.20. Marginal — grid GATHER is memory-
  bandwidth-bound, bf16 only speeds matmuls. Throughput ~unchanged (~1200 steps).
- hashgrid_ema (EMA decay 0.997): 0.00089834 WORSE. Weight-averaging is DESTRUCTIVE for a
  high-freq grid fit — averaging table entries blurs fine oscillations. Don't use EMA here.
- Current best 0.00047836 (hashgrid_amp). Model still improving at time limit -> MORE STEPS
  is the clearest path. Next: cheaper mining (2x pool), bigger capacity, or MFN/Gabor MLP
  (compute-bound -> AMP would actually help + more steps).

## Findings
- Hash grid >> Fourier >> MLP. Capacity helps (v1->v2). Adaptive sampling: small win.
- FIXED data bank OVERFITS badly (train 2e-5, eval 1.3e-3) despite 2.5x more steps.
  The huge-capacity grid memorizes points; fresh infinite sampling is essential for this
  high-freq target. KEEP FRESH SAMPLING. The per-step GT cost is worth it.
- n_max=8192 already gives sub-eval-pixel grid cells; the limit is now table COLLISIONS
  at fine levels (level 8192 has 67M cells vs T=2M) + MLP capacity + step count.
  -> next: raise log2_T (fewer collisions) and/or F features.

## Key insight
Spectral bias kills plain MLPs on this target. Fourier features help. The signal is a
dense spatial field with detail at all scales -> multiresolution hash-grid encoding
(Instant-NGP) is the natural SOTA approach and should win big. It's a universal function
approximator (interp feature grid + small MLP), so it's in-scope.

## Threads to pursue
- [next] Instant-NGP multiresolution hash grid encoding.
- Tune Fourier sigma (most important RFF knob).
- Boundary-oversampled / adaptive sampling (all the detail is near the boundary).
- SIREN (sine activations).
- Ensembling / larger nets (compute is free within budget).
