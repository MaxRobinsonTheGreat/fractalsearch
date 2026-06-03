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
