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
