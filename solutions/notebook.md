Don't replace this text. Below, write your current notes for ideas and research threads pursued. Try not to bloat, prefer updating existing text rather than appending.

# Research Notes

## Session 2026-06-09 — attack the gather properly
Plan: (1) champion baseline rerun (running). (2) hashgrid_packed — ONE fancy-index
gather over a packed table (differs from failed embedding_bag try; verified numerically
identical to champion encoder, diff ~1e-11). Risk: materializes [B,L,4] int64 indices
(~1.2GB at pool size) — may be DRAM-traffic-bound. (3) hashgrid_triton — fused Triton
kernel (triton 3.5.1 ships with torch, in-scope): index+gather+interp in one pass, zero
intermediates, atomic-add backward. This is the tiny-cuda-nn trick the notebook said was
never actually engineered. Success criterion: steps/300s well above ~600 at batch 768k,
then MSE < 0.000335. If steps jump but MSE doesn't move, the irreducible-boundary story
is finally proven.
- champion baseline rerun: 0.00033489 (~600 steps) — reproducible, slightly better than
  logged 0.000335.
- encoder bench (batch 768k fwd+bwd / 3.1M pool fwd): champ 47.5/47.8ms,
  packed 43.4/79.2ms (index-tensor DRAM traffic kills it — skip full run),
  TRITON FUSED 22.6/24.0ms = 2x. Correct: fwd diff 3e-7, grad diff = atomic reorder noise.
- *** hashgrid_triton: 0.00032359 — NEW BEST, first real move below the 0.000335 "floor".
  ~700 steps (600@258s). Gains > step count alone (+17% steps, -3.4% MSE); fused kernel
  also runs encoder in fp32 (champion used bf16 autocast) — precision may contribute.
  The "irreducible floor" was partly an engineering artifact, as suspected. Promoted to
  champion.py.
- STEP PROFILE (post-triton, batch 768k pool 3.1M): pool GT mandelbrot 346ms (81%!),
  pool model fwd 36ms, multinomial 0.6ms, train fwd+bwd+adam 43ms. GT is now THE
  bottleneck (the old "GT is not the bottleneck" note was an artifact of the slow encoder).
- hashgrid_megabank (250M precomputed bank, GT-free pools): 0.00067270 WORSE 2x despite
  ~3400 steps. Train loss 1e-5 vs eval 6.7e-4 = pure memorization, even at 27x eval-grid
  density. FINAL WORD: no fixed bank of any size; the 33M-param grid memorizes points.
  Fresh sampling is structurally required.
- *** hashgrid_gtfree: 0.00029260 — NEW BEST (-9.6%), first sub-3e-4. Fresh coords each
  step, mining by finite-diff HF proxy |f(x+2e-4 d)-f(x)| on the model itself (no pool
  GT), GT only on the selected 768k batch. ~1170 steps (1.7x). Promoted to champion.py.
  KEY INSIGHT: the model's own local variation is a good-enough hardness signal; true
  per-point error is not required for mining. boundary_mse 0.000300 ~= mse — error is
  now spread, not boundary-concentrated like before.
- *** gtfree pool_mult=8: 0.00027747 — NEW BEST (-5.2%). ~900 steps (fewer than pool4's
  1170) but stronger mining selectivity wins. Train loss HIGHER (0.00208 vs 0.00119) while
  eval improves — mined samples are harder; train loss is no longer comparable across
  mining strengths. Promoted to champion.py.
- pool_mult bracket SETTLED: 4 -> 0.000293, 8 -> 0.000277, 12 -> 0.000274 (best,
  champion), 16 -> 0.000277 (worse, steps drop to ~610). Optimum = 12.
- Proxy floor at pool12: 1e-4 -> 0.00027444, 3e-5 -> 0.00027450 = TIE. Floor insensitive
  in this range; keeping 1e-4, no further floor runs.
- Batch bracket at pool12 SETTLED: 524k -> 0.000276, 768k -> 0.000274 (champion), 1M ->
  0.000275. Keep 768k.
- hashgrid_2stage (FD preselect 2x -> GT -> true-error select): 0.00027854 WORSE than
  pure FD-proxy (0.00027444). True-error refinement not worth its GT cost (~610 vs 700
  steps); the FD proxy alone is a sufficient hardness signal. Mining is settled:
  12x GT-free pool, FD proxy, 85% hard, batch 768k.
- hashgrid_F4 (retry under cheap encoder): 0.00030234 WORSE, steps fell to ~440 (2x
  table = more gather DRAM + 2x Adam state). F=2 confirmed optimal in the new regime too.
- NEW BOTTLENECK after gtfree: the two 9.4M-point proxy fwds (~220ms) dominate the step.
- *** hashgrid_errfield: 0.00024397 — NEW BEST (-11.1%!), ~1600 steps. Persistent
  2048x1296 EMA(0.9) field of per-cell MEAN |error| (mean, not sum, so oversampled cells
  don't self-reinforce), updated free from train residuals; hard coords =
  cell-multinomial + in-cell jitter, 15% uniform, batch 768k. Zero pool forwards.
  Promoted to champion.py. Mining is now FREE; the error signal is temporally averaged
  (less noisy than single-pool estimates) — both cheaper AND better.
- Field res bracket SETTLED: 1024 -> 0.000245, 2048 -> 0.000244 (champion), 4096 ->
  0.000260 (sparse/stale per-cell stats). Keep 2048.
- Hard fraction: 85% -> 0.000244, 90% -> 0.000243, 95% -> 0.00024134 (best, promoted),
  98% running. Unlike pool-mining (where >90% hurt), the errfield tolerates high hard
  fractions — its EMA + mean-update provide implicit coverage. Then EMA bracket, then
  16-lvl / Nmax-65536 retry under the ~1600-step regime.
  Session trajectory: 0.000335 -> 324 (triton) -> 293 (gtfree) -> 274 (pool12) -> 244
  (errfield). The "irreducible floor at 0.000336" is now beaten by 27%.

## Folder cleanup (2026-06-09)
~70 hyperparameter-sweep variants were deleted from solutions/ to save tokens — every
result is recorded below and every file is recoverable from git history (they lived at
commit a409e2b and earlier). What remains is one file per *distinct approach*:
champion.py (best, 0.000335 — the matured hashgrid lineage), baseline_mlp, fourier_mlp,
hashgrid_bag (packed gather, next-bet #1), hashgrid_siren2 (sine decoder),
hashgrid_densify (anchor-jitter sampling), hashgrid_replay2 (replay bank),
hashgrid_ens (ensembling). PREFER editing these in place (small diffs) over creating
new near-copies; only add a new file for a genuinely new mechanism.

## >>> NEXT SESSION — START HERE (the floor is NOT irreducible) <<<
Champion is ~0.000336 (12.3x) but that's a THROUGHPUT/BUDGET floor, not a representation
floor. The tell: train loss was STILL DROPPING at the 5-min buzzer -> the grid is
undertrained, it just ran out of gradient steps (~400-600). "Irreducible HF boundary" was
too convenient a story; the real bottleneck is steps, and steps are capped by a slow gather.

The gather was complained about but never actually engineered — only off-the-shelf fixes
tried (bf16, stock torch.compile, both ~no help). Root cause: the encoder is a PYTHON LOOP
over 12 levels, each doing 4 separate fancy-index lookups = ~48 tiny gather kernels per
forward, x3 forwards/step. That's launch-overhead + uncoalesced access, not a hard bandwidth
limit (if it were purely bandwidth-bound, compile would've helped). Likely 2-4x in steps here.

Ranked bets to try next:
1. PACKED/VECTORIZED GATHER (highest confidence, pure engineering): pack all levels into ONE
   contiguous table with per-level offsets; do the whole multi-level x 4-corner lookup as a
   single batched gather (few kernels, coalesced). Then test the ONE regime we never had:
   BIG batch (768k, low-variance grads) AND many steps (~2000). Data hints this is the gap:
   524k@~800steps (0.000347) lost to 768k@~600steps (0.000341) on gradient quality, but
   nobody has tried 768k@~2000steps. Faster gather unlocks exactly that.
2. GT-FREE MINING: rank the mining pool by the model's own output gradient/local variation
   (a boundary proxy needing NO mandelbrot eval); compute GT only on the selected batch ->
   frees budget for more steps. (Note: confirmed the gather, not GT, is the main cost — so
   pair this with #1; alone it won't help much.)
3. CUDA graphs to kill per-step Python overhead once batch/pool shapes are fixed.
Hypothesis to verify first: log train-loss vs eval and confirm undertraining (loss still
dropping at deadline). If a faster gather gives ~2x steps at batch 768k, expect a real drop
below 0.000336. If NOT — then the irreducible-boundary story is finally earned.

2026-06-03 Codex continuation:
- hashgrid_bag (packed single table + embedding_bag weighted 4-corner reduction) tested the
  "fewer gather kernels" idea in pure PyTorch. Result 0.00035389, step 200 at 169.6s: worse
  than champion and not faster. Lesson: PyTorch embedding_bag is not the missing fused
  tiny-cuda-nn-style kernel here; it likely adds overhead / poorer memory behavior.
- Next observation: existing "hard mining" uses torch.multinomial(perr, n_hard), i.e.
  error-proportional sampling, not the actual top hard examples. Test true top-k mining with
  otherwise champion-like settings; it may improve boundary focus, or it may overfocus and
  confirm proportional sampling's regularizing value.
- hashgrid_topk4 (pool4, 75% true top-k): 0.00034016. Worse than pool4/proportional
  0.00033660 and champion 0.00033559, despite low train loss. True hardest mining overfocuses;
  proportional stochastic mining is acting as useful regularization / coverage. Next test:
  intermediate selectivity via multinomial(error^2), not top-k.
- hashgrid_pow2 (pool4, 75% multinomial(error^2)): 0.00033908. Still worse. Mining
  selectivity bracketed: linear error-proportional sampling is the sweet spot; stronger
  hard focus improves train loss but hurts uniform eval. Move back to throughput/capacity:
  try smaller decoder width 128 with champion pool6 to see if MLP compute can be traded for
  more steps without losing hash-grid detail.
- hashgrid_h128 (champion pool6, decoder width 128): 0.00033551, tiny new best vs pool6
  0.00033559. Step 400 at 272s. Decoder 256 was not obviously needed; smaller MLP may
  improve throughput/regularization while grid carries detail. Sweep lower width 64 next.
- hashgrid_h64 (champion pool6, decoder width 64): 0.00033671, worse. Width 64 loses
  decoder capacity; width 128 looks like the useful compression point. Next combine h128
  with the previously near-best cheaper mining regime: 4x pool, 90% hard.
- hashgrid_h128_hard90 (width 128, 4x pool, 90% hard): 0.00033548, tiny new best.
  Crucial throughput signal: step 600 at 299s, vs h128 pool6 only step 400 at 272s. Width
  128 + cheaper strong mining is the current local optimum. Try hard fraction 95% next;
  if worse, bracket around 85/90.
- hashgrid_h128_hard95: 0.00033553, slightly worse than hard90. 95% hard lowers train
  loss a little but does not improve uniform eval. Hard90 remains the local best, but the
  win is tiny enough that it needs a validation rerun before promoting champion.py.
- hashgrid_h128_hard90 validation rerun: 0.00033547 (new best, confirmed). Promoted to
  champion.py. Current recipe: 12-level F=2 T=2^24 Nmax32768 hash grid, decoder width 128,
  batch 786432, 4x pool, 90% error-proportional hard mining, table LR 6e-1 / MLP LR 5e-3,
  8% warmup + cosine, bf16. Improvement is small but reproducible; keep exploring nearby
  mining fractions and width/LR interactions.
- hashgrid_h128_hard85: 0.00033536, new best. Same width-128 / 4x pool setup, but 85%
  hard samples. Promoted to champion.py at user stop request. Next if resumed: validate
  champion.py once, then bracket hard fraction at 80/87.5 or tune table LR around 6e-1
  under the h128/4x-pool/85%-hard regime.

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

## *** CHAMPION: hashgrid_pool6 = 0.00033559 (psnr 34.74). ~12.3x better than baseline. ***
Mining strength re-tuned at the bigger batch (768k) — the pool/batch interaction reopened it:
- pool_mult 3x->4x->6x at batch 768k: 0.000341->0.000337->0.000336. Bigger ABSOLUTE pool
  finds harder boundary points (stronger mining). Saturates ~4-6x (pool GT cost caps steps).
- n_hard 75%->90% at 4x pool: 0.000337->0.000336 (ties pool6, cheaper). Both ~0.000336.
Robust pick: hashgrid_hard90 (4x pool, 90% hard) ~ pool6. Note: ~400-560 steps (pool-GT bound).
- combo (batch 1M + 4x pool + 90% hard): 0.000336 — tied. Everything clusters at ~0.000336.

## More confirmations (all ~floor or worse)
- sigmoid output head: 0.000337 (tied; output activation irrelevant).
- 2phase (cheap uniform 65% then mining 35%): 0.000366 worse — continuous mining beats
  deferred mining; the easy phase doesn't build the boundary that late mining must refine.

## VALIDATED: champion.py re-run = 0.00033565 (matches pool6 0.00033559 within CUDA noise).
Result is reproducible & stable. Canonical file: solutions/champion.py.
Why mining strength > steps (the convergence proof): wloss=1200 cheap steps -> 0.000373;
pool6=~400 steps strong mining -> 0.000336. More steps don't help; representation/boundary
is the limit. Hence the floor is real, not a throughput artifact.

## Three independent hypothesis tests on the residual — all say "irreducible HF boundary"
- dualhash (2 hashes/level): no change -> NOT hash collisions.
- rot (axis-aligned + 45deg rotated dual grid): 0.000343 worse -> NOT directional aliasing
  (consistent with the smooth, non-staircase error filament).
- fp32: worse -> NOT interpolation precision.
=> residual is genuine unresolved high-frequency content at the boundary + representation
   limit, within a throughput-bounded budget. Floor ~0.000336 is real.

## ============ FINAL: CONVERGED at MSE ~0.000336 (PSNR 34.74), 12.3x vs baseline ============
54 experiments. Winning recipe = Instant-NGP hash grid (12 lvl, F=2, T=2^24, Nmin16/Nmax32768)
+ small GELU MLP 256x4, fresh-data adaptive hard-mining (4-6x uniform pool, 75-90% hard),
big batch (768k-1M), high table LR (4-6e-1) + low MLP LR (5e-3) with 8% warmup + cosine, bf16.
THE LEVERS THAT MATTERED: (1) hash grid arch, (2) big-batch+high-LR+warmup, (3) Nmax FAR past
eval res (point-sampled targets), (4) strong mining scaled with batch. Residual = irreducible
HF boundary filament + throughput cap (~400-600 steps, pool-GT-bound). Floor reached.

## (prior) hashgrid_n32b1m = 0.00034029 (psnr 34.68)
After Nmax=32768, the finer grid wants BIGGER batches (more fine-cell coverage/step):
batch 524k->0.000347, 768k->0.000341, 1M->0.000340 (marginal, ~560 steps). LR 8e-1 tied 6e-1.
(n32b768 = 0.000341 is the robust pick; 1M barely better with far fewer steps.)

## Throughput-vs-quality, final word
- wloss (error-weighted loss on uniform batch, no pool, 1200 steps): 0.000373 worse. The
  3x mining POOL's selectivity (hardest of 3x) beats weighting + 2x steps. Mining earns its GT.
- fp32 (no autocast, 400 steps): 0.000357 worse. bf16 throughput > fp32 precision; the
  interpolation precision is NOT the limiter. Keep bf16.

## SIREN decoder, properly scaled (final architecture test)
- siren2 (LayerNorm grid features + sine decoder omega0=10): 0.000449 worse. Trains fine now
  (LayerNorm fixed the dead-gradient scaling) but sine decode loses to GELU; grid features
  already encode local structure, LayerNorm discards useful magnitude. GELU decoder optimal.

## Pattern across ~48 experiments
WINS were compute-EFFICIENCY (big batch, bf16, mining-from-pool, high LR+warmup) and
RESOLUTION (Nmax=32768). LOSSES were every "more compute/step" idea (fp32, 2x→3x is fine but
bigger pools, Sobolev-style, replay) — we are THROUGHPUT-BOUND (~600 steps), steps are
precious. And every over-capacity (F4, T25, 256x6, 16 lvl) or over-focus (densify) idea.

## Collision hypothesis DISPROVEN (dual hash)
- dualhash (2 independent hash tables/fine level, averaged): 0.00034127 = IDENTICAL to
  champion. Decorrelating collisions gives nothing -> T=2^24 collisions are already well
  handled (even beneficially regularizing). Residual is irreducible boundary HF content,
  not collision noise. Confirms we are at the true floor.

## Resolution ceiling + final state
- Nmax=65536 + 14 levels: 0.00034077 — ties Nmax=32768 champion. Resolution ceiling
  ~32768-65536; finer gives nothing more (collisions/throughput).
- Error map at 0.00034: still the thin boundary filament but MUCH fainter than at 0.00046.
  Residual = near-irreducible high-freq boundary + throughput-bound step count (~600).
- CONVERGED at ~0.000340 (psnr 34.68), 12.1x vs baseline. Canonical champion:
  hashgrid_n32b1m (0.00034029) or robust hashgrid_n32b768 (0.00034129) / n64l14 (0.00034077).

## Re-tuning around Nmax=32768 champion (all confirm robustness)
- F=4: 0.000363 worse. MLP 256x6: 0.000347 worse. Nmin=64: 0.000343 worse. 2x pool: 0.000351
  worse (3x mining still best). 16 levels: flat. -> 12 lvl/F2/256x4/Nmin16/3xpool all optimal.

## (intermediate) hashgrid_nmax32k = 0.00034667 (psnr 34.60), batch 524k
BREAKTHROUGH: raising Nmax (finest grid resolution) FAR past eval resolution helps a lot.
EARLIER CLAIM "Nmax>eval is useless" WAS WRONG — the target is POINT-sampled, so finer
cells give the grid more DOF to fit each near-boundary eval point (adjacent eval pixels sit
on different fine bands).
- Nmax: 8192->0.000432, 16384->0.000363, 32768->0.000347 BEST, 65536->0.000347 (flat).
- at Nmax=32768: 16 levels->flat, T=2^25->0.000352 (worse, 2^24 collision-reg is best).
Champion config: 12 lvl, F=2, T=2^24, Nmin=16 Nmax=32768, MLP 256x4, batch 524k,
table LR 6e-1 + MLP 5e-3, 8% warmup+cosine, bf16, 3x mining pool 75% hard. ~800 steps.

## (superseded) hashgrid_warmup6 = 0.00043186 (psnr 33.65)
Config: 12 lvl, F=2, T=2^24, Nmax=8192, MLP 256x4, batch 524k, table LR 6e-1 + MLP LR
5e-3, 8% linear warmup then cosine, bf16, 3x mining pool 75% hard. ~800 steps.
- T=2^24: 0.000437 (slightly > 2^23 0.000438, high LR trains the bigger table).
- 8% LR warmup: 0.000433 (helps stabilize the aggressive table LR early).
- peak table LR 6e-1 (with warmup): 0.000432, marginal over 4e-1. LR near ceiling.
- Remaining diffs are ~1e-6 (noise). Effective optimum for this architecture/budget.

## (intermediate) hashgrid_T24 = 0.00043693 (psnr 33.60)
- replay2 (32M churn bank, champion regime): 0.000492 worse — mining-pool forward still
  dominates (cheaper GT didn't add steps) + bank staleness. Fresh sampling wins, final word.

## prior champion: hashgrid_bb_lr4 = 0.00043821 (psnr 33.58)
BIG-BATCH + HIGH-LR was the breakthrough past the "converged" 0.000459 plateau:
- batch 262k->524k: 0.000459->0.000453 (better gather utilization + lower-variance grads).
  768k/1M tied-or-worse (too few steps). 524k optimal.
- table LR (with batch 524k): 5e-2->0.000453, 1e-1->0.000448, 2e-1->0.000443,
  4e-1->0.000438 BEST, 8e-1->0.000440. Big batch supports much higher LR. Peak ~4e-1.
- MLP LR: 5e-3 best (1.5e-2 worse). Keep 5e-3.
Champion config: 12 lvl, F=2, T=2^23, Nmax=8192, MLP 256x4, batch 524k, table LR 4e-1
(cosine), MLP LR 5e-3, bf16, 3x mining pool 75% hard. ~1000 steps. ~9.4x vs baseline.

## (superseded) SEARCH CONVERGED (~0.000459, psnr 33.4) — hashgrid_l12
~9x better than baseline. Step cap (~1400) is GT-bound; fresh data is required (banks
overfit) so GT can't be cheapened -> cap is fundamental. Untried low-prob ideas if
resuming: LayerNorm-then-1-sine-layer decode; batch 524k; tuned MFN/Gabor net.
- hashgrid_l12p2 (2x pool): 0.000473 worse — 3x mining pool optimal even when GT-bound.
- hashgrid_xy (concat raw coord): 0.000459 neutral.
- hashgrid_l16F4 (F=4): 0.000468 worse.

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
