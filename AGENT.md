# fractalsearch — autonomous research protocol

This is an experiment in having an LLM agent do its own open-ended research: **find the
best algorithm to fit the Mandelbrot set**, i.e. learn the map `(real, imag) -> target`
defined in `harness/groundtruth.py`. You are that agent.

## The shape of the problem

- The target function lives in `harness/groundtruth.py` (smooth normalized escape time,
  values in `[0, 1]`). It is **immutable** — read it, never change it.
- The metric is **MSE over a fixed dense grid** vs. the ground truth (`harness/evaluate.py`).
  **Lower is better. This is the only thing you are optimizing.**
- Each run trains for a **fixed 5-minute budget** (a run is force-killed at 10 minutes).
  Because the budget is fixed, you don't trade off compute — use it all. A massive
  network that scores better *is* better. Param count and speed do not matter.

## Setup (first time on a fresh run)

1. Agree a run tag with the human (e.g. `may30`) and create a branch
   `fractalsearch/<tag>` from main. The repo `fractalsearch/` is a git repo.
2. Read the in-scope files for full context:
   - `harness/groundtruth.py` — the target (read-only).
   - `harness/interface.py` — the `Solution` contract (read-only).
   - `harness/evaluate.py` — the runner + metric (read-only).
   - `solutions/baseline_mlp.py` — the template to copy.
3. Establish the baseline first: `uv run python -m harness.evaluate solutions/baseline_mlp.py`.

## What you CAN do

- **Create and edit files in `solutions/`.** Each file is one approach and must expose a
  module-level `SOLUTION` instance (or a `build()` factory) implementing `Solution`.
  Everything inside a solution is fair game: architecture, Fourier/positional features,
  activations, optimizer, LR schedule, sampling strategy (uniform vs. boundary-
  oversampled vs. adaptive vs. multi-resolution/curriculum), loss shaping, normalization,
  ensembling, closed-form/analytic components, KANs, splines — anything that runs in the
  time budget and produces `predict(coords) -> values in [0,1]`.

## What you CANNOT do

- Modify anything under `harness/` — the target, the `Solution` interface, the evaluation
  metric, the time budget. These are the ground truth and must not be gamed.
- Add dependencies beyond `pyproject.toml` (torch, numpy, pillow are available).

## The experiment loop — LOOP FOREVER

1. Look at the git state and the current leaderboard (`runs.jsonl`, or the dashboard).
2. Form a hypothesis. Create a new solution file (copy the closest prior winner) **or**
   iterate on the current best file. Keep each file a single coherent idea.
3. `git add -A && git commit -m "<idea>"`.
4. Run it: `uv run python -m harness.evaluate solutions/<file>.py > run.log 2>&1`
   (redirect everything — do NOT flood your context).
5. Read the result: `grep "^mse:\|^status:" run.log`. The evaluator already appended a
   full record to `runs.jsonl` and saved artifacts under `runs/<id>/`.
6. If `run.log` shows a crash, `tail -n 50 run.log` for the traceback. Fix obvious bugs
   (typo, shape mismatch) and re-run. If the idea is fundamentally broken, move on — the
   crash is already logged with status `crash`.
7. **Keep vs. discard:** if MSE improved over the best so far, keep the commit and keep
   iterating from it. If it's equal or worse, `git reset --hard` back to the prior good
   commit (the solution file goes away; the run stays in `runs.jsonl` as a record).
8. Repeat.

**Timeout:** a run should take ~5 minutes. If it exceeds 10 minutes the evaluator kills it
and logs status `timeout`; treat that as a discard.

**NEVER STOP.** Once the loop has begun, do not pause to ask the human whether to
continue. They may be asleep and expect a stack of results when they return. If you run
out of ideas, think harder: re-read the ground truth for structure to exploit, revisit
mandelbrotnn findings (Fourier features + skip connections were strongest; Taylor features
were poor), combine near-misses, try more radical architectures or sampling schemes. The
loop runs until the human interrupts you.

## Ideas to seed thinking (not a checklist)

- Higher Fourier order / 2D Fourier basis / random Fourier features (RFF).
- Boundary-aware sampling: the set boundary is fractal and carries almost all the error —
  oversample where `0 < target < 1`, or sample adaptively from current high-error regions.
- Deeper/wider skip nets; SIREN/sine activations; gaussian/wavelet activations.
- Learning-rate schedules (warmup + cosine/linear decay), AdamW, Muon.
- Multi-resolution / progressive training; hash-grid (instant-NGP style) encodings.
- Ensembles or mixture-of-experts over regions.
