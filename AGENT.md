# fractalsearch — Autonomous AI Research

This is an experiment in having an LLM agent do its own open-ended research: **find the
best algorithm to fit the Mandelbrot set**, i.e. learn the map `(real, imag) -> target`
defined in `harness/groundtruth.py`, while still being a universal function approximator.
You are that agent.

## The Goal

- The target function lives in `harness/groundtruth.py` (smooth normalized escape time,
  values in `[0, 1]`). It is **immutable** — read it, never change it.
- The metric is **MSE over a fixed dense grid** vs. the ground truth (`harness/evaluate.py`).
  **Lower is better. This is the only thing you are optimizing.**
- Each run trains for a **fixed 5-minute budget** (a run is force-killed at 10 minutes).
  Because the budget is fixed, you don't trade off compute — use it all. A massive
  network that scores better *is* better. Param count and speed do not matter, except in that they
  might limit run-time.
- Discover novel architectures, training methods, and algorithms to achieve this goal. 

## Setup (first time on a fresh run)

1. Agree a run tag with the human and create a branch
   `fractalsearch/<tag>` from master, or pickup from an existing branch.
2. Read the in-scope files for full context:
   - `harness/groundtruth.py` — the target (read-only).
   - `harness/interface.py` — the `Solution` contract (read-only).
   - `harness/evaluate.py` — the runner + metric (read-only).
   - `solutions/baseline_mlp.py` — the template to copy and improve upon.
3. Establish the baseline first: `uv run python -m harness.evaluate solutions/baseline_mlp.py`.
4. With human approval, start the dashboard found in `dashboard` so progress can be monitored. `uv run uvicorn dashboard.app:app --port 8000` Do not shut it down unless asked.
5. Await human approval before starting a research loop.

## What you CAN do

- **Create and edit files in `solutions/`.** Each file is one approach and must expose a
  module-level `SOLUTION` instance (or a `build()` factory) implementing `Solution`.
  Everything inside a solution is fair game: architecture, additional features,
  activations, optimizer, LR schedule, sampling strategy (uniform vs. boundary-
  oversampled vs. adaptive vs. multi-resolution/curriculum), loss shaping, normalization,
  ensembling, closed-form/analytic components. You can use any universal ML method
  with the form `predict(coords) -> values in [0,1]`.
- Search the web for research or ideas, while being open to your own crazy ideas.
- Temporarily accept higher loss or worse performance to allow for experimentation and avoid getting stuck in local minima.
- Write notes in `solutions/notebook.md` to keep track of ideas and follow up on experiments. Use this instead of whatever built-in memory module you might have, as it can be reused by different agents in the future.

## What you CANNOT do

- Do not modify anything under `harness/` — the target, the `Solution` interface, the evaluation metric, the time budget. These are the ground truth and must not be gamed.
- Do not hard-code the logic of the mandelbrot set into the solution itself. Remember the algorithm MUST still be a universal function approximator, able to fit any dataset.
- Do not add dependencies beyond `pyproject.toml` (torch, numpy, pillow are available).

## The experiment loop — LOOP FOREVER

1. Look at the git state and the current leaderboard (`runs.jsonl`, or the dashboard).
2. Form a hypothesis. Create a new solution file (**copy** the closest prior winner, do not rewrite it in full) **or** iterate on the current best file. Keep each file a single coherent idea.
3. `git add -A && git commit -m "<idea>"`.
4. Run it: `uv run python -m harness.evaluate solutions/<file>.py > run.log 2>&1`
   (redirect everything — do NOT flood your context).
5. Read the result: `grep "^mse:\|^status:" run.log`. The evaluator already appended a
   full record to `runs.jsonl` and saved artifacts under `runs/<id>/`.
6. If `run.log` shows a crash, `tail -n 50 run.log` for the traceback. Fix obvious bugs
   (typo, shape mismatch) and re-run. If the idea is fundamentally broken, move on — the
   crash is already logged with status `crash`.
7. Repeat.

**Timeout:** a run should take ~5 minutes. If it exceeds 10 minutes the evaluator kills it
and logs status `timeout`; treat that as a discard. You MUST enforce the timeout.

**Saving Solutions:** you should minimize rewriting files from scratch. Each run will automatically save a copy of the solution file, so you can edit existing solution files rather than rewriting in full. For new and different ideas where a full rewrite is necessary, create new files. 

**NEVER STOP.** Once the loop has begun, do not pause to ask the human whether to
continue. They may be asleep and expect a stack of results when they return. If you run
out of ideas, think harder: re-read the ground truth for structure to exploit, search the internet,
combine near-misses, try more radical architectures or sampling schemes. The loop runs until the human interrupts you.
