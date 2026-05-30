# fractalsearch

Autonomous research into **the best algorithm to fit the Mandelbrot set** — learning the
map `(real, imag) -> measure of escape iterations`. A loose framework where an agent
proposes approaches, the harness scores them on a fixed metric under a fixed time budget,
and a live control panel lets a human watch progress and visualize the best solutions.

A sibling of [`autoresearch`](../autoresearch) — same paradigm (immutable harness +
mutable candidate + fixed budget + keep/discard loop), applied to fractal fitting.

## Layout

```
harness/            IMMUTABLE — the rules of the game (agent reads, never edits)
  groundtruth.py    the target: (real, imag) -> value in [0,1]   ← single source of truth
  interface.py      the Solution contract + FitContext (data/budget given to fit())
  evaluate.py       runner: trains a solution (5 min), scores MSE, logs, saves artifacts
solutions/          MUTABLE registry — one file per approach (the agent's workspace)
  baseline_mlp.py   plain MLP (template — copy this to start a new approach)
  fourier.py        Fourier features + skip connections (mandelbrotnn's strongest family)
dashboard/          FastAPI live control panel
  app.py            serves the log + renders; static/index.html is the UI
runs.jsonl          append-only performance log (gitignored)
runs/<id>/          per-run artifacts: model.pt, preview.png, result.json
AGENT.md            the autonomous research protocol the agent follows
```

## How it works

- **Metric:** mean-squared error against the ground truth over a fixed 1000×1000 grid.
  Lower is better — the single optimization target. (MAE, PSNR, and a boundary-weighted
  error are logged too, for insight.)
- **Budget:** every run trains for a fixed **5 minutes** (hard-killed at 10). Compute is
  free to use; a bigger model that scores better simply wins.
- **Contract:** a solution implements `fit(ctx)` (train, using `ctx.sample()` for data and
  `ctx.expired()` to respect the budget) and `predict(coords) -> values`. Anything goes
  inside — nets, Fourier/hash encodings, KANs, analytic forms, custom sampling.

## Running

Canonical runtime is `uv` (Python 3.10, PyTorch CUDA), mirroring `autoresearch`.

```bash
cd fractalsearch
uv sync                                                   # set up the environment

# evaluate a solution (trains 5 min, scores, logs to runs.jsonl)
uv run python -m harness.evaluate solutions/fourier.py

# launch the control panel, then open http://localhost:8000
uv run uvicorn dashboard.app:app --port 8000
```

For a quick experiment loop, see `AGENT.md` — point an agent at it and it will iterate
autonomously, committing each idea and keeping only what lowers MSE.

## Changing the target

The output definition (smooth iterations, raw count, binary membership, log-scaled, …)
lives entirely in `harness/groundtruth.py`. Change it there and start a fresh run tag —
past results become incomparable once the target changes.
