# 🌀 fractalsearch 🌀

Can AI Agents do AI research?


This project attempts to facilitate this strange loop on a toy ML problem:
How well can a function approximator fit the mandelbrot set? 
It is a low-dimensional curve-fitting problem, like fitting an image, but this image has infinite detail and complexity at every scale. You cannot 'overfit' on the mandelbrot dataset. 
This has been a [pet project of mine](https://github.com/MaxRobinsonTheGreat/mandelbrotnn) for many years, and I've run many of my own experiments on it. Because it is so simple, easy to run, and not resource-intensive, it is perfect for this kind of autonomous AI research loop. 


As the human overseer, you can edit the prompt file AGENT.MD to guide bot behavior, rather than writing any code directly. Spin up any AI agent, point it at AGENT.MD, talk with it for a bit, and let 'er rip. You can monitor performance through your webbrowser at `localhost:8000`.


This project is directly adapted form Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). 
All code was AI generated with claude (this is human written btw). 


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

- **Metric:** mean-squared error against the ground truth over a fixed grid.
  Lower is better — the single optimization target. (MAE, PSNR, and a boundary-weighted
  error are logged too, for insight.)
- **Budget:** every run trains for a fixed amount of time (defined in AGENT.md). Compute is
  free to use; a bigger model that scores better simply wins.
- **Contract:** a solution implements `fit(ctx)` (train, using `ctx.sample()` for data and
  `ctx.expired()` to respect the budget) and `predict(coords) -> values`. Anything goes
  inside, whatever works.

## Running

Canonical runtime is `uv` (Python 3.10, PyTorch CUDA), mirroring `autoresearch`.

```bash
uv sync                                                   # set up the environment

# evaluate a solution (trains 5 min, scores, logs to runs.jsonl)
uv run python -m harness.evaluate solutions/test.py

# launch the control panel, then open http://localhost:8000
uv run uvicorn dashboard.app:app --port 8000
```

For a quick experiment loop, see `AGENT.md` — point an agent at it and it will iterate
autonomously.
