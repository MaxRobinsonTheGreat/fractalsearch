"""fractalsearch control panel — a FastAPI live dashboard.

Run it (from the project root):
    uv run uvicorn dashboard.app:app --reload --port 8000
then open http://localhost:8000

Reads runs.jsonl + runs/<id>/ artifacts. Refresh-while-the-agent-works friendly:
nothing here writes to the research log, so it's safe to keep open during a run.
"""

from __future__ import annotations

import io
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_LOG = os.path.join(ROOT, "runs.jsonl")
RUNS_DIR = os.path.join(ROOT, "runs")
CACHE_DIR = os.path.join(ROOT, ".cache")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="fractalsearch")


def read_runs():
    if not os.path.exists(RUNS_LOG):
        return []
    runs = []
    with open(RUNS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return runs


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC, "index.html")) as f:
        return f.read()


@app.get("/api/runs")
def api_runs():
    runs = read_runs()
    ok = [r for r in runs if r.get("status") == "ok" and r.get("mse") is not None]
    best = min(ok, key=lambda r: r["mse"]) if ok else None
    # running best over time (in log order) for the progress chart
    frontier, cur = [], None
    for r in runs:
        if r.get("status") == "ok" and r.get("mse") is not None:
            cur = r["mse"] if cur is None else min(cur, r["mse"])
        frontier.append(cur)
    return {"runs": runs, "frontier": frontier,
            "best_run_id": best["run_id"] if best else None,
            "count": len(runs), "ok_count": len(ok)}


@app.get("/api/preview/{run_id}")
def api_preview(run_id: str):
    """Side-by-side strip: [ground truth | prediction | error] saved at eval time."""
    path = os.path.join(RUNS_DIR, run_id, "preview.png")
    if not os.path.exists(path):
        raise HTTPException(404, "no preview for this run")
    return FileResponse(path, media_type="image/png")


@app.get("/api/groundtruth")
def api_groundtruth(res: int = 512):
    """Reference render of the target fractal (cached)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"groundtruth_{res}.png")
    if not os.path.exists(cache):
        import numpy as np
        import torch
        from PIL import Image
        from harness import groundtruth as gt
        coords = gt.make_grid(res, res, device="cpu")
        img = gt.mandelbrot(coords).reshape(res, res).numpy()
        Image.fromarray((np.clip(img, 0, 1) * 255).astype("uint8")).save(cache)
    return FileResponse(cache, media_type="image/png")


@app.get("/health")
def health():
    return {"ok": True, "runs": len(read_runs())}
