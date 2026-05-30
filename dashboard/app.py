"""fractalsearch control panel — a FastAPI live dashboard.

Run it (from the project root):
    uv run uvicorn dashboard.app:app --reload --port 8000
then open http://localhost:8000

Reads runs.jsonl + runs/<id>/ artifacts. Refresh-while-the-agent-works friendly:
nothing here writes to the research log, so it's safe to keep open during a run.
"""

from __future__ import annotations

import asyncio
import json
import math
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_LOG = os.path.join(ROOT, "runs.jsonl")
RUNS_DIR = os.path.join(ROOT, "runs")
CACHE_DIR = os.path.join(ROOT, ".cache")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="fractalsearch")


def _finite(rec: dict) -> dict:
    """Replace non-finite floats (inf/nan) with None so the response is valid JSON.
    Starlette serializes with allow_nan=False, so a stray inf/nan would 500 otherwise."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in rec.items()}


def read_runs():
    if not os.path.exists(RUNS_LOG):
        return []
    runs = []
    with open(RUNS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(_finite(json.loads(line)))
                except json.JSONDecodeError:
                    pass
    return runs


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC, "index.html")) as f:
        return f.read()


def runs_payload():
    runs = read_runs()
    ok = [r for r in runs if r.get("status") == "ok" and r.get("mse") is not None]
    best = min(ok, key=lambda r: r["mse"]) if ok else None
    # running best over time (in log order) for the progress chart
    frontier, cur = [], None
    for r in runs:
        if r.get("status") == "ok" and r.get("mse") is not None:
            cur = r["mse"] if cur is None else min(cur, r["mse"])
        frontier.append(cur)
    # latest experiment = last record appended to the log
    latest = runs[-1]["run_id"] if runs else None
    return {"runs": runs, "frontier": frontier,
            "best_run_id": best["run_id"] if best else None,
            "latest_run_id": latest,
            "count": len(runs), "ok_count": len(ok)}


@app.get("/api/runs")
def api_runs():
    return runs_payload()


def _log_mtime() -> float:
    try:
        return os.path.getmtime(RUNS_LOG)
    except OSError:
        return 0.0


@app.get("/api/stream")
async def api_stream():
    """Server-sent events: push the full payload whenever runs.jsonl changes.

    The evaluator appends a record per run, which bumps the file mtime; we poll
    that cheaply server-side and only serialize when something actually changed,
    so the browser updates the instant an experiment lands — no manual refresh."""
    async def gen():
        last_mtime = None
        while True:
            mtime = _log_mtime()
            if mtime != last_mtime:
                last_mtime = mtime
                payload = json.dumps(runs_payload())
                yield f"data: {payload}\n\n"
            else:
                yield ": keepalive\n\n"  # keep the connection from idling out
            await asyncio.sleep(1.0)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/preview/{run_id}")
def api_preview(run_id: str):
    """Side-by-side strip: [ground truth | prediction | error] saved at eval time."""
    path = os.path.join(RUNS_DIR, run_id, "preview.png")
    if not os.path.exists(path):
        raise HTTPException(404, "no preview for this run")
    return FileResponse(path, media_type="image/png")


@app.get("/api/groundtruth")
def api_groundtruth(height: int = 512):
    """Reference render of the target fractal, aspect-correct and cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    from harness import groundtruth as gt
    coords, W, H = gt.aspect_grid(height, device="cpu")
    cache = os.path.join(CACHE_DIR, f"groundtruth_{W}x{H}.png")
    if not os.path.exists(cache):
        from PIL import Image
        from harness import colormap as cm
        img = gt.mandelbrot(coords).reshape(H, W).numpy()
        Image.fromarray(cm.apply(img, cm.VALUE_CMAP)).save(cache)
    return FileResponse(cache, media_type="image/png")


@app.get("/health")
def health():
    return {"ok": True, "runs": len(read_runs())}
