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
    """Legacy side-by-side strip: [ground truth | prediction | error]."""
    path = os.path.join(RUNS_DIR, run_id, "preview.png")
    if not os.path.exists(path):
        raise HTTPException(404, "no preview for this run")
    return FileResponse(path, media_type="image/png")


def _layer_response(run_id: str, layer: str):
    """Serve a single preview layer ('prediction' or 'error').

    New runs save these as standalone PNGs. For older runs that only have the
    combined preview.png strip [gt | pred | err], slice the requested third out
    of it on the fly so the dashboard works for the whole history."""
    direct = os.path.join(RUNS_DIR, run_id, f"{layer}.png")
    if os.path.exists(direct):
        return FileResponse(direct, media_type="image/png")
    strip = os.path.join(RUNS_DIR, run_id, "preview.png")
    if os.path.exists(strip):
        from io import BytesIO
        from PIL import Image
        img = Image.open(strip)
        third = img.width // 3
        idx = {"prediction": 1, "error": 2}[layer]  # 0=gt, 1=pred, 2=err
        crop = img.crop((idx * third, 0, (idx + 1) * third, img.height))
        buf = BytesIO()
        crop.save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")
    raise HTTPException(404, f"no {layer} for this run")


@app.get("/api/prediction/{run_id}")
def api_prediction(run_id: str):
    """The run's predicted fractal (inferno-colored), pixel-aligned to ground truth."""
    return _layer_response(run_id, "prediction")


@app.get("/api/error/{run_id}")
def api_error(run_id: str):
    """The run's |prediction - truth| heatmap (inferno), pixel-aligned to ground truth."""
    return _layer_response(run_id, "error")


@app.get("/api/code/{run_id}")
def api_code(run_id: str):
    """Serve the solution source snapshot saved alongside the run (runs/<id>/<name>.py).
    The evaluator copies the solution file into the run dir, so this is exactly the code
    that produced the run — even if the original solutions/ file later changed."""
    import glob
    pys = sorted(glob.glob(os.path.join(RUNS_DIR, run_id, "*.py")))
    if not pys:
        raise HTTPException(404, "no solution snapshot for this run")
    return FileResponse(pys[0], media_type="text/plain")


@app.get("/api/groundtruth")
def api_groundtruth():
    """Serve the committed 4K ground-truth reference (dashboard/static/groundtruth.png),
    rendered on the eval grid so it is pixel-aligned with the prediction/error layers.
    Regenerate after a window/target change: `python -m dashboard.make_groundtruth`."""
    path = os.path.join(STATIC, "groundtruth.png")
    if not os.path.exists(path):
        raise HTTPException(404, "groundtruth.png missing — run "
                                 "`python -m dashboard.make_groundtruth`")
    return FileResponse(path, media_type="image/png")


@app.get("/health")
def health():
    return {"ok": True, "runs": len(read_runs())}
