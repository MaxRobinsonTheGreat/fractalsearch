"""Evaluate one candidate solution and log the result.

Usage:
    uv run python -m harness.evaluate solutions/fourier.py
    uv run python -m harness.evaluate solutions/fourier.py --note "order=64, 6 layers"

What it does (fixed, immutable protocol):
    1. Load the Solution from the given module file.
    2. Train it via Solution.fit() with a TRAIN_BUDGET_S target (default 300s = 5 min).
       A hard SIGALRM backstop kills the run at HARD_KILL_S (default 600s = 10 min).
    3. Score it: MSE against the ground truth over a FIXED dense evaluation grid.
       (Lower MSE is better. This is THE metric.)
    4. Save the trained artifact + a preview render under runs/<run_id>/.
    5. Append a structured record to runs.jsonl and print a human summary.

The metric and the eval grid live here and in groundtruth.py — they are the ground
truth. Do not modify them mid-run; doing so makes logged results incomparable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback

import torch

from harness import groundtruth as gt
from harness.interface import FitContext, Solution

# --- Fixed evaluation protocol constants --------------------------------------
TRAIN_BUDGET_S = 300        # target training time handed to fit() (5 minutes)
HARD_KILL_S = 600           # absolute backstop; run is killed past this (10 minutes)
EVAL_RESX, EVAL_RESY = 1000, 1000   # 1M-point dense evaluation grid
PREVIEW_RES = 512           # preview render resolution (square-ish)
SEED = 1234

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs")
RUNS_LOG = os.path.join(ROOT, "runs.jsonl")
CACHE_DIR = os.path.join(ROOT, ".cache")


class _HardTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _HardTimeout()


def load_solution(path: str) -> Solution:
    """Import a solutions/*.py file and return its Solution instance."""
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("candidate_solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "SOLUTION"):
        sol = module.SOLUTION
    elif hasattr(module, "build"):
        sol = module.build()
    else:
        raise AttributeError(
            f"{path} must define a module-level `SOLUTION` or a `build()` factory.")
    if not isinstance(sol, Solution):
        raise TypeError(f"{path}: SOLUTION/build() must return a harness Solution.")
    return sol


def eval_targets(device):
    """Fixed ground-truth values on the dense eval grid (cached to disk)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"eval_{EVAL_RESX}x{EVAL_RESY}_d{gt.MAX_DEPTH}.pt"
    cache = os.path.join(CACHE_DIR, key)
    coords = gt.make_grid(EVAL_RESX, EVAL_RESY, device=device)
    if os.path.exists(cache):
        targets = torch.load(cache, map_location=device)
    else:
        targets = gt.mandelbrot(coords)
        torch.save(targets.cpu(), cache)
        targets = targets.to(device)
    return coords, targets


@torch.no_grad()
def predict_batched(sol: Solution, coords: torch.Tensor, batch=200_000) -> torch.Tensor:
    outs = []
    for i in range(0, coords.shape[0], batch):
        outs.append(sol.predict(coords[i:i + batch]).reshape(-1).to(coords.device))
    return torch.cat(outs)


def score(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    err = preds - targets
    mse = torch.mean(err * err).item()
    mae = torch.mean(err.abs()).item()
    psnr = float("inf") if mse == 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()
    # boundary-weighted error: emphasize the hard, high-detail region (target near,
    # but not at, the set). Reported only; the PRIMARY metric is mse.
    w = ((targets > 0.05) & (targets < 0.999)).float()
    bmse = (torch.sum(w * err * err) / w.sum().clamp_min(1)).item()
    return {"mse": mse, "mae": mae, "psnr": psnr, "boundary_mse": bmse}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def save_preview(sol: Solution, run_dir: str, device):
    """Render predicted fractal + ground truth + error heatmap as a single PNG."""
    try:
        from PIL import Image
        import numpy as np
        res = PREVIEW_RES
        coords = gt.make_grid(res, res, device=device)
        pred = predict_batched(sol, coords).reshape(res, res).cpu().numpy()
        truth = gt.mandelbrot(coords).reshape(res, res).cpu().numpy()
        err = np.abs(pred - truth)
        err = err / max(err.max(), 1e-8)

        def gray(a):
            return (np.clip(a, 0, 1) * 255).astype("uint8")

        strip = np.concatenate([gray(truth), gray(pred), gray(err)], axis=1)
        Image.fromarray(strip).save(os.path.join(run_dir, "preview.png"))
    except Exception as e:
        print(f"(preview render skipped: {e})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution", help="path to a solutions/*.py file")
    ap.add_argument("--note", default="", help="extra description for the log")
    ap.add_argument("--budget", type=float, default=TRAIN_BUDGET_S)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    sol = load_solution(args.solution)
    record = {
        "run_id": run_id,
        "timestamp": time.time(),
        "solution": os.path.relpath(os.path.abspath(args.solution), ROOT),
        "name": getattr(sol, "name", "unnamed"),
        "description": (getattr(sol, "description", "") + (" | " + args.note if args.note else "")).strip(" |"),
        "commit": git_commit(),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "status": "running",
        "train_seconds": 0.0,
        "mse": None, "mae": None, "psnr": None, "boundary_mse": None,
    }

    ctx = FitContext(device=device, time_budget_s=args.budget, seed=SEED)
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(HARD_KILL_S))
    t0 = time.monotonic()
    try:
        sol.fit(ctx)
        record["train_seconds"] = time.monotonic() - t0
        signal.alarm(0)

        coords, targets = eval_targets(device)
        preds = predict_batched(sol, coords)
        if device.type == "cuda":
            torch.cuda.synchronize()
        record.update(score(preds, targets))
        record["status"] = "ok"

        try:
            sol.save(os.path.join(run_dir, "model.pt"))
        except NotImplementedError:
            pass
        save_preview(sol, run_dir, device)

    except _HardTimeout:
        record["status"] = "timeout"
        record["train_seconds"] = time.monotonic() - t0
    except Exception:
        record["status"] = "crash"
        record["train_seconds"] = time.monotonic() - t0
        record["error"] = traceback.format_exc()
        print(record["error"], flush=True)
    finally:
        signal.alarm(0)

    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(record, f, indent=2)
    with open(RUNS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    print("---")
    print(f"run_id:        {record['run_id']}")
    print(f"solution:      {record['solution']}")
    print(f"status:        {record['status']}")
    print(f"train_seconds: {record['train_seconds']:.1f}")
    if record["status"] == "ok":
        print(f"mse:           {record['mse']:.8f}")
        print(f"mae:           {record['mae']:.6f}")
        print(f"psnr_db:       {record['psnr']:.2f}")
        print(f"boundary_mse:  {record['boundary_mse']:.8f}")
    sys.exit(0 if record["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
