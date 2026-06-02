"""Ground truth for fractalsearch — the single source of truth for the target.

Every solution learns the mapping  (real, imag) -> target value  defined here.

IMMUTABLE: solutions may import and read this module, but must NEVER modify it,
and the evaluation metric (see harness/evaluate.py) is computed against it. Changing
any constant or the target definition invalidates comparability across all logged runs.

If you (the human) want to change what is being fit (e.g. raw iteration count vs.
smooth normalized vs. binary membership), this is the ONE place to do it. Do it
deliberately and start a fresh run tag, because past results become incomparable.

Target definition (current):
    - escape-time iteration with a continuous (smooth) iteration count
    - mapped through `smooth()` to (0, 1]; points in the set -> 1.0
    - this matches the mandelbrotnn `smoothMandelbrot` framing, but uses the
      *continuous* escape count so the target is smooth enough to fit well.
"""

import math
import torch

# --- View window --------------------------------------------------------------
# Zoomed out a hair from the tighter [-2.5,1.0]x[-1.1,1.1] framing: the set's upper
# and lower tips reach ~±1.12 and were being clipped at the top/bottom edges. This is
# a centered ~9% zoom-out about the original center (-0.75, 0), so the whole set now
# sits inside the frame with a small margin. (Changing the window changes what is being
# fit -> past results are not comparable; this is a deliberate, fresh-slate change.)
XMIN, XMAX = -2.65, 1.15
YMIN, YMAX = -1.2, 1.2

# --- Target parameters --------------------------------------------------------
MAX_DEPTH = 100      # escape-time iteration cap (higher => sharper boundary detail)
SMOOTHNESS = 50.0    # smoothMandelbrot smoothing constant
ESCAPE_R = 2.0       # escape radius

_LOG2 = math.log(2.0)


def smooth(iters: torch.Tensor) -> torch.Tensor:
    """Map an (continuous) escape-iteration count to (0, 1].

    1 - 1 / ((iters / SMOOTHNESS) + 1). Monotonic in iters; -> 1.0 as iters -> inf.
    """
    return 1.0 - 1.0 / (iters / SMOOTHNESS + 1.0)


@torch.no_grad()
def mandelbrot(coords: torch.Tensor, max_depth: int = MAX_DEPTH) -> torch.Tensor:
    """Compute the target for a batch of points.

    Args:
        coords: (N, 2) tensor of (real, imag). Any device/dtype; computed in float64.
        max_depth: escape-time iteration cap.

    Returns:
        (N,) float32 tensor of target values in (0, 1]. In-set points -> 1.0.
    """
    device = coords.device
    re = coords[:, 0].to(torch.float64)
    im = coords[:, 1].to(torch.float64)
    c = torch.complex(re, im)
    z = torch.zeros_like(c)

    nu = torch.zeros(c.shape[0], dtype=torch.float64, device=device)
    alive = torch.ones(c.shape[0], dtype=torch.bool, device=device)

    for n in range(max_depth):
        z = torch.where(alive, z * z + c, z)
        mag = z.abs()
        escaped = alive & (mag > ESCAPE_R)
        if escaped.any():
            # Continuous escape count: n + 1 - log2(log|z|/log2)
            log_zn = torch.log(mag[escaped].clamp_min(1e-12))
            cont = (n + 1) - torch.log2(log_zn / _LOG2)
            nu[escaped] = cont.clamp_min(0.0)
        alive = alive & ~escaped
        if not alive.any():
            break

    target = smooth(nu)
    target[alive] = 1.0  # never escaped -> in the set
    return target.to(torch.float32)


def make_grid(resx: int, resy: int, device="cpu",
              xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX) -> torch.Tensor:
    """Return an (resx*resy, 2) tensor of (real, imag) over the view window.

    Row-major over a (resy, resx) image: index = row * resx + col, row top->bottom
    corresponds to imag ymax->ymin (image convention), col left->right real xmin->xmax.
    Reshape the predictions to (resy, resx) to render an image.
    """
    xs = torch.linspace(xmin, xmax, resx, device=device)
    ys = torch.linspace(ymax, ymin, resy, device=device)  # top row = ymax
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


# True aspect ratio of the view window (width / height ~ 1.59). Render grids should
# use this so pixels are square in coordinate space and the fractal isn't distorted.
ASPECT = (XMAX - XMIN) / (YMAX - YMIN)


def aspect_grid(height: int, device="cpu"):
    """Aspect-correct render grid. Width is chosen so coordinate-space pixels are
    square. Returns (coords[(h*w), 2], width, height); reshape preds to (height, width)."""
    width = max(1, round(height * ASPECT))
    return make_grid(width, height, device=device), width, height


def sample_uniform(n: int, generator: torch.Generator = None, device="cpu") -> torch.Tensor:
    """Return (n, 2) uniform random points over the view window."""
    u = torch.rand(n, 2, generator=generator, device=device)
    x = XMIN + u[:, 0] * (XMAX - XMIN)
    y = YMIN + u[:, 1] * (YMAX - YMIN)
    return torch.stack([x, y], dim=1)
