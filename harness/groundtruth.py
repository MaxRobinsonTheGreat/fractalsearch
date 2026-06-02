"""Ground truth for fractalsearch — the single source of truth for the target.

Every solution learns the mapping  (real, imag) -> target value  defined here.

IMMUTABLE: solutions may import and read this module, but must NEVER modify it,
and the evaluation metric (see harness/evaluate.py) is computed against it. Changing
any constant or the target definition invalidates comparability across all logged runs.

If you (the human) want to change what is being fit (e.g. raw iteration count vs.
smooth normalized vs. binary membership), this is the ONE place to do it. Do it
deliberately and start a fresh run tag, because past results become incomparable.

Target definition (current): periodic log-distance.
    - escape-time iteration with the orbit derivative z' = dz/dc tracked alongside z,
      yielding a distance-to-set estimate  d = |z|·log|z| / |z'|  at the escape step.
    - mapped to a bounded, scale-free, PERIODIC value in [0, 1]:
          phase  = BETA · log(d)
          target = 0.5 + 0.5·sin(2π·phase)     (escaped / exterior)
          target = 1.0                          (never escaped / in-set)
    - Unlike a monotonic smooth-iter target this never saturates and exposes fine
      structure at every scale near the boundary (the band frequency -> inf as d -> 0).
      That is deliberate: it captures more of the fractal's fine complexity and is
      intentionally harder for the learner to fit.
    - Matches the "Periodic · log-distance" transform in
      dashboard/static/mandelbrot_lab.html (BETA = 0.050, R = 1e4 bailout).
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
MAX_DEPTH = 200      # escape-time iteration cap (higher => sharper boundary detail)
BETA = 0.050         # periodic log-distance frequency: phase = BETA * log(distance)
ESCAPE_R = 1.0e4     # escape radius — large bailout => accurate distance estimate
SMOOTHNESS = 50.0    # (legacy; unused by the periodic target, kept for smooth())

_LOG2 = math.log(2.0)
TWO_PI = 2.0 * math.pi


def smooth(iters: torch.Tensor) -> torch.Tensor:
    """Legacy monotonic smooth-iter squash, kept for reference / experimentation.

    1 - 1 / ((iters / SMOOTHNESS) + 1). Monotonic in iters; -> 1.0 as iters -> inf.
    Not used by the current periodic log-distance target.
    """
    return 1.0 - 1.0 / (iters / SMOOTHNESS + 1.0)


@torch.no_grad()
def mandelbrot(coords: torch.Tensor, max_depth: int = MAX_DEPTH) -> torch.Tensor:
    """Compute the periodic log-distance target for a batch of points.

    Iterates z <- z^2 + c while tracking the derivative z' = dz/dc; at escape this gives
    the distance-to-set estimate d = |z|*log|z| / |z'|, which is folded through
    phase = BETA*log(d) and a sine into a bounded, scale-free, periodic value. See the
    module docstring for the full definition.

    Args:
        coords: (N, 2) tensor of (real, imag). Any device/dtype; computed in float64.
        max_depth: escape-time iteration cap.

    Returns:
        (N,) float32 tensor of target values in [0, 1]. In-set points -> 1.0.
    """
    device = coords.device
    re = coords[:, 0].to(torch.float64)
    im = coords[:, 1].to(torch.float64)
    c = torch.complex(re, im)
    z = torch.zeros_like(c)
    dz = torch.zeros_like(c)             # orbit derivative z' = dz/dc (z'_0 = 0)

    z_esc = torch.zeros_like(c)          # z and z' captured at the escape iteration
    dz_esc = torch.zeros_like(c)
    alive = torch.ones(c.shape[0], dtype=torch.bool, device=device)

    for n in range(max_depth):
        # Derivative recurrence uses z BEFORE the z update: z'_{n+1} = 2*z_n*z'_n + 1.
        dz = torch.where(alive, 2.0 * z * dz + 1.0, dz)
        z = torch.where(alive, z * z + c, z)
        escaped = alive & (z.abs() > ESCAPE_R)
        if escaped.any():
            z_esc[escaped] = z[escaped]
            dz_esc[escaped] = dz[escaped]
        alive = alive & ~escaped
        if not alive.any():
            break

    # Distance-to-set estimate at escape: d = |z|*log|z| / |z'|.
    zmag = z_esc.abs()
    dzmag = dz_esc.abs().clamp_min(1e-20)
    dist = zmag * torch.log(zmag.clamp_min(1.0001)) / dzmag
    # Periodic, scale-free encoding: phase = BETA*log(d), folded through a sine -> [0,1].
    phase = BETA * torch.log(dist.clamp_min(1e-30))
    target = 0.5 + 0.5 * torch.sin(TWO_PI * phase)
    target[alive] = 1.0                  # never escaped -> in the set
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
