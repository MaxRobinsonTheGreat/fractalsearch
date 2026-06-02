"""Generate the static 4K ground-truth reference image.

The dashboard serves a committed PNG (dashboard/static/groundtruth.png) rather than
rendering the target on the fly. The image is rendered on the SAME grid the evaluator
scores against (harness.evaluate.EVAL_RESX x EVAL_RESY), so it is pixel-aligned with the
per-run prediction.png / error.png layers.

Re-run this whenever the view window (harness/groundtruth.py) or the target definition
changes, then commit the regenerated image:

    uv run python -m dashboard.make_groundtruth
"""

import os

import torch
from PIL import Image

from harness import groundtruth as gt
from harness import colormap as cm
from harness.evaluate import EVAL_RESX, EVAL_RESY

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "groundtruth.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords = gt.make_grid(EVAL_RESX, EVAL_RESY, device=device)
    img = gt.mandelbrot(coords).reshape(EVAL_RESY, EVAL_RESX).cpu().numpy()
    Image.fromarray(cm.apply(img, cm.VALUE_CMAP)).save(OUT)
    print(f"wrote {OUT}  ({EVAL_RESX}x{EVAL_RESY}, "
          f"window X[{gt.XMIN},{gt.XMAX}] Y[{gt.YMIN},{gt.YMAX}])")


if __name__ == "__main__":
    main()
