"""Contact sheets from rollout videos, for checking stage verdicts by eye.
The rollout mp4 holds one frame every 4 control steps (frame i = after step 4*(i+1)), each [agentview_left | eye_in_hand],
each view 256x256.
usage:
  python frames.py <mp4> <out png> <step> [<step> ...] [--view both|left|wrist] [--zoom X0 Y0 X1 Y1] [--scale K]
  python frames.py --compare <out png> <mp4>:<step> [<mp4>:<step> ...] [--view ...] [--zoom ...] [--scale K]
--zoom crops the chosen view(s) to pixels [X0, X1) x [Y0, Y1) of the 256x256 view and --scale enlarges each tile
(default 2 when zooming, else 1). --compare puts one tile per (video, step), labelled with the video name."""
import argparse, os
import av
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("items", nargs="+")
ap.add_argument("--compare", action="store_true")
ap.add_argument("--view", choices=["both", "left", "wrist"], default="both")
ap.add_argument("--zoom", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
ap.add_argument("--scale", type=int)
ap.add_argument("--cols", type=int, default=2)
a = ap.parse_args()
scale = a.scale or (2 if a.zoom else 1)
_cache = {}


def frames(mp4):
    if mp4 not in _cache:
        _cache[mp4] = [f.to_ndarray(format="rgb24") for f in av.open(mp4).decode(video=0)]
    return _cache[mp4]


def tile(mp4, step, label):
    fr = frames(mp4)
    i = min(max(step // 4 - 1, 0), len(fr) - 1)
    img = fr[i]
    views = {"left": img[:, :256], "wrist": img[:, 256:512]}
    parts = [views["left"], views["wrist"]] if a.view == "both" else [views[a.view]]
    if a.zoom:
        x0, y0, x1, y1 = a.zoom
        parts = [p[y0:y1, x0:x1] for p in parts]
    im = Image.fromarray(np.concatenate(parts, 1))
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    ImageDraw.Draw(im).text((4, 4), f"{label} step {4 * (i + 1)}", fill=(255, 255, 0))
    return np.asarray(im)


if a.compare:
    out, pairs = a.items[0], a.items[1:]
    tiles = []
    for pr in pairs:
        mp4, st = pr.rsplit(":", 1)
        tiles.append(tile(mp4, int(st), os.path.basename(mp4).replace("_single.mp4", "")))
else:
    mp4, out, steps = a.items[0], a.items[1], [int(x) for x in a.items[2:]]
    tiles = [tile(mp4, s, "") for s in steps]
blank = np.zeros_like(tiles[0])
rows = [np.concatenate(tiles[r:r + a.cols] + [blank] * (a.cols - len(tiles[r:r + a.cols])), 1)
        for r in range(0, len(tiles), a.cols)]
Image.fromarray(np.concatenate(rows, 0)).save(out)
print(out, len(tiles), "tiles")
