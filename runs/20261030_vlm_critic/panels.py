"""Blind candidate panels for a VLM critic (2026-10-30): what a critic sees at a decision point, without the simulator.

Points: the critical chunks (confirm) and the random control points (control) of ../20261026_critic_data, all 3 tasks.
Per point, under panels/<pid>/ (pid = p01.. in a fixed shuffled order, so names and order reveal neither the dataset nor
the outcome):
  - context.png: the three cameras GR00T sees (left, right, wrist; the 480 px renders of the decision point), the end
    effector marked;
  - left.png, right.png, wrist.png: 8 thumbnails per camera, one per candidate (#0-#7), each with the candidate's
    commanded end-effector path over its 16-step chunk (commanded.py: from the action and the observation alone),
    dots where the gripper is commanded closed, and a gripper glyph at the end (palm + fingers, open or closed as
    commanded);
  - prompt.json: the task instruction and the current stage's goal.
Written outside panels/: key.json (pid -> task, dataset, tag, t, stage) and truth.json (pid -> each candidate's
success rate over its 4 continuations). Neither is shown to a critic.
usage: python -I panels.py"""
import json, os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # runs/
sys.path[:0] = [f"{R}/20261028_critical", HERE]
import data  # noqa: E402
from commanded import commanded, first_chunk  # noqa: E402

CAMS = [("left", "robot0_agentview_left"), ("right", "robot0_agentview_right"), ("wrist", "robot0_eye_in_hand")]
# finger bodies in the end-effector site frame (x = opening direction, z = approach): [left base, right base, left tip,
# right tip], measured once on the robot model (fully open and fully closed)
FING = {"open": np.array([[-0.04, 0, -0.0446], [0.04, 0, -0.0446], [-0.0485, 0, 0.0114], [0.0485, 0, 0.0114]]),
        "closed": np.array([[0, 0, -0.0446], [0, 0, -0.0446], [-0.0085, 0, 0.0114], [0.0085, 0, 0.0114]])}
STAGE_GOAL = {
    "KettleBoiling": ["pick up the kettle from the counter and place it on a stove burner",
                      "turn on the burner under the kettle"],
    "SteamInMicrowave": ["pick up the vegetable from the sink and place it in the bowl on the counter",
                         "pick up the bowl (with the vegetable) and place it inside the microwave",
                         "close the microwave door", "press the microwave's start button"],
    "PackIdenticalLunches": ["take the next food item from the fridge and place it in an empty spot of a tupperware on the counter"] * 3
                            + ["take the last food item from the fridge and place it in the tupperware that is still missing it"],
}
TH, PAD = 300, 6  # thumbnail size, gap


def project(Pw, K, T, H=480, W=480):
    """(x, y) pixel per world point, None behind the camera (robosuite camera looks along +z of its frame)"""
    K4 = np.eye(4); K4[:3, :3] = K
    Pw = np.asarray(Pw, float)
    ph = np.c_[Pw, np.ones(len(Pw))]
    cam = (np.linalg.inv(T) @ ph.T)
    pix = (K4 @ np.linalg.inv(T) @ ph.T)
    out = []
    for i in range(len(Pw)):
        if cam[2, i] <= 0.02:
            out.append(None); continue
        u, v = pix[0, i] / pix[2, i], pix[1, i] / pix[2, i]
        out.append((float(u), float(v)))
    return out


def _font(n):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, n)
    return ImageFont.load_default()


def grad(i, n):
    """time colour along the path: yellow at the start to red at the end"""
    f = i / max(n - 1, 1)
    return (255, int(225 * (1 - f)), int(40 * (1 - f)))


def line(d, a, b, col, w):
    if a is not None and b is not None:
        d.line([a, b], fill=(15, 15, 15), width=w + 2)
        d.line([a, b], fill=col, width=w)


def glyph_points(c):
    return c["pos"][-1] + FING["closed" if c["close"][-1] else "open"] @ c["rot"][-1].T


def crop_box(cands, K, T, margin=36, min_size=150):
    """one square window (x0, y0, size) in the 480 px image around every candidate's path and end glyph"""
    pts = [q for c in cands for q in project(np.r_[c["pos"], glyph_points(c)], K, T) if q is not None]
    if not pts:
        return 0, 0, 480
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    size = min(480, max(min_size, max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * margin))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    x0 = int(min(max(cx - size / 2, 0), 480 - size)); y0 = int(min(max(cy - size / 2, 0), 480 - size))
    return x0, y0, int(size)


def draw_candidate(rgb, K, T, c, label, box, glyph_w=3):
    """the candidate on the image crop `box`, scaled to TH px; lines are drawn after scaling, so their width is fixed"""
    x0, y0, size = box
    img = Image.fromarray(rgb).convert("RGB").crop((x0, y0, x0 + size, y0 + size)).resize((TH, TH), Image.LANCZOS)
    k = TH / size
    tr = lambda q: None if q is None else ((q[0] - x0) * k, (q[1] - y0) * k)
    d = ImageDraw.Draw(img)
    xy = [tr(q) for q in project(c["pos"], K, T)]
    for i in range(16):
        line(d, xy[i], xy[i + 1], grad(i, 16), 3)
    for i in np.nonzero(c["close"])[0]:  # steps commanding the gripper closed
        if xy[i + 1] is not None:
            x, y = xy[i + 1]; d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 255), outline=(15, 15, 15), width=2)
    if xy[0] is not None:
        x, y = xy[0]; d.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 255, 255), width=2)
    gw = [tr(q) for q in project(glyph_points(c), K, T)]
    for a, b in [(0, 1), (0, 2), (1, 3)]:
        line(d, gw[a], gw[b], (60, 220, 255), glyph_w)
    d.rectangle([0, 0, 52, 34], fill=(15, 15, 15))
    d.text((7, 3), label, fill=(255, 255, 255), font=_font(24))
    return img


def make_point(pid, r, out):
    s = np.load(data.snap(r))
    cands = []
    for i in range(8):
        chunk, _ = first_chunk(f"{data.roll(r)}/{r['tag']}_t{r['t']}_c{i}_m0.npz")
        cands.append(commanded(s, chunk))
    os.makedirs(out, exist_ok=True)
    # context
    ctx = Image.new("RGB", (3 * 480 + 2 * PAD, 480 + 34), (250, 250, 248))
    for j, (name, cam) in enumerate(CAMS):
        im = Image.fromarray(s[f"rgb/{cam}"]).convert("RGB")
        d = ImageDraw.Draw(im)
        p0 = project(cands[0]["pos"][:1], s[f"K/{cam}"], s[f"T/{cam}"])[0]
        if p0 is not None:
            x, y = p0; d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(255, 255, 255), width=3)
        ctx.paste(im, (j * (480 + PAD), 34))
        ImageDraw.Draw(ctx).text((j * (480 + PAD) + 6, 6), f"{name} camera (white ring: gripper now)", fill=(20, 20, 20), font=_font(20))
    ctx.save(f"{out}/context.png")
    # one grid per camera; the agent views are cropped to the same window around all 8 candidates (zoomed), the wrist
    # view is shown whole (the gripper is close to that camera)
    for name, cam in CAMS:
        K, T = s[f"K/{cam}"], s[f"T/{cam}"]
        box = (0, 0, 480) if name == "wrist" else crop_box(cands, K, T)
        grid = Image.new("RGB", (4 * TH + 3 * PAD, 2 * TH + PAD + 30), (250, 250, 248))
        zoom = f", zoomed {480 / box[2]:.1f}x on the gripper's region" if box[2] < 480 else ""
        ImageDraw.Draw(grid).text((6, 4), f"{name} camera{zoom}: candidates #0-#7", fill=(20, 20, 20), font=_font(20))
        for i, c in enumerate(cands):
            im = draw_candidate(s[f"rgb/{cam}"], K, T, c, f"#{i}", box, glyph_w=2 if name == "wrist" else 3)
            grid.paste(im, ((i % 4) * (TH + PAD), 30 + (i // 4) * (TH + PAD)))
        grid.save(f"{out}/{name}.png")
    instr = str(np.asarray(s["req/annotation.human.task_description"]).reshape(-1)[0])
    base_moves = [bool(c["base_mode"].any()) for c in cands]
    json.dump(dict(task_instruction=instr, stage=int(r["k"]), n_stages=len(STAGE_GOAL[r["task"]]), stage_goal=STAGE_GOAL[r["task"]][r["k"]],
                   candidates_moving_base=[i for i, b in enumerate(base_moves) if b]), open(f"{out}/prompt.json", "w"), indent=1)
    return [float(x) for x in data.outcomes(r).mean(1)]


def main():
    pts = [(task, ds, r) for task in data.TASKS for ds in ("confirm", "control") for r in sorted(data.rows(task, ds).values(), key=lambda r: (r["tag"], r["t"]))]
    order = list(range(len(pts)))
    random.Random(780).shuffle(order)
    key, truth = {}, {}
    for n, j in enumerate(order):
        task, ds, r = pts[j]
        pid = f"p{n + 1:02d}"
        truth[pid] = make_point(pid, r, os.path.join(HERE, "panels", pid))
        key[pid] = dict(task=task, dataset=ds, tag=r["tag"], t=r["t"], stage=r["k"])
        print(pid, task, ds, r["tag"], r["t"], "rates", truth[pid], flush=True)
    json.dump(key, open(os.path.join(HERE, "key.json"), "w"), indent=1)
    json.dump(truth, open(os.path.join(HERE, "truth.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
