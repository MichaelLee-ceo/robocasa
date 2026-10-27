"""3D-aware version of the candidate overlay (see overlay_demo.py), for the same probe state:
(A) the same 3D gripper path drawn in all three cameras GR00T sees (agent view left, agent view right, wrist), and
(C) a gripper glyph at the start and at the end of the chunk (and at steps that command the gripper closed): palm line
between the two finger bases plus the two fingers to their tips, from the sim's finger bodies, projected into each
camera. The glyph shows how the fingers are oriented relative to the handle, which a 2D path line cannot.
Writes <out>/base_<view>.png, cand<i>_<view>.png, fan_<view>.png for view in left/right/wrist, and meta.json.
usage (groot env): python overlay_views.py <out dir> [tag] [t0]"""
import json, glob, os, sys
import numpy as np
import gymnasium as gym
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from atomic_eval import load_scene  # noqa: E402
from oracle import set_state  # noqa: E402
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera  # noqa: E402

RUNS = os.path.join(HERE, "..")
CAMS = {"left": "robot0_agentview_left", "right": "robot0_agentview_right", "wrist": "robot0_eye_in_hand"}
FINGERS = ["gripper0_right_leftfinger", "gripper0_right_rightfinger", "gripper0_right_finger_joint1_tip", "gripper0_right_finger_joint2_tip"]
UP = 3


def grad(i, n):
    return (255, int(220 * (1 - i / max(n, 1))), 0)


def gripper_pts(kenv):
    """[left finger base, right finger base, left tip, right tip] in world coordinates."""
    return np.array([kenv.sim.data.body_xpos[kenv.sim.model.body_name2id(b)] for b in FINGERS])


def draw_path(d, px, colour=None, width=5, dots=True):
    xy = [(float(p[1]) * UP, float(p[0]) * UP) for p in px]
    n = len(xy) - 1
    for i in range(n):
        d.line([xy[i], xy[i + 1]], fill=colour or grad(i, n), width=width)
    if dots:
        for i, (x, y) in enumerate(xy[1:]):
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=colour or grad(i, n - 1))


def draw_glyph(d, px, colour, width=4):
    """px: projected [lbase, rbase, ltip, rtip]: palm line + two fingers, with a dark outline for contrast."""
    xy = [(float(p[1]) * UP, float(p[0]) * UP) for p in px]
    segs = [(xy[0], xy[1]), (xy[0], xy[2]), (xy[1], xy[3])]
    for a, b in segs:
        d.line([a, b], fill=(20, 20, 20), width=width + 4)
    for a, b in segs:
        d.line([a, b], fill=colour, width=width)
    for p in xy[2:]:
        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=colour, outline=(20, 20, 20), width=2)


def main():
    out = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "KettleBoiling_s2_single"
    t0 = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    task = tag.split("_")[0]
    os.makedirs(out, exist_ok=True)
    probe = {r["cand"]: r for p in glob.glob(f"{RUNS}/20261026_noise_probe/results_w*.jsonl") for r in map(json.loads, open(p))
             if r.get("tag") == tag and r.get("t0") == t0}
    scene = {j["tag"]: j["scene"] for j in json.load(open(f"{RUNS}/20261025_seen480/run/jobs.json"))}[tag]
    states = np.load(f"{RUNS}/20261025_seen480/run/{task}/{tag}_steps.npz")["states"]
    base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
    kenv = base.env
    load_scene(kenv, scene)
    set_state(kenv, states[t0])
    o = base.get_observation(kenv._get_observations(force_update=True))
    imgs = {v: o[f"video.{c}"] for v, c in CAMS.items()}
    H, W = imgs["left"].shape[:2]
    mats = {v: get_camera_transform_matrix(kenv.sim, c, H, W) for v, c in CAMS.items()}
    big = {v: np.asarray(Image.fromarray(im).resize((W * UP, H * UP), Image.BICUBIC)) for v, im in imgs.items()}
    for v in big:
        Image.fromarray(big[v]).save(f"{out}/base_{v}.png")
    site = kenv.robots[0].eef_site_id["right"]
    proj = lambda pts, v: project_points_from_world_to_camera(np.asarray(pts), mats[v], H, W)
    meta, rec = [], {}
    for c in sorted(probe):
        r = probe[c]
        chunk = {k: np.asarray(v) for k, v in r["first_chunk"].items()}
        close = chunk["action.gripper_close"][:, 0] > 0.5
        set_state(kenv, states[t0])
        path, glyphs = [kenv.sim.data.site_xpos[site].copy()], [("start", gripper_pts(kenv))]
        for i in range(16):
            base.step({k: v[i] for k, v in chunk.items()})
            path.append(kenv.sim.data.site_xpos[site].copy())
            if close[i]:
                glyphs.append(("close", gripper_pts(kenv)))
        glyphs.append(("end", gripper_pts(kenv)))
        rec[c] = (np.array(path), glyphs)
        for v in CAMS:
            im = Image.fromarray(big[v].copy()); d = ImageDraw.Draw(im)
            draw_path(d, proj(path, v))
            for kind, g in glyphs:
                draw_glyph(d, proj(g, v), {"start": (255, 255, 255), "close": (255, 60, 220), "end": (255, 40, 0)}[kind])
            im.save(f"{out}/cand{c}_{v}.png")
        meta.append(dict(cand=c, lift_cm=round(100 * r["kettle_lift_m"], 1), stage0=bool(r["stage_steps"]),
                         closes=int(close.sum())))
    for v in CAMS:
        im = Image.fromarray(big[v].copy()); d = ImageDraw.Draw(im)
        order = sorted(rec, key=lambda c: probe[c]["kettle_lift_m"] >= 0.05)
        for c in order:
            ok = probe[c]["kettle_lift_m"] >= 0.05
            path, glyphs = rec[c]
            col = (40, 200, 90) if ok else (205, 205, 205)
            draw_path(d, proj(path, v), col, 6 if ok else 3, dots=False)
            draw_glyph(d, proj(glyphs[-1][1], v), col, 4 if ok else 2)
        im.save(f"{out}/fan_{v}.png")
    json.dump(dict(tag=tag, t0=t0, candidates=meta), open(f"{out}/meta.json", "w"), indent=1)
    print(out, len(meta), "candidates")


if __name__ == "__main__":
    main()
