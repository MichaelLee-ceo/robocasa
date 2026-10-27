"""Illustration of the "candidate trajectory overlay" for a VLM critic.
From a saved state (default: KettleBoiling s2, step 96, just before the grasp), take the 16 candidate action chunks
GR00T sampled there in the noise probe (runs/20261026_noise_probe). Execute each chunk's 16 steps from the saved state,
record the gripper position per step, and draw the path onto the camera images at that state (agent view + wrist view).
Colour runs from yellow (now) to red (0.8 s later); a white ring marks steps where the chunk commands the gripper closed.
Also draws all candidates on one image, green if the candidate (with the policy continuing) later lifted the kettle,
grey otherwise. (A deployable version would draw the commanded path from the robot's kinematics instead of executing it.)
Writes <out dir>/base_{agent,wrist}.png, cand<i>_{agent,wrist}.png, fan_{agent,wrist}.png and meta.json.
usage (groot env): python overlay_demo.py <out dir> [tag] [t0]"""
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
CAMS = {"agent": "robot0_agentview_left", "wrist": "robot0_eye_in_hand"}
UP = 3  # upscale before drawing, for crisp lines


def draw(img, px, close=None, colour=None, width=5):
    """img: HxWx3 uint8; px: (n, 2) (row, col) points; colour None = yellow->red gradient."""
    H, W = img.shape[:2]
    im = Image.fromarray(img) if img.shape[0] != H * UP else Image.fromarray(img)
    d = ImageDraw.Draw(im)
    xy = [(float(p[1]) * UP, float(p[0]) * UP) for p in px]
    n = len(xy) - 1
    for i in range(n):
        c = colour or (255, int(220 * (1 - i / n)), 0)
        d.line([xy[i], xy[i + 1]], fill=c, width=width)
    for i, (x, y) in enumerate(xy[1:]):
        c = colour or (255, int(220 * (1 - i / max(n - 1, 1))), 0)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=c)
        if close is not None and close[i]:
            d.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(255, 255, 255), width=3)
    return im


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
    imgs = {"agent": o["video.robot0_agentview_left"], "wrist": o["video.robot0_eye_in_hand"]}
    H, W = imgs["agent"].shape[:2]
    mats = {v: get_camera_transform_matrix(kenv.sim, c, H, W) for v, c in CAMS.items()}
    big = {v: np.asarray(Image.fromarray(im).resize((W * UP, H * UP), Image.BICUBIC)) for v, im in imgs.items()}
    for v in big:
        Image.fromarray(big[v]).save(f"{out}/base_{v}.png")
    site = kenv.robots[0].eef_site_id["right"]
    meta, paths = [], {}
    for c in sorted(probe):
        r = probe[c]
        chunk = {k: np.asarray(v) for k, v in r["first_chunk"].items()}
        set_state(kenv, states[t0])
        pts = [kenv.sim.data.site_xpos[site].copy()]
        for i in range(16):
            base.step({k: v[i] for k, v in chunk.items()})
            pts.append(kenv.sim.data.site_xpos[site].copy())
        pts = np.array(pts)
        close = (chunk["action.gripper_close"][:, 0] > 0.5).tolist()
        paths[c] = {v: project_points_from_world_to_camera(pts, mats[v], H, W) for v in CAMS}
        for v in CAMS:
            draw(big[v].copy(), paths[c][v], close).save(f"{out}/cand{c}_{v}.png")
        meta.append(dict(cand=c, lift_cm=round(100 * r["kettle_lift_m"], 1), stage0=bool(r["stage_steps"]),
                         success=r["success"], move_cm=round(100 * float(np.linalg.norm(pts[-1] - pts[0])), 1),
                         closes=int(sum(close))))
    for v in CAMS:
        im = Image.fromarray(big[v].copy())
        order = sorted(probe, key=lambda c: probe[c]["kettle_lift_m"] >= 0.05)  # lifted ones drawn last, on top
        for c in order:
            ok = probe[c]["kettle_lift_m"] >= 0.05
            im = draw(np.asarray(im).copy(), paths[c][v], None, (40, 200, 90) if ok else (200, 200, 200), 6 if ok else 3)
        im.save(f"{out}/fan_{v}.png")
    json.dump(dict(tag=tag, t0=t0, candidates=meta), open(f"{out}/meta.json", "w"), indent=1)
    print(out, len(meta), "candidates")


if __name__ == "__main__":
    main()
