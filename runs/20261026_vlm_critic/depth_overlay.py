"""Depth-coloured candidate paths on the wrist camera (no point cloud, no virtual view).
For each candidate chunk from the noise probe at (tag, t0): execute it from the saved state to get the gripper path and
glyphs (as in overlay_views.py), project the path into the wrist camera at t0, and compare each path point's depth along
the camera axis with the wrist depth map at that pixel: clearance = scene depth - point depth (how far in front of the
nearest surface the gripper still is; for the downward-looking wrist camera this is about the height above it).
The path is coloured by clearance (green >= 10 cm, yellow 5-10, orange 2-5, red < 2 = at or touching the surface), the
end point is labelled with its clearance, and steps that command the gripper closed get a magenta ring with the
clearance there (closing far from any surface = closing on air).
Writes <out>/depth_<view>.png (colourised depth), cand<i>_depth.png, fan_depth.png, meta.json.
usage (groot env): python depth_overlay.py <out dir> <tag> <t0>"""
import json, glob, os, sys
import numpy as np
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)
from atomic_eval import load_scene  # noqa: E402
from oracle import set_state  # noqa: E402
from overlay_views import FINGERS, draw_glyph, UP  # noqa: E402
from robosuite.utils.camera_utils import (get_camera_transform_matrix, get_camera_extrinsic_matrix,  # noqa: E402
                                          get_real_depth_map, project_points_from_world_to_camera)

RUNS = os.path.join(HERE, "..")
CAM = "robot0_eye_in_hand"
FONT = None
for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
    if os.path.exists(f):
        FONT = ImageFont.truetype(f, 30); break


def clr_colour(c):
    if c is None: return (150, 150, 150)  # outside the wrist image: unknown
    if c >= 0.10: return (40, 200, 90)
    if c >= 0.05: return (240, 210, 30)
    if c >= 0.02: return (255, 140, 0)
    return (230, 30, 30)


def main():
    out, tag, t0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
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
    d = kenv.sim.data._data
    o = base.get_observation(kenv._get_observations(force_update=True))
    rgb = o["video.robot0_eye_in_hand"]; H, W = rgb.shape[:2]
    # depth of the environment only: the wrist camera sees the robot's own fingers, so render depth with the robot's
    # geoms made invisible (on a real robot: mask the robot from its known geometry and fill from earlier frames)
    import mujoco
    m = kenv.sim.model._model
    robot = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("robot0_", "gripper0_", "mobilebase0_", "mount0_"))]
    alpha = m.geom_rgba[robot, 3].copy(); m.geom_rgba[robot, 3] = 0
    rend = mujoco.Renderer(m, height=H, width=W); rend.enable_depth_rendering()
    opt = mujoco.MjvOption(); opt.sitegroup[:] = 0  # sites (e.g. the grip-site marker) are visual aids, not surfaces
    rend.update_scene(d, camera=CAM, scene_option=opt); depth = rend.render().copy()  # metres, top row first like rgb
    m.geom_rgba[robot, 3] = alpha
    _, dep = kenv.sim.render(camera_name=CAM, width=W, height=H, depth=True)
    depth_robot = get_real_depth_map(kenv.sim, dep[::-1])
    print(f"hid {len(robot)} robot geoms; median |depth with robot - without| on unmasked pixels:",
          round(float(np.median(np.abs(depth - depth_robot)[np.abs(depth - depth_robot) < 0.01])), 4))
    P = get_camera_transform_matrix(kenv.sim, CAM, H, W)
    world2cam = np.linalg.inv(get_camera_extrinsic_matrix(kenv.sim, CAM))
    dv = np.clip((depth - depth.min()) / (np.percentile(depth, 98) - depth.min() + 1e-6), 0, 1)
    dimg = (np.stack([1 - dv, 0.4 + 0.0 * dv, dv], -1) * 255).astype(np.uint8)  # near = red, far = blue
    Image.fromarray(dimg).resize((W * UP, H * UP), Image.NEAREST).save(f"{out}/depth_wrist.png")
    big = np.asarray(Image.fromarray(rgb).resize((W * UP, H * UP), Image.BICUBIC))
    Image.fromarray(big).save(f"{out}/base_wrist.png")
    site = kenv.robots[0].eef_site_id["right"]
    fingers = [kenv.sim.model.body_name2id(b) for b in FINGERS]

    def clearance(p):
        """(clearance in m, or None when the point is outside the wrist image, (row, col))."""
        rc = project_points_from_world_to_camera(p[None], P, H, W)[0]
        zc = (world2cam @ np.append(p, 1))[2]
        r, c = int(rc[0]), int(rc[1])
        if not (0 <= r < H and 0 <= c < W) or zc <= 0:
            return None, rc
        return float(depth[r, c] - zc), rc

    meta, rec = [], {}
    for c in sorted(probe):
        r = probe[c]
        chunk = {k: np.asarray(v) for k, v in r["first_chunk"].items()}
        close = chunk["action.gripper_close"][:, 0] > 0.5
        set_state(kenv, states[t0])
        g = lambda: np.array([d.xpos[f] for f in fingers])
        path, end_g = [d.site_xpos[site].copy()], None
        for i in range(16):
            base.step({k: v[i] for k, v in chunk.items()})
            path.append(d.site_xpos[site].copy())
        end_g = g()
        cl = [clearance(p) for p in path]
        rec[c] = (cl, close, end_g)
        im = Image.fromarray(big.copy()); dr = ImageDraw.Draw(im)
        xy = [(rc[1] * UP, rc[0] * UP) for _, rc in cl]
        for i in range(len(xy) - 1):
            dr.line([xy[i], xy[i + 1]], fill=clr_colour(cl[i + 1][0]), width=7)
        draw_glyph(dr, project_points_from_world_to_camera(end_g, P, H, W), (255, 255, 255), 3)
        closes = [i for i in range(16) if close[i]]
        for i in closes:
            x, y = xy[i + 1]
            dr.ellipse([x - 12, y - 12, x + 12, y + 12], outline=(255, 60, 220), width=4)
        x, y = xy[-1]
        label = f"{max(cl[-1][0], 0) * 100:.0f} cm" if cl[-1][0] is not None else "?"
        dr.text((x + 14, y - 34), label, fill=(255, 255, 255), font=FONT, stroke_width=4, stroke_fill=(0, 0, 0))
        if closes:
            first = closes[0]
            cc = cl[first + 1][0]
            dr.text((xy[first + 1][0] + 14, xy[first + 1][1] + 4), f"close @ {max(cc, 0) * 100:.0f} cm" if cc is not None else "close @ ?",
                    fill=(255, 120, 230), font=FONT, stroke_width=4, stroke_fill=(0, 0, 0))
        im.save(f"{out}/cand{c}_depth.png")
        meta.append(dict(cand=c, lift_cm=round(100 * r["kettle_lift_m"], 1), stage0=bool(r["stage_steps"]), success=r["success"],
                         end_clearance_cm=None if cl[-1][0] is None else round(100 * cl[-1][0], 1),
                         min_clearance_cm=round(100 * min([x for x, _ in cl if x is not None] or [float("nan")]), 1),
                         close_steps=closes,
                         clearance_at_first_close_cm=(None if not closes or cl[closes[0] + 1][0] is None else round(100 * cl[closes[0] + 1][0], 1))))
    im = Image.fromarray(big.copy()); dr = ImageDraw.Draw(im)
    for c in sorted(rec, key=lambda c: probe[c]["kettle_lift_m"] >= 0.05):
        cl = rec[c][0]; xy = [(rc[1] * UP, rc[0] * UP) for _, rc in cl]
        ok = probe[c]["kettle_lift_m"] >= 0.05
        for i in range(len(xy) - 1):
            dr.line([xy[i], xy[i + 1]], fill=clr_colour(cl[i + 1][0]), width=6 if ok else 3)
        x, y = xy[-1]
        dr.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 255, 255) if ok else (60, 60, 60))
    im.save(f"{out}/fan_depth.png")
    json.dump(dict(tag=tag, t0=t0, candidates=meta), open(f"{out}/meta.json", "w"), indent=1)
    for mm in meta:
        print(mm)


if __name__ == "__main__":
    main()
