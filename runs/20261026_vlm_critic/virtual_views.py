"""Virtual side views for the candidate overlay: build a world-frame point cloud from the robot's three RGB-D cameras
(agent view left / right, wrist), render it from a virtual horizontal camera beside the gripper, and draw each
candidate's 3D gripper path and gripper glyphs (start white, end red) in that view. For reference, the simulator also
renders the true scene from the same virtual camera. Also measures, per candidate, how far the fingertips descend in
the chunk and their clearance to whatever is straight below them (ray cast), at the start and at the end.
Writes <out>/side_pc_base.png, side_ideal_base.png, cand<i>_side_pc.png, cand<i>_side_ideal.png, fan_side_{pc,ideal}.png,
meta.json. usage (groot env): python virtual_views.py <out dir> [tag] [t0]"""
import json, glob, os, sys
import numpy as np
import mujoco
import gymnasium as gym
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, HERE)
from atomic_eval import load_scene  # noqa: E402
from oracle import set_state  # noqa: E402
from overlay_views import FINGERS, grad  # noqa: E402
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map  # noqa: E402

RUNS = os.path.join(HERE, "..")
CAMS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
RES = 480      # source camera resolution for the point cloud (offscreen buffer is 640x480)
VRES = 480     # virtual view resolution


def cloud(kenv):
    """world-frame coloured point cloud from the three robot cameras (RGB + depth)."""
    pts, cols = [], []
    for c in CAMS:
        rgb, dep = kenv.sim.render(camera_name=c, width=RES, height=RES, depth=True)
        rgb, dep = rgb[::-1], dep[::-1]
        z = get_real_depth_map(kenv.sim, dep)
        K = get_camera_intrinsic_matrix(kenv.sim, c, RES, RES)
        T = get_camera_extrinsic_matrix(kenv.sim, c)  # camera -> world
        v, u = np.mgrid[0:RES, 0:RES]
        keep = z < 2.5
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        P = np.stack([x[keep], y[keep], z[keep], np.ones(keep.sum())], 1) @ T.T
        pts.append(P[:, :3]); cols.append(rgb[keep])
    return np.concatenate(pts), np.concatenate(cols)


class VCam:
    """pinhole camera from MuJoCo's free-camera pose (pos, forward, up) and the model's fovy."""
    def __init__(self, scam, fovy, res):
        self.pos, self.fwd, self.up = np.array(scam.pos), np.array(scam.forward), np.array(scam.up)
        self.right = np.cross(self.fwd, self.up)
        self.f = (res / 2) / np.tan(np.deg2rad(fovy) / 2); self.res = res

    def project(self, P):
        r = np.asarray(P) - self.pos
        zc = r @ self.fwd
        u = self.res / 2 + self.f * (r @ self.right) / zc
        v = self.res / 2 - self.f * (r @ self.up) / zc
        return np.stack([u, v], -1), zc


def splat(cam, pts, cols, res):
    uv, zc = cam.project(pts)
    ok = (zc > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < res) & (uv[:, 1] >= 0) & (uv[:, 1] < res)
    uv, zc, c = uv[ok].astype(int), zc[ok], cols[ok]
    img = np.full((res, res, 3), 235, np.uint8); zbuf = np.full((res, res), np.inf)
    order = np.argsort(-zc)  # far first, near overwrites
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = np.clip(uv[order, 0] + du, 0, res - 1), np.clip(uv[order, 1] + dv, 0, res - 1)
            img[vv, uu] = c[order]
    return img


def draw(img, cam, path, glyphs, colour=None, width=5):
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    d = ImageDraw.Draw(im)
    xy = [tuple(p) for p in cam.project(path)[0]]
    n = len(xy) - 1
    for i in range(n):
        d.line([xy[i], xy[i + 1]], fill=colour or grad(i, n), width=width)
    for kind, g in glyphs:
        q = [tuple(p) for p in cam.project(g)[0]]
        col = colour or {"start": (255, 255, 255), "close": (255, 60, 220), "end": (255, 40, 0)}[kind]
        segs = [(q[0], q[1]), (q[0], q[2]), (q[1], q[3])]
        for a, b in segs:
            d.line([a, b], fill=(20, 20, 20), width=width + 3)
        for a, b in segs:
            d.line([a, b], fill=col, width=max(width - 1, 2))
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
    m, d = kenv.sim.model._model, kenv.sim.data._data  # after load_scene: the model is rebuilt there
    bid = lambda n: kenv.sim.model.body_name2id(n)
    tips = [bid("gripper0_right_finger_joint1_tip"), bid("gripper0_right_finger_joint2_tip")]
    fingers = [bid(b) for b in FINGERS]
    site = kenv.robots[0].eef_site_id["right"]

    def tipmid():
        return (d.xpos[tips[0]] + d.xpos[tips[1]]) / 2

    def clearance():
        p = tipmid(); gid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(m, d, p - [0, 0, 0.012], np.array([0, 0, -1.0]), None, 1, -1, gid)
        root = m.body(m.body_rootid[m.geom_bodyid[gid[0]]]).name if gid[0] >= 0 else None
        return dist + 0.012, root

    # virtual horizontal camera beside the gripper-kettle pair, looking across the approach direction
    kettle = np.array(d.xpos[kenv.obj_body_id["obj"]]); tip0 = tipmid()
    arm = np.array(d.xpos[bid("robot0_link0")])
    appr = kettle[:2] - arm[:2]; appr /= np.linalg.norm(appr)
    side = np.array([-appr[1], appr[0]])
    target = (kettle + tip0) / 2
    vcam = mujoco.MjvCamera(); vcam.type = mujoco.mjtCamera.mjCAMERA_FREE
    vcam.lookat[:] = target; vcam.distance = 0.75
    # camera on the -side of the gripper-kettle pair (in s2 the +side camera ends up inside a cabinet)
    vcam.azimuth = float(np.degrees(np.arctan2(side[1], side[0]))); vcam.elevation = -8.0
    renderer = mujoco.Renderer(m, height=VRES, width=VRES)
    opt = mujoco.MjvOption(); opt.geomgroup[0] = 0; opt.sitegroup[:] = 0  # hide collision geoms and site markers, as robosuite's renders do
    renderer.update_scene(d, camera=vcam, scene_option=opt)
    cam = VCam(renderer.scene.camera[0], m.vis.global_.fovy, VRES)
    ideal0 = renderer.render().copy()
    pts, cols = cloud(kenv)
    pc0 = splat(cam, pts, cols, VRES)
    Image.fromarray(pc0).save(f"{out}/side_pc_base.png"); Image.fromarray(ideal0).save(f"{out}/side_ideal_base.png")

    c0, below0 = clearance()
    meta, rec = dict(start_clearance_cm=round(100 * c0, 1), start_below=below0, candidates=[]), {}
    for c in sorted(probe):
        r = probe[c]
        chunk = {k: np.asarray(v) for k, v in r["first_chunk"].items()}
        close = chunk["action.gripper_close"][:, 0] > 0.5
        set_state(kenv, states[t0])
        g = lambda: np.array([d.xpos[f] for f in fingers])
        path, glyphs = [d.site_xpos[site].copy()], [("start", g())]
        for i in range(16):
            base.step({k: v[i] for k, v in chunk.items()})
            path.append(d.site_xpos[site].copy())
            if close[i]:
                glyphs.append(("close", g()))
        glyphs.append(("end", g()))
        cl, below = clearance(); tip = tipmid()
        rec[c] = (np.array(path), glyphs)
        draw(pc0.copy(), cam, rec[c][0], glyphs).save(f"{out}/cand{c}_side_pc.png")
        draw(ideal0.copy(), cam, rec[c][0], glyphs).save(f"{out}/cand{c}_side_ideal.png")
        meta["candidates"].append(dict(cand=c, lift_cm=round(100 * r["kettle_lift_m"], 1), stage0=bool(r["stage_steps"]),
                                       descend_cm=round(100 * (tip0[2] - tip[2]), 1),
                                       move_xy_cm=round(100 * float(np.linalg.norm(tip[:2] - tip0[:2])), 1),
                                       end_clearance_cm=round(100 * cl, 1), end_below=below))
    for kind, img0 in (("pc", pc0), ("ideal", ideal0)):
        im = Image.fromarray(img0.copy())
        for c in sorted(rec, key=lambda c: probe[c]["kettle_lift_m"] >= 0.05):
            ok = probe[c]["kettle_lift_m"] >= 0.05
            im = draw(im, cam, rec[c][0], [rec[c][1][-1]], (40, 200, 90) if ok else (150, 150, 150), 5 if ok else 3)
        im.save(f"{out}/fan_side_{kind}.png")
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=0)[:3000])


if __name__ == "__main__":
    main()
