"""What a VLM critic would see at the critical points: the decision point's camera images with the 8 candidates' actual
end-effector paths over their chunk, coloured by outcome.

For the strict critical points (>= 2 candidates that finished the stage in both continuations and >= 2 in neither,
candidates.py) with the largest effect, 6 per task: each candidate's first 16 steps (rollout m0; both continuations
share them) are replayed without rendering (../20261025_seen480/analyze_stages.py:replay_env) to get the end effector
and the finger bodies per step. They are projected into the three cameras GR00T sees, using the decision point's saved
snapshot (rgb, K, T). Drawn per candidate:
  - the path;
  - dots at the steps that command the gripper closed;
  - the gripper glyph at the end of the chunk: palm line and two fingers (../20261026_vlm_critic/overlay_views.py).
Colours: blue = finished the stage in both continuations, orange = in neither, grey = one of two.
Labels, rollouts and snapshot all come from the coarse row in use (data.py).
Writes overlay/<tag>_t<t>.png (left | right | wrist) to $CRIT_OUT. usage: python -I overlay_critical.py <workers>"""
import json, multiprocessing, os, sys
import numpy as np

R = os.environ["ROBOCASA_ROOT"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", HERE]
import data  # noqa: E402
CAMS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
FINGERS = ["gripper0_right_leftfinger", "gripper0_right_rightfinger", "gripper0_right_finger_joint1_tip", "gripper0_right_finger_joint2_tip"]
COL = {2: (42, 120, 214), 1: (185, 184, 179), 0: (235, 104, 52)}


def project(Pw, K, T, H=480, W=480):
    """(row, col) per point, None for points less than 2 cm in front of the camera (they would project far off)"""
    from robosuite.utils.camera_utils import project_points_from_world_to_camera
    K4 = np.eye(4); K4[:3, :3] = K
    Pw = np.asarray(Pw)
    depth = (np.linalg.inv(T) @ np.c_[Pw, np.ones(len(Pw))].T)[2]  # camera looks along +z of its frame (robosuite)
    px = project_points_from_world_to_camera(Pw, K4 @ np.linalg.inv(T), H, W)
    return [q if z > 0.02 else None for q, z in zip(px, depth)]


def segs(d, pts, fill, width):
    for a, b in zip(pts[:-1], pts[1:]):
        if a is not None and b is not None:
            d.line([a, b], fill=fill, width=width)


def work(job):
    import atomic_eval  # noqa: F401
    from oracle import reset_to, set_state
    from analyze_stages import replay_env
    from PIL import Image, ImageDraw
    p, roll, snap = job
    try:
        z = np.load(p["scene"], allow_pickle=True)
        env = replay_env(p["task"])
        reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
        site = env.robots[0].eef_site_id["right"]
        fid = [env.sim.model.body_name2id(b) for b in FINGERS]
        paths = []
        for i in range(8):
            r = np.load(f"{roll}/{p['tag']}_t{p['t']}_c{i}_m0.npz")
            eef, fing = [], []
            for f in range(17):
                set_state(env, r["states"][f])
                eef.append(np.array(env.sim.data.site_xpos[site])); fing.append(np.array([env.sim.data.body_xpos[b] for b in fid]))
            paths.append(dict(eef=np.array(eef), fing=np.array(fing), close=r["action.gripper_close"][0, :, 0] > 0.5))
        s = np.load(snap)
        panels = []
        for cam in CAMS:
            img = Image.fromarray(s[f"rgb/{cam}"]).convert("RGB")
            d = ImageDraw.Draw(img)
            K, T = s[f"K/{cam}"], s[f"T/{cam}"]
            for i in np.argsort([-abs(n - 1) for n in p["per"]]):  # grey first, then the decided ones on top
                c = COL[p["per"][i]]
                xy = [None if q is None else (float(q[1]), float(q[0])) for q in project(paths[i]["eef"], K, T)]
                segs(d, xy, c, 3)
                for f in np.nonzero(paths[i]["close"])[0]:
                    if xy[f + 1] is not None:
                        x, y = xy[f + 1]; d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=c)
                g = [None if q is None else (float(q[1]), float(q[0])) for q in project(paths[i]["fing"][-1], K, T)]
                for a, b in [(0, 1), (0, 2), (1, 3)]:
                    segs(d, [g[a], g[b]], (20, 20, 20), 5); segs(d, [g[a], g[b]], c, 3)
            if xy[0] is not None:
                x0, y0 = xy[0]
                d.ellipse([x0 - 5, y0 - 5, x0 + 5, y0 + 5], outline=(255, 255, 255), width=2)
            panels.append(img)
        out = Image.new("RGB", (480 * 3 + 20, 480 + 28), (252, 252, 251))
        for j, im in enumerate(panels):
            out.paste(im, (j * 490, 28))
        ImageDraw.Draw(out).text((6, 6), f"{p['tag']} t{p['t']} stage S{p['k']}  P(done) {p['pd']:.2f}  {p.get('legend', 'candidates finishing in both continuations / one / neither')}: "
                                          f"{p['per'].count(2)} / {p['per'].count(1)} / {p['per'].count(0)}   (left | right | wrist)", fill=(11, 11, 11))
        od = os.path.join(data.OUT, p.get("outdir", "overlay"))
        os.makedirs(od, exist_ok=True)
        out.save(os.path.join(od, f"{p['tag']}_t{p['t']}.png"))
        return dict(tag=p["tag"], t=p["t"], ok=True)
    except Exception as e:
        import traceback
        return dict(tag=p["tag"], t=p["t"], error=repr(e), tb=traceback.format_exc()[-1200:])


def jobs(per_task=6):
    from effect_by_event import points
    pts, out = points(), []
    for task in data.TASKS:
        g = [p for p in pts if p["task"] == task and (p["y"].sum(1) == 2).sum() >= 2 and (p["y"].sum(1) == 0).sum() >= 2]
        for p in sorted(g, key=lambda p: (-p["cb"], p["tag"], p["t"]))[:per_task]:  # var_b is discrete: ties broken by (tag, t)
            out.append((dict(task=task, tag=p["tag"], t=p["t"], k=p["k"], pd=p["pd"], per=[int(x) for x in p["y"].sum(1)],
                             scene=data.scenes()[p["tag"]]), p["roll"], p["snap"]))
    return out


if __name__ == "__main__":
    js = jobs()
    print(f"{len(js)} points", flush=True)
    with multiprocessing.get_context("spawn").Pool(int(sys.argv[1])) as pool:
        for r in pool.imap_unordered(work, js):
            print(r.get("tag"), r.get("t"), r.get("error", "ok"), r.get("tb", "")[-400:], flush=True)
