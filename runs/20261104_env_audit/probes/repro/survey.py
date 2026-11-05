"""CPU-only (Mesa llvmpipe): per task, semi-transparent visible geoms and how many pixels they cover in each policy
camera at the scene's initial state (segmentation render). usage: survey.py <task> <n_scenes>"""
import collections, json, os, sys, time
import numpy as np, gymnasium as gym, mujoco
R = "/home/michaellee/mclee/robocasa/runs"
sys.path.insert(0, f"{R}/scripts")
import atomic_eval as A
task, n = sys.argv[1], int(sys.argv[2])
def scene_path(t, s):
    for d in ("scenes_seen5_v2", "scenes_seen5_v2_s30", "scenes_seen11_v2"):
        p = f"{R}/{d}/{t}_s{s}.npz"
        if os.path.exists(p) and (d != "scenes_seen5_v2" or s < 20):
            return p
base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
kenv = base.env
out = dict(task=task, scenes=[])
for s in range(n):
    p = scene_path(task, s)
    if p is None: continue
    A.load_scene(kenv, p)
    m = kenv.sim.model._model
    vis = kenv.sim._render_context_offscreen.vopt.geomgroup
    semi = {}
    for i in range(m.ngeom):
        if not vis[m.geom_group[i]]: continue
        mi = m.geom_matid[i]
        a = m.mat_rgba[mi][3] if mi >= 0 else m.geom_rgba[i][3]
        if 0 < a < 1:
            mat = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mi) if mi >= 0 else "(geom rgba)"
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_rootid[m.geom_bodyid[i]])
            semi[i] = f"{body}|{mat}|a={a:.2f}"
    cams = {}
    for cam in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
        seg = kenv.sim.render(camera_name=cam, width=256, height=256, segmentation=True)
        typ, gid = seg[..., 0], seg[..., 1]
        isgeom = typ == int(mujoco.mjtObj.mjOBJ_GEOM)
        cnt = collections.Counter()
        for g in np.unique(gid[isgeom]):
            if int(g) in semi:
                cnt[semi[int(g)]] += int(((gid == g) & isgeom).sum())
        cams[cam] = dict(semi_pixels=int(sum(cnt.values())), top=cnt.most_common(4))
    out["scenes"].append(dict(s=s, n_semi_geoms=len(semi), offsamples=int(m.vis.quality.offsamples),
                              kinds=sorted({v.split("|", 1)[1].split("_group_")[-1] for v in semi.values()})[:12], cams=cams))
print(json.dumps(out), flush=True)
