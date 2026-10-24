"""Which visible geoms are semi-transparent (0 < alpha < 1) in the frozen Composite-Seen scenes?
Loads scenes s0..s(n-1) of every task and counts visible (robosuite render geom groups) semi-transparent geoms by
material name. usage: python -I transparent_survey.py [n_scenes_per_task=5]"""
import collections, json, os, sys
import gymnasium as gym
import mujoco
R = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{R}/runs/scripts")
import atomic_eval as A  # noqa: E402
from atomic_steps import TASKS  # noqa: E402

SEEN = ["DeliverStraw", "GetToastedBread", "KettleBoiling", "LoadDishwasher", "PackIdenticalLunches", "PreSoakPan",
        "PrepareCoffee", "RinseSinkBasin", "ScrubCuttingBoard", "SearingMeat", "SetUpCuttingStation", "StackBowlsCabinet",
        "SteamInMicrowave", "StirVegetables", "StoreLeftoversInBowl", "WashLettuce"]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def scene_path(t, s):
    for d in ("scenes_seen5_v2", "scenes_seen5_v2_s30", "scenes_seen11_v2"):
        p = f"{R}/runs/{d}/{t}_s{s}.npz"
        if os.path.exists(p) and (d != "scenes_seen5_v2" or s < 20):
            return p


by_mat, scenes_with, per_task = collections.Counter(), collections.Counter(), {}
for t in SEEN:
    assert t in TASKS
    base = gym.make(f"robocasa/{t}", split="target", enable_render=True).unwrapped
    kenv = base.env
    for s in range(n):
        A.load_scene(kenv, scene_path(t, s))
        m, vis = kenv.sim.model._model, kenv.sim._render_context_offscreen.vopt.geomgroup
        found = set()
        for i in range(m.ngeom):
            if not vis[m.geom_group[i]]:
                continue
            mi = m.geom_matid[i]
            a = m.mat_rgba[mi][3] if mi >= 0 else m.geom_rgba[i][3]
            if 0 < a < 1:
                key = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mi) if mi >= 0 else f"(geom rgba) {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)}"
                key = key.split("_group_")[-1] if "_group_" in key else key  # drop the fixture-instance prefix
                found.add(f"{key} a={a:.2f}")
        for k in found:
            by_mat[k] += 1
        per_task.setdefault(t, []).append(len(found))
    base.close()
    print(t, per_task[t], flush=True)
print(json.dumps(dict(scenes=n * len(SEEN), by_material=by_mat.most_common()), indent=1))
