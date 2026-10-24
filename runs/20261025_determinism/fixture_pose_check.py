"""After load_scene (= oracle.reset_to: env.reset() -> reset_from_xml_string -> set_state), do the python fixtures
still describe the discarded random scene? (robocasa issue #209 / PR #225). For every fixture, compares the python
pose (fixture.pos / yaw from the MJCF element) with the simulator's body pose, and marks the task's own fixture refs.
usage: python -I fixture_pose_check.py [n_scenes_per_task=3]"""
import collections, json, os, sys
import numpy as np
import gymnasium as gym
R = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{R}/runs/scripts")
import atomic_eval as A  # noqa: E402
from robocasa.models.fixtures import Fixture  # noqa: E402

SEEN = ["DeliverStraw", "GetToastedBread", "KettleBoiling", "LoadDishwasher", "PackIdenticalLunches", "PreSoakPan",
        "PrepareCoffee", "RinseSinkBasin", "ScrubCuttingBoard", "SearingMeat", "SetUpCuttingStation", "StackBowlsCabinet",
        "SteamInMicrowave", "StirVegetables", "StoreLeftoversInBowl", "WashLettuce"]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
if len(sys.argv) > 2:  # optional comma-separated task subset
    SEEN = sys.argv[2].split(",")


def scene_path(t, s):
    for d in ("scenes_seen5_v2", "scenes_seen5_v2_s30", "scenes_seen11_v2"):
        p = f"{R}/runs/{d}/{t}_s{s}.npz"
        if os.path.exists(p) and (d != "scenes_seen5_v2" or s < 20):
            return p


def yaw_of(mat):
    return np.arctan2(mat[1, 0], mat[0, 0])


summary = {}
for t in SEEN:
    base = gym.make(f"robocasa/{t}", split="target", enable_render=True).unwrapped
    kenv = base.env
    rows = []
    for s in range(n):
        A.load_scene(kenv, scene_path(t, s))
        refs = {name: fx for name, fx in vars(kenv).items() if isinstance(fx, Fixture)}  # e.g. kenv.stove, kenv.sink
        ref_ids = {id(fx): name for name, fx in refs.items()}
        for fname, fx in kenv.fixtures.items():
            if not isinstance(fx, Fixture):
                continue
            try:
                bid = kenv.sim.model.body_name2id(fx.root_body)
            except Exception:
                continue
            sim_pos = np.array(kenv.sim.data.body_xpos[bid])
            sim_yaw = yaw_of(np.array(kenv.sim.data.body_xmat[bid]).reshape(3, 3))
            py_pos = np.array(fx.pos, dtype=float)
            py_yaw = float(np.asarray(fx.rot)) if np.ndim(fx.rot) == 0 else float(np.asarray(fx.rot)[-1])
            dpos = float(np.linalg.norm(sim_pos - py_pos))
            dyaw = float(abs((sim_yaw - py_yaw + np.pi) % (2 * np.pi) - np.pi))
            if dpos > 1e-3 or dyaw > 1e-2:
                rows.append(dict(scene=s, fixture=fname, type=type(fx).__name__, task_ref=ref_ids.get(id(fx)),
                                 dpos_m=round(dpos, 3), dyaw_rad=round(dyaw, 3)))
    base.close()
    by_fx = collections.Counter((r["type"], r["task_ref"]) for r in rows)
    summary[t] = dict(mismatches=len(rows), by_type_and_ref=[f"{k[0]} ref={k[1]} x{v}" for k, v in by_fx.most_common()],
                      worst=sorted(rows, key=lambda r: -r["dpos_m"])[:3])
    print(t, json.dumps(summary[t]), flush=True)
out = os.environ.get("CHECK_OUT", "fixture_pose_check.json")
json.dump(summary, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), out), "w"), indent=1)
