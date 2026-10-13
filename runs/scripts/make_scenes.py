"""Freeze one fresh scene per (task, seed): reset(seed) once, save model xml + ep_meta + initial state.
reset(seed) alone is not reproducible across a worker's history (the outcome depends on earlier resets), so
every condition loads these files with reset_to() instead. Each scene is checked by reloading it and comparing
the rendered observation with the original one."""
import os, sys, json, hashlib, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/Desktop/robocasa/third_party/Isaac-GR00T"))
import gymnasium as gym
import robocasa  # noqa
import groot_eval  # noqa  (registers the robocasa/* gym envs)
from oracle import reset_to, set_state

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="KettleBoiling,PreSoakPan,StackBowlsCabinet,SteamInMicrowave,PrepareCoffee")
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--out", default=os.path.expanduser("~/Desktop/robocasa/runs/scenes_seen5_v2"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def fingerprint(kenv):
    m = kenv.get_ep_meta()
    return dict(layout=m.get("layout_id"), style=m.get("style_id"), nq=int(kenv.sim.model.nq),
                objects={c["name"]: c.get("info", {}).get("mjcf_path", "").split("/objects/")[-1] for c in m["object_cfgs"]},
                phys_sha1=phys_hash(kenv))


def phys_hash(kenv):
    m = kenv.sim.model
    return hashlib.sha1(b"".join(np.round(np.asarray(a, dtype=float), 6).tobytes() for a in
                                 (m.body_mass, m.body_inertia, m.body_ipos, m.geom_friction))).hexdigest()[:12]


def view(base, kenv):
    o = base.get_observation(kenv._get_observations(force_update=True))
    return np.concatenate([o["video.robot0_agentview_left"], o["video.robot0_eye_in_hand"]], 1).astype(np.float32)


for task in a.tasks.split(","):
    base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped; kenv = base.env
    for s in range(a.seeds):
        f = f"{a.out}/{task}_s{s}.npz"
        if os.path.exists(f):
            continue
        base.reset(seed=s)
        # robosuite's pre-compile MJCF (what the datasets store). MuJoCo's saved xml (kenv.sim.model.get_xml()) recompiles
        # with different body masses/inertias (bowls ~10x heavier), which changed the policy's outcomes.
        xml, meta = kenv.model.get_xml(), kenv.get_ep_meta()
        state = kenv.sim.get_state().flatten().copy(); fp = fingerprint(kenv); img0 = view(base, kenv)
        # reload check: rebuild from the saved xml/ep_meta and compare
        reset_to(kenv, xml, meta, state)
        fp2 = fingerprint(kenv); d = float(np.abs(view(base, kenv) - img0).mean())
        kenv.unset_ep_meta()  # reset_to() pins ep_meta; without this the next reset(seed) rebuilds this same scene
        ok = fp2 == fp and d < 1.0
        np.savez_compressed(f, xml=xml, ep_meta=json.dumps(meta), state=state, fp=json.dumps(fp),
                            lang=meta["lang"], reload_mad=d, ok=ok)
        print(json.dumps(dict(task=task, seed=s, ok=ok, reload_mad=round(d, 3), **{k: fp[k] for k in ("layout", "style", "nq")},
                              objects=fp["objects"])), flush=True)
    base.close()
