"""Official path (gym reset(seed) then step) vs our frozen-scene path (reset_to + set_state + forced obs, as
load_scene + rollout(hidden=None)) on the same scene and the same action sequence. CPU only: camera obs off."""
import os, sys, json, time
import numpy as np
os.environ["MUJOCO_GL"] = "disable" if False else os.environ.get("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/michaellee/mclee/robocasa/runs/scripts")
sys.path.insert(0, "/home/michaellee/mclee/robocasa")
import robosuite
_orig = robosuite.make
def _mk(*a, **kw):
    kw.update(use_camera_obs=False, has_offscreen_renderer=False)
    return _orig(*a, **kw)
robosuite.make = _mk
import gymnasium as gym
import robocasa  # noqa
import robocasa.wrappers.gym_wrapper  # noqa
from oracle import reset_to, set_state
import hidden_state

def acts(n=64, seed=0):
    r = np.random.default_rng(seed); out = []
    for i in range(n):
        base_mode = 1.0 if ((i // 16) + OFF) % 2 == 0 else 0.0
        out.append({"action.end_effector_position": r.uniform(-0.3, 0.3, 3).astype(np.float32),
                    "action.end_effector_rotation": r.uniform(-0.3, 0.3, 3).astype(np.float32),
                    "action.gripper_close": np.array([1.0 if i >= 24 else 0.0], np.float32),
                    "action.base_motion": np.concatenate([r.uniform(-0.5, 0.5, 3), [0.0]]).astype(np.float32) if base_mode else np.zeros(4, np.float32),
                    "action.control_mode": np.array([base_mode], np.float32)})
    return out

def run(base, kenv, A):
    traj, succ = [], []
    for a in A:
        o, r, d, tr, info = base.step({k: v.copy() for k, v in a.items()})
        traj.append(kenv.sim.get_state().flatten().copy()); succ.append(r)
    return np.array(traj), o

def flat_hidden(h):
    out = {}
    for i, r in enumerate(h["robots"]):
        for part, d in r["parts"].items():
            for k, v in d.items():
                if isinstance(v, np.ndarray) or isinstance(v, (float, int, str, bool)):
                    out[f"{part}.{k}"] = v
        for g, d in r["grippers"].items():
            for k, v in d.items(): out[f"grip.{g}.{k}"] = v
    out["timestep"] = h["timestep"]; out["cur_time"] = h["cur_time"]
    for n, c in h["obs_clock"].items(): out[f"clock.{n}"] = c[0]
    return out

task = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(",")]; OFF = int(sys.argv[3]); VARIANT = sys.argv[4]
t0 = time.time()
base = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped; kenv = base.env
res = []
for s in seeds:
    A = acts(64, s)
    oA, _ = base.reset(seed=s)          # official: RoboCasaGymEnv.reset -> env.reset() (forced obs inside)
    xml, meta = kenv.model.get_xml(), kenv.get_ep_meta(); s0 = kenv.sim.get_state().flatten().copy()
    HA = hidden_state.capture(kenv, obs=True); hA = flat_hidden(HA)
    trA, oAe = run(base, kenv, A)
    # ours: load_scene (reset_to + unset_ep_meta) then rollout(hidden=None): restore_node -> set_state, forced obs
    reset_to(kenv, xml, meta, s0); kenv.unset_ep_meta()
    set_state(kenv, s0)
    if VARIANT == "capture":
        from atomic_eval import restore_node
        restore_node(kenv, s0, HA)
    if VARIANT in ("ctrl", "ctrl_clock"):
        for rb in kenv.robots:
            rb.composite_controller.update_state(); rb.composite_controller.reset()
            for g in rb.gripper.values(): g.current_action = np.zeros(g.dof)
    if VARIANT in ("clock", "ctrl_clock"):
        for ob in kenv._observables.values(): ob.reset()
    oB = base.get_observation(kenv._get_observations(force_update=(VARIANT != "capture")))
    hB = flat_hidden(hidden_state.capture(kenv))
    assert np.array_equal(kenv.sim.get_state().flatten(), s0)
    trB, oBe = run(base, kenv, A)
    hd = {}
    for k in sorted(set(hA) | set(hB)):
        a, b = hA.get(k), hB.get(k)
        try:
            eq = (a is None and b is None) or (np.array_equal(np.asarray(a), np.asarray(b)))
        except Exception:
            eq = str(a) == str(b)
        if not eq:
            try: dv = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
            except Exception: dv = f"{a!r} vs {b!r}"[:120]
            hd[k] = dv
    obs0 = {k: float(np.max(np.abs(np.asarray(oA[k], float) - np.asarray(oB[k], float)))) for k in oA if k.startswith("state.")}
    dq = np.abs(trA - trB).max(axis=1)
    first = int(np.argmax(dq > 0)) if np.any(dq > 0) else None
    nq = kenv.sim.model.nq
    res.append(dict(task=task, seed=s, hidden_diff=hd, obs0_diff=obs0, first_div_step=first,
                    maxdiff_at=[float(dq[i]) for i in (0, 1, 15, 16, 31, 32, 47, 63)],
                    robot_qpos_diff_end=float(np.abs(trA[-1, 1:1 + nq] - trB[-1, 1:1 + nq]).max())))
    print(json.dumps(res[-1]), flush=True)
print("wall", round(time.time() - t0, 1))
