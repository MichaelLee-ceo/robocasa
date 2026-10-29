"""Replay harness for the restore review (2026-10-29). Builds the env exactly as ../../20261026_critic_data/collect.py
does, rebuilds a decision point (load_scene -> replay -> settle -> capture), and answers rollout() with saved chunks
through a fake client whose hook runs right before each chunk is executed (i.e. at every chunk boundary).
Read-only use of runs/scripts, collect.py and the data; nothing outside this folder is modified."""
import copy, glob, hashlib, json, os, sys, time
import numpy as np

R = os.environ.get("ROBOCASA_ROOT", "/home/michaellee/mclee/robocasa")
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261026_imagine", f"{R}/runs/20261026_critic_data"]
import gymnasium as gym  # noqa: E402
import hidden_state  # noqa: E402
from atomic_eval import load_scene, rollout  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
from collect import replay  # noqa: E402
from imagine import py_state, set_py_state  # noqa: E402
from oracle import set_state  # noqa: E402

DATA = f"{R}/runs/20261026_critic_data"
MAIN = f"{R}/runs/20261025_seen480/run"
HERE = os.path.dirname(os.path.abspath(__file__))
OPEN_V3 = abs(0.020833 - (-0.020833)) * 1.9 - 0.004


def make_env(task):
    for attempt in range(2):
        try:
            base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
            return base, base.env
        except Exception as e:  # "could not initialize task" is flaky
            print("env build failed:", repr(e)[:200], flush=True)
            if attempt:
                raise


def find_row(d, tag, t):
    for p in glob.glob(f"{DATA}/{d}/results_w*.jsonl"):
        for l in open(p):
            if f'"tag": "{tag}"' in l and f'"t": {t},' in l and '"error"' not in l:
                return json.loads(l)
    raise KeyError((d, tag, t))


def find_job(tag, t):
    for p in sorted(glob.glob(f"{DATA}/jobs*.json")):
        for j in json.load(open(p)):
            if j["tag"] == tag and j["t"] == t:
                return j
    raise KeyError((tag, t))


def roll_path(d, tag, t, c, m):
    return f"{DATA}/{d}/roll/{tag}_t{t}_c{c}_m{m}.npz"


def settle_v3_gripper(kenv, prev_state):
    """the gripper part of settle as it was for rows 'settled' .. 'settled-v3' (old open threshold)."""
    for r in kenv.robots:
        for arm, g in r.gripper.items():
            q = kenv.sim.data.qpos[r._ref_gripper_joint_pos_indexes[arm]]
            gap = abs(q[0] - q[1])
            closing = gap < OPEN_V3
            if prev_state is not None and closing:
                qp = prev_state[1 + np.asarray(r._ref_gripper_joint_pos_indexes[arm])]
                closing = not (gap - abs(qp[0] - qp[1]) > 0.0005)
            g.current_action = np.array([-1.0, 1.0]) * (1.0 if closing else -1.0)


def settle_v4_gripper(kenv, prev_state):
    """the gripper part of settle for rows 'settled-v4' (open threshold from the joint limits, +-1 only)."""
    for r in kenv.robots:
        for arm, g in r.gripper.items():
            if not hasattr(g, "current_action"):
                continue
            q = kenv.sim.data.qpos[r._ref_gripper_joint_pos_indexes[arm]]
            gap = abs(q[0] - q[1])
            lo, hi = kenv.sim.model.jnt_range[[kenv.sim.model.joint_name2id(j) for j in g.joints]].T
            closing = gap < max(abs(hi[0] - lo[1]), abs(lo[0] - hi[1])) - 0.0015
            if prev_state is not None and closing:
                qp = prev_state[1 + np.asarray(r._ref_gripper_joint_pos_indexes[arm])]
                closing = not (gap - abs(qp[0] - qp[1]) > 0.0005)
            g.current_action = np.array([-1.0, 1.0]) * (1.0 if closing else -1.0)


VERSION = {"settled": "v1", "settled-basefix": "v2", "settled-v3": "v3", "settled-v4": "v4", "settled-v5": "v5"}


def settle(kenv, prev_state, version="v4"):
    """hidden_state.settle as it was for each row version:
    v1 'settled' (git HEAD): every part's reset_goal (base start pose overwritten, OSC goal in world coordinates), old
       gripper threshold;  v2 'settled-basefix': OSC goal in the base frame;  v3 'settled-v3': base start pose kept;
    v4 'settled-v4': open threshold from the joint limits;  v5 'settled-v5': hidden_state.settle as it is now (gripper
    ramp from the servo target and contacts), and no settle at all at t = 0 (Point skips it).
    (2026-10-29: v3 and v4 now emulate their gripper rule on top of the current file, whose controller part is unchanged.)"""
    if version in ("v3", "v4", "v5"):
        hidden_state.settle(kenv, prev_state)
    else:
        kenv.sim.data.qacc_warmstart[:] = 0
        for r in kenv.robots:
            cc = r.composite_controller
            cc.update_state()
            for c in cc.part_controllers.values():
                c.new_update = True
                c.update(force=True)
                c.reset_goal()
                if version == "v2" and getattr(c, "input_ref_frame", None) == "base" and hasattr(c, "world_to_origin_frame"):
                    c.goal_pos = c.world_to_origin_frame(np.array(c.ref_pos))
                    c.goal_ori = c.goal_origin_to_eef_pose()[:3, :3]
    if version == "v4":
        settle_v4_gripper(kenv, prev_state)
    elif version != "v5":
        settle_v3_gripper(kenv, prev_state)


class Saved:
    """client answering with saved chunks, from chunk `start`; hook(i, req) runs before chunk i is returned."""
    def __init__(self, npz, start=0, hook=None):
        keys = [k for k in npz.files if k.startswith("action.")]
        self.chunks = [{k: npz[k][c] for k in keys} for c in range(len(npz[keys[0]]))]
        self.i, self.start, self.hook, self.reqs = 0, start, hook, []

    def get_action(self, req):
        if self.hook is not None:
            self.hook(self.i, req)
        self.reqs.append({k: v for k, v in req.items() if k.startswith("state.")})
        ch = self.chunks[min(self.start + self.i, len(self.chunks) - 1)]
        self.i += 1
        return {k: np.asarray(v)[None] for k, v in ch.items()}


class Point:
    """a decision point rebuilt as collect.decision_point does (up to H0)."""
    def __init__(self, base, kenv, task, tag, t, k, scene, version="v4", row=None):
        self.base, self.kenv, self.task, self.tag, self.t, self.k = base, kenv, task, tag, t, k
        self.version = version
        self.steps = TASKS[task]
        load_scene(kenv, scene)
        self.after_load = hidden_state.capture(kenv)  # exact hidden state of the main run at t=0
        self.instr = kenv.get_ep_meta()["lang"]
        self.ep = np.load(f"{MAIN}/{task}/{tag}_steps.npz")["states"]
        self.S0, self.k_replay = replay(kenv, self.ep, t, self.steps)
        if t or version != "v5":  # v5: at t = 0 the state after load_scene is the main run's exactly
            settle(kenv, self.ep[t - 1] if t else None, version)
        self.H0 = hidden_state.capture(kenv)
        self.ps = row["py_state"] if row is not None else py_state(kenv)

    def run(self, npz, start_state, horizon, hidden, chunk0=0, hook=None, S=None):
        """rollout() with the saved chunks from chunk0; returns the states (start + every step) and the client."""
        states = []
        cl = Saved(npz, chunk0, hook)
        set_py_state(self.kenv, self.ps)
        rollout(cl, self.base, self.kenv, copy.deepcopy(self.S0 if S is None else S), self.steps, start_state, horizon,
                "task", self.instr, k0=self.k, k_stop=self.k + 1, step_states=states, hidden=hidden)
        return np.array(states), cl


# ---------------------------------------------------------------- state groups
def groups(kenv):
    """index arrays (into the flattened state) of the state groups."""
    m = kenv.sim.model
    nq = m.nq
    g = {"base_q": [], "base_v": [], "torso_q": [], "torso_v": [], "arm_q": [], "arm_v": [], "finger_q": [],
         "finger_v": [], "obj_q": [], "obj_v": [], "fixture_q": [], "fixture_v": []}
    objs = {}
    for j in range(m.njnt):
        n = m.joint_id2name(j)
        qa, va, ty = m.jnt_qposadr[j], m.jnt_dofadr[j], m.jnt_type[j]
        nqj, nvj = {0: 7, 1: 4, 2: 1, 3: 1}[ty], {0: 6, 1: 3, 2: 1, 3: 1}[ty]
        qi = list(range(1 + qa, 1 + qa + nqj)); vi = list(range(1 + nq + va, 1 + nq + va + nvj))
        if n.startswith("mobilebase0_joint_mobile"):
            key = "base"
        elif n.startswith("mobilebase0_joint_torso"):
            key = "torso"
        elif n.startswith("robot0_"):
            key = "arm"
        elif n.startswith("gripper0_"):
            key = "finger"
        elif ty == 0:
            key = "obj"; objs[n] = (qi, vi)
        else:
            key = "fixture"
        g[key + "_q"] += qi; g[key + "_v"] += vi
    return {k: np.array(v, dtype=int) for k, v in g.items()}, objs


def group_diff(a, b, G):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return {k: (float(d[v].max()) if len(v) else 0.0) for k, v in G.items()}


# ---------------------------------------------------------------- attribute snapshots
PLAIN = (np.ndarray, np.generic, int, float, bool, str, type(None))


def _describe(v, depth=0):
    if isinstance(v, PLAIN):
        return copy.deepcopy(v)
    if isinstance(v, (list, tuple)) and depth < 2:
        return [_describe(x, depth + 1) for x in v]
    if isinstance(v, dict) and depth < 2:
        return {str(k): _describe(x, depth + 1) for k, x in v.items()}
    if type(v).__name__ in ("DeltaBuffer", "RingBuffer") and depth < 2:
        return {k: _describe(x, depth + 1) for k, x in vars(v).items()}
    return f"<{type(v).__name__}>"


def snap_attrs(kenv):
    out = {}
    d = kenv.sim.data
    for f in ("time", "qacc_warmstart", "ctrl", "act", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat",
              "userdata", "plugin_state", "eq_active"):
        if hasattr(d, f):
            out[f"mjData.{f}"] = np.array(getattr(d, f))
    m = kenv.sim.model
    h = hashlib.sha1()
    for f in ("body_pos", "body_quat", "body_mass", "geom_friction", "geom_size", "geom_pos", "jnt_range", "dof_damping",
              "actuator_gainprm", "actuator_biasprm", "opt.timestep", "eq_data", "site_pos"):
        try:
            v = m
            for p in f.split("."):
                v = getattr(v, p)
            h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
        except AttributeError:
            pass
    out["model.physics_sha1"] = h.hexdigest()[:12]
    for k, v in vars(kenv).items():
        if k in ("sim", "model", "_observables"):
            continue
        out[f"env.{k}"] = _describe(v)
    for n, ob in kenv._observables.items():
        for k, v in vars(ob).items():
            if k.startswith("_") and not callable(v):
                out[f"observable.{n}.{k}"] = _describe(v)
    for n, fx in kenv.fixtures.items():
        for k, v in vars(fx).items():
            if isinstance(v, PLAIN) and not isinstance(v, np.ndarray) or (isinstance(v, np.ndarray) and v.size < 50):
                out[f"fixture.{n}.{k}"] = _describe(v)
    for ri, r in enumerate(kenv.robots):
        for k, v in vars(r).items():
            if k in ("sim", "robot_model", "composite_controller", "gripper"):
                continue
            out[f"robot{ri}.{k}"] = _describe(v)
        cc = r.composite_controller
        for k, v in vars(cc).items():
            if k in ("sim", "robot_model", "part_controllers", "grippers"):
                continue
            out[f"robot{ri}.composite.{k}"] = _describe(v)
        for pn, c in cc.part_controllers.items():
            for k, v in vars(c).items():
                if k == "sim":
                    continue
                out[f"robot{ri}.part.{pn}.{k}"] = _describe(v)
        for gn, g in r.gripper.items():
            for k, v in vars(g).items():
                if isinstance(v, PLAIN):
                    out[f"robot{ri}.gripper.{gn}.{k}"] = _describe(v)
    return out


def _flat(prefix, v, out):
    if isinstance(v, dict):
        for k, x in v.items():
            _flat(f"{prefix}.{k}", x, out)
    elif isinstance(v, list) and v and not all(isinstance(x, (int, float, bool, np.generic)) for x in v):
        for i, x in enumerate(v):
            _flat(f"{prefix}[{i}]", x, out)
    else:
        out[prefix] = v


def diff_attrs(a, b, tol=0.0):
    fa, fb = {}, {}
    for k, v in a.items():
        _flat(k, v, fa)
    for k, v in b.items():
        _flat(k, v, fb)
    rows = []
    for k in sorted(set(fa) | set(fb)):
        if k not in fa or k not in fb:
            rows.append((k, "missing in " + ("continuous" if k not in fa else "restored"), None))
            continue
        x, y = fa[k], fb[k]
        try:
            xa, ya = np.asarray(x), np.asarray(y)
            if xa.dtype.kind in "fiub" and ya.dtype.kind in "fiub":
                if xa.shape != ya.shape:
                    rows.append((k, f"shape {xa.shape} vs {ya.shape}", None)); continue
                dd = float(np.max(np.abs(xa.astype(float) - ya.astype(float)))) if xa.size else 0.0
                if dd > tol:
                    rows.append((k, dd, (np.round(xa, 5).tolist() if xa.size <= 9 else f"max {dd:.3g}",
                                         np.round(ya, 5).tolist() if ya.size <= 9 else "")))
                continue
        except Exception:
            pass
        if str(x) != str(y):
            rows.append((k, "differs", (str(x)[:80], str(y)[:80])))
    return rows


def contacts_with_fingers(kenv):
    """names of the root bodies the gripper fingers touch."""
    m, d = kenv.sim.model, kenv.sim.data
    out = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        n1, n2 = m.body_id2name(b1) or "", m.body_id2name(b2) or ""
        for x, y in ((n1, b2), (n2, b1)):
            if x.startswith("gripper0_") and "finger" in x:
                yn = m.body_id2name(m.body_rootid[y]) or ""
                if not yn.startswith(("robot0_", "gripper0_", "mobilebase0_")):
                    out.add(yn)
    return sorted(out)
