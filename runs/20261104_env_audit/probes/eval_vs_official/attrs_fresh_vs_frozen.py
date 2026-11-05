"""After an official reset(seed) vs after our load_scene path on the same scene: compare python-side state that the
success checks can read: task attributes (plain values), fixture pose/size/int-site geometry, fixture py_state,
get_ep_meta(), and _check_success() / every stage check at t=0. CPU only (camera obs off)."""
import os, sys, json, time, copy
import numpy as np
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
from robocasa.models.fixtures import Fixture
from oracle import reset_to, set_state
from atomic_steps import TASKS
import hidden_state

def plain(v):
    return v is None or isinstance(v, (bool, int, float, str, np.bool_, np.integer, np.floating)) or (
        isinstance(v, (list, tuple)) and all(plain(x) for x in v)) or (
        isinstance(v, dict) and all(isinstance(k, (str, int)) and plain(x) for k, x in v.items())) or (
        isinstance(v, np.ndarray) and v.size < 64 and v.dtype != object)

def task_attrs(k):
    skip = {"timestep", "cur_time", "_ep_meta", "rng", "_obs_cache"}
    return {n: copy.deepcopy(v) for n, v in vars(k).items() if n not in skip and plain(v)}

def fx_geom(k):
    out = {}
    for n, fx in k.fixtures.items():
        if not isinstance(fx, Fixture): continue
        d = {}
        try: d["pos"] = np.asarray(fx.pos, float).round(5).tolist()
        except Exception as e: d["pos"] = repr(e)
        try: d["rot"] = round(float(fx.rot), 5)
        except Exception as e: d["rot"] = repr(e)
        try: d["size"] = np.asarray(fx.size, float).round(5).tolist()
        except Exception as e: d["size"] = None
        try: d["int"] = {kk: [np.asarray(x, float).round(4).tolist() for x in vv] for kk, vv in fx.get_int_sites(relative=False).items()}
        except Exception as e: d["int"] = None
        out[n] = d
    return out

def eq(a, b):
    try: return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    except Exception: return str(a) == str(b)

task = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(",")]
base = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped; kenv = base.env
for s in seeds:
    base.reset(seed=s)
    xml, meta = kenv.model.get_xml(), kenv.get_ep_meta(); s0 = kenv.sim.get_state().flatten().copy()
    S = {}
    A = dict(task=task_attrs(kenv), fx=fx_geom(kenv), py=hidden_state.py_state(kenv), meta=kenv.get_ep_meta(),
             succ=bool(kenv._check_success()), stages=[bool(st["check"](kenv, S)) for st in TASKS[task]])
    reset_to(kenv, xml, meta, s0); kenv.unset_ep_meta(); set_state(kenv, s0)
    S = {}
    B = dict(task=task_attrs(kenv), fx=fx_geom(kenv), py=hidden_state.py_state(kenv), meta=kenv.get_ep_meta(),
             succ=bool(kenv._check_success()), stages=[bool(st["check"](kenv, S)) for st in TASKS[task]])
    diffs = {}
    for n in sorted(set(A["task"]) | set(B["task"])):
        if not eq(A["task"].get(n, "<missing>"), B["task"].get(n, "<missing>")):
            diffs[f"task.{n}"] = (str(A["task"].get(n, "<missing>"))[:80], str(B["task"].get(n, "<missing>"))[:80])
    for n in sorted(set(A["fx"]) | set(B["fx"])):
        a, b = A["fx"].get(n), B["fx"].get(n)
        if a is None or b is None:
            diffs[f"fx.{n}"] = ("missing" if a is None else "ok", "missing" if b is None else "ok"); continue
        for kk in a:
            if not eq(a[kk], b[kk]): diffs[f"fx.{n}.{kk}"] = (str(a[kk])[:80], str(b[kk])[:80])
    for n in sorted(set(A["py"]) | set(B["py"])):
        a, b = A["py"].get(n, {}), B["py"].get(n, {})
        for kk in sorted(set(a) | set(b)):
            if not eq(a.get(kk, "<missing>"), b.get(kk, "<missing>")): diffs[f"py.{n}.{kk}"] = (str(a.get(kk, "<missing>"))[:80], str(b.get(kk, "<missing>"))[:80])
    md = sorted(k for k in set(A["meta"]) | set(B["meta"]) if not eq(A["meta"].get(k), B["meta"].get(k)))
    print(json.dumps(dict(task=task, seed=s, n_diffs=len(diffs), diffs=diffs, meta_diff_keys=md,
                          succ=(A["succ"], B["succ"]), stages=(A["stages"], B["stages"]))), flush=True)
