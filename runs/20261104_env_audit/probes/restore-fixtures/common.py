"""shared helpers for the restore-fixtures audit (read-only on the repo)."""
import os, sys, json, glob, copy
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{ROOT}/runs/scripts")
import robosuite
import robocasa  # noqa
from robosuite.controllers import load_composite_controller_config
from atomic_eval import load_scene, restore_node
from oracle import set_state
import hidden_state
from hidden_state import py_state, set_py_state

RUN = f"{ROOT}/runs/20261025_seen480/run"


def make_env(task):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=cc,
                         camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
                         camera_widths=128, camera_heights=128, has_renderer=False, has_offscreen_renderer=False,
                         ignore_done=True, use_object_obs=True, use_camera_obs=False, camera_depths=False, seed=None,
                         obj_instance_split="target", generative_textures=None, randomize_cameras=False,
                         layout_and_style_ids=list(zip(range(1, 11), range(1, 11))), layout_ids=None, style_ids=None,
                         translucent_robot=False)
    return env


def rows():
    out = []
    for p in glob.glob(f"{RUN}/results_w*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            if "error" not in r:
                out.append(r)
    return out


def scene_of(row):
    from paths import local
    return local(row["scene"])


_SKIP_ENV = {"sim", "model", "_obs_cache"}


def canon(v, depth=0):
    """hashable canonical form of plain data; None marker for non-plain."""
    if depth > 6:
        return ("deep",)
    if v is None or isinstance(v, (bool, int, float, str, np.bool_, np.integer, np.floating)):
        return ("s", type(v).__name__, repr(v))
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return ("objarr", tuple(canon(x, depth + 1) for x in v.ravel()))
        return ("a", v.shape, str(v.dtype), v.tobytes())
    if isinstance(v, (list, tuple)):
        return (type(v).__name__, tuple(canon(x, depth + 1) for x in v))
    if isinstance(v, dict):
        try:
            items = sorted(v.items(), key=lambda kv: repr(kv[0]))
        except Exception:
            items = list(v.items())
        return ("d", tuple((repr(k), canon(x, depth + 1)) for k, x in items))
    return ("obj", type(v).__name__)


def snap(kenv):
    """every attribute of env / fixtures / objects, canonical."""
    S = {}
    for k, v in vars(kenv).items():
        if k in _SKIP_ENV:
            continue
        S[("env", k)] = canon(v)
    for n, fx in kenv.fixtures.items():
        for k, v in vars(fx).items():
            S[("fx:" + n + ":" + type(fx).__name__, k)] = canon(v)
    for n, o in kenv.objects.items():
        for k, v in vars(o).items():
            S[("obj:" + n, k)] = canon(v)
    return S


def diff(a, b):
    keys = set(a) | set(b)
    return sorted(k for k in keys if a.get(k) != b.get(k))
