import os, sys, json
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
os.environ.setdefault("MUJOCO_GL", "egl")
import robosuite
import robocasa  # noqa
from robocasa.utils.env_utils import load_composite_controller_config
from oracle import reset_to, set_state
import robocasa.utils.object_utils as OU


def make_env(task):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=cc,
                         has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                         ignore_done=True, use_object_obs=True, translucent_robot=False,
                         obj_instance_split="target", layout_and_style_ids=list(zip(range(1, 11), range(1, 11))))
    return env


def scene_path(task, s):
    if task in ("KettleBoiling", "PreSoakPan", "PrepareCoffee", "StackBowlsCabinet", "SteamInMicrowave"):
        d = "scenes_seen5_v2" if s < 20 else "scenes_seen5_v2_s30"
    else:
        d = "scenes_seen11_v2"
    return f"{R}/runs/{d}/{task}_s{s}.npz"


def load(env, task, s):
    from paths import local
    z = np.load(scene_path(task, s), allow_pickle=True)
    reset_to(env, local(str(z["xml"])), json.loads(str(z["ep_meta"])), z["state"])
    env.unset_ep_meta()
    return z["state"].copy()
