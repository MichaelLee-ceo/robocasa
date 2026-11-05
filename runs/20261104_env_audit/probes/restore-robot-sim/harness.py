"""Audit harness (restore-robot-sim): build a RoboCasa env exactly like robocasa.utils.env_utils.create_env(split="target")
but with rendering off (no offscreen renderer, no camera observables), so it runs on CPU."""
import os, sys, json, copy
import numpy as np

ROOT = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{ROOT}/runs/scripts")
sys.path.insert(0, f"{ROOT}/third_party/Isaac-GR00T")
import robosuite
import robocasa  # noqa
from robosuite.controllers import load_composite_controller_config


def make_env(task):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(
        env_name=task, robots="PandaOmron", controller_configs=cc,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=256, camera_heights=256, has_renderer=False, has_offscreen_renderer=False,
        ignore_done=True, use_object_obs=True, use_camera_obs=False, camera_depths=False, seed=None,
        obj_instance_split="target", generative_textures=None, randomize_cameras=False,
        layout_and_style_ids=list(zip(range(1, 11), range(1, 11))), layout_ids=None, style_ids=None,
        translucent_robot=False)
    env.reset()
    return env


def scene_path(task, s=0):
    for d in ("scenes_seen5_v2", "scenes_seen11_v2", "scenes_seen5_v2_s30"):
        p = f"{ROOT}/runs/{d}/{task}_s{s}.npz"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(task)


def action_vec(env, arm6, grip, base3, torso, mode):
    """env action like RoboCasaGymEnv.step builds it (PandaOmronKeyConverter.unmap_action)."""
    r = env.robots[0]; cc = r.composite_controller
    a = np.zeros(cc.action_limits[0].shape)
    parts = {"right": arm6, "right_gripper": [grip], "base": base3, "torso": [torso]}
    for p, (s, e) in cc._action_split_indexes.items():
        a[s:e] = parts[p]
    a[-1] = mode
    return a


def rand_actions(n, seed, p_base=0.3):
    rng = np.random.default_rng(seed)
    out = []
    mode = -1.0; grip = -1.0
    for i in range(n):
        if i % 8 == 0:
            mode = 1.0 if rng.random() < p_base else -1.0
            grip = 1.0 if rng.random() < 0.5 else -1.0
        arm = rng.uniform(-0.6, 0.6, 6)
        base = rng.uniform(-0.6, 0.6, 3) if mode > 0 else np.zeros(3)
        torso = rng.uniform(-0.5, 0.5)
        out.append((arm, grip, base, torso, mode))
    return out


def plain_vars(obj, skip=()):
    from hidden_state import _plain
    out = {}
    for k, v in vars(obj).items():
        if k in skip:
            continue
        if isinstance(v, np.ndarray) and v.size < 10000:
            out[k] = v.copy()
        elif _plain(v):
            out[k] = copy.deepcopy(v)
    return out


def diff_vars(a, b):
    ks = set(a) | set(b)
    d = []
    for k in sorted(ks):
        if k not in a or k not in b:
            d.append(k); continue
        x, y = a[k], b[k]
        try:
            if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                if np.shape(x) != np.shape(y) or not np.array_equal(np.asarray(x), np.asarray(y)):
                    d.append(k)
            elif x != y:
                d.append(k)
        except Exception:
            if repr(x) != repr(y):
                d.append(k)
    return d
