"""no-render gym wrapper and a deterministic fake policy, so atomic_eval.rollout runs unchanged on CPU."""
import hashlib
import numpy as np
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv, PandaOmronKeyConverter


def wrap(env):
    base = RoboCasaGymEnv.__new__(RoboCasaGymEnv)
    base.key_converter = PandaOmronKeyConverter
    _, cams, w, h = base.key_converter.get_camera_config()
    base.env = env; base.env_name = type(env).__name__
    base.camera_names, base.camera_widths, base.camera_heights = cams, w, h
    base.enable_render = False
    base.render_obs_key = f"{cams[0]}_image"; base.render_cache = None
    base._create_obs_and_action_space()
    return base


class FakeClient:
    """actions = seeded noise (seed from req['_seed']) + a small deterministic function of the proprio observation, so
    any difference in what the policy is sent changes the actions (closed loop)."""
    def __init__(self, p_base=0.35, obs_gain=0.05):
        self.p_base, self.g = p_base, obs_gain
        self.reqs = []

    def get_action(self, req):
        self.reqs.append({k: (np.array(v) if not isinstance(v, tuple) else v) for k, v in req.items()})
        rng = np.random.default_rng(req.get("_seed", 0) % (2**63))
        st = np.concatenate([np.asarray(req[k]).ravel() for k in sorted(req) if k.startswith("state.")])
        f = np.tanh(st.sum() * 7.0)  # obs-dependent
        base_mode = float(rng.random() < self.p_base)
        out = {
            "action.end_effector_position": (rng.uniform(-0.7, 0.7, (1, 16, 3)) + self.g * f),
            "action.end_effector_rotation": rng.uniform(-0.5, 0.5, (1, 16, 3)),
            "action.gripper_close": np.full((1, 16, 1), float(rng.random() < 0.5)),
            "action.base_motion": np.concatenate([rng.uniform(-0.8, 0.8, (1, 16, 3)) * base_mode, rng.uniform(-0.6, 0.6, (1, 16, 1))], -1),
            "action.control_mode": np.full((1, 16, 1), base_mode),
        }
        if "_seed" in req:
            out["_seed_echo"] = req["_seed"]
        return out


def seedf(tag, off=0):
    return lambda ci: int.from_bytes(hashlib.sha256(f"{tag}|{ci + off}".encode()).digest()[:8], "little")
