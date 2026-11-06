"""synthetic observations for the offline tests"""
import numpy as np

from coordinator.schema import CAMERAS, Obs


def obs(chunk, t=None, hand=(0.3, 0.0, 0.6), base=(1.0, 2.0, 0.7), gap=0.04, grip=0.0, img=True):
    t = chunk * 16 if t is None else t
    rng = np.random.default_rng(chunk)
    return Obs(chunk=chunk, t=t,
               images={c: rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for c in CAMERAS} if img else {},
               proprio={"state.end_effector_position_relative": np.array(hand, float),
                        "state.base_position": np.array(base, float),
                        "state.gripper_qpos": np.array([gap / 2, -gap / 2])},
               gripper_cmd=grip)
