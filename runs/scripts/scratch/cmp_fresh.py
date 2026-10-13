import sys
sys.path.insert(0, "/home/michaellee/Desktop/robocasa/runs/scripts"); sys.path.insert(0, "/home/michaellee/Desktop/robocasa/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval, imageio.v3 as iio
from atomic_eval import load_scene
def mk(): b = gym.make("robocasa/StackBowlsCabinet", split="target", enable_render=True).unwrapped; return b, b.env
def grab(b, k):
    o = b.get_observation(k._get_observations(force_update=True)); return {c: o[c].astype(float) for c in ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")}
def cams(k):
    m = k.sim.model; out = {}
    for n in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
        i = m.camera_name2id(n); out[n] = (np.round(m.cam_pos[i], 4).tolist(), np.round(m.cam_quat[i], 4).tolist(), float(m.cam_fovy[i]))
    return out
b1, k1 = mk(); b1.reset(seed=0); A = grab(b1, k1); cA = cams(k1)
b2, k2 = mk(); load_scene(k2, "/home/michaellee/Desktop/robocasa/runs/scenes_seen5/StackBowlsCabinet_s0.npz"); B = grab(b2, k2); cB = cams(k2)
for c in A: print(c, "mean", round(float(np.abs(A[c] - B[c]).mean()), 2))
for n in cA: print(n, cA[n], "|", cB[n])
print("state equal", np.abs(k1.sim.get_state().flatten() - k2.sim.get_state().flatten()).max())
for attr in ("randomize_cameras", "generative_textures", "use_cotraining_cameras"):
    print(attr, getattr(k1, attr, "n/a"), getattr(k2, attr, "n/a"))
print("ngeom", k1.sim.model.ngeom, k2.sim.model.ngeom, "nmat", k1.sim.model.nmat, k2.sim.model.nmat, "nlight", k1.sim.model.nlight, k2.sim.model.nlight)
iio.imwrite("/tmp/claude-1001/-home-michaellee-Desktop-robocasa/d2b6537f-619c-5af7-ab37-a54e14a4b87a/scratchpad/fresh_cmp.png", np.concatenate([np.concatenate([A[c], B[c]], 1) for c in A], 0).astype(np.uint8))
