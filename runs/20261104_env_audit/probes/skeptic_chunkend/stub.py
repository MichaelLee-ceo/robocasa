import sys
sys.path.insert(0, "/home/michaellee/mclee/robocasa/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym
from gymnasium import spaces
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
PATTERN = set(range(5, 12))   # success only at steps 5..11 of chunk 0 (mid-chunk), never at a chunk end
class Stub(gym.Env):
    observation_space = spaces.Dict({"state.x": spaces.Box(-1, 1, (1,)), "video.v": spaces.Box(0, 255, (2, 2, 3), np.uint8)})
    action_space = spaces.Dict({"action.a": spaces.Box(-1, 1, (1,))})
    def reset(self, seed=None, options=None):
        self.t = 0; return self.observation_space.sample(), {"success": False}
    def step(self, a):
        self.t += 1; s = self.t in PATTERN
        return self.observation_space.sample(), float(s), self.t >= 64, False, {"success": s}
def make():
    return MultiStepWrapper(Stub(), np.array([0]), np.array([0]), n_action_steps=16, max_episode_steps=64)
venv = gym.vector.SyncVectorEnv([make])
venv.reset()
official = False; ours = False
for c in range(4):
    act = {"action.a": np.zeros((1, 16, 1), np.float32)}
    _, r, term, trunc, info = venv.step(act)
    official |= bool(info["success"][0][0])
    # our rule needs the single-env info['rewards']; reproduce on a single wrapper
    print("chunk", c, "info['success'] shape", np.asarray(info["success"]).shape, "value", info["success"][0][0], "term", term)
m = make(); m.reset()
for c in range(4):
    _, _, d, _, inf = m.step({"action.a": np.zeros((16, 1), np.float32)})
    ours |= bool(np.any(inf["rewards"] > 0))
print("official run_simulation rule:", official, "| our any-step rule:", ours)
