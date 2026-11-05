"""collect.py replay() (set_state per saved step, stage checks every 16 steps) vs the live semantics (env.step runs
update_state AND reward() -> _check_success() every step): task-level latches set only inside _check_success."""
import sys, json, copy
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, "/home/michaellee/mclee/robocasa/runs/20261026_critic_data"); sys.path.insert(0, "/home/michaellee/mclee/robocasa/runs/20261026_imagine")
from harness import *
from atomic_eval import load_scene
from atomic_steps import TASKS
from oracle import set_state
from collect import replay

task, tag, attr, t = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
RUN = f"{ROOT}/runs/20261025_seen480/run"
job = next(j for j in json.load(open(f"{RUN}/jobs.json")) if j["tag"] == tag)
env = make_env(task)
states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
load_scene(env, job["scene"])
S, k = replay(env, states, t, TASKS[task])
a = copy.deepcopy(getattr(env, attr)); sa = bool(env._check_success())
load_scene(env, job["scene"])
for i in range(1, t + 1):
    set_state(env, states[i]); env._check_success(); env._check_success()
b = copy.deepcopy(getattr(env, attr)); sb = bool(env._check_success())
print("RESULT", json.dumps(dict(task=task, tag=tag, t=t, T=len(states) - 1, attr=attr, collect_replay=a, live_semantics=b, k_replay=k), default=str))
