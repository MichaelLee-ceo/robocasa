import sys, json
sys.path.insert(0, sys.argv[1])
from harness import *
from atomic_eval import load_scene
import hidden_state
exec(open(sys.argv[1] + "/test_t0.py").read().split("def stale")[0].split("import hidden_state")[1])  # flat, cmp
task = sys.argv[2]
env = make_env(task)
s0, s1 = scene_path(task, 0), scene_path(task, 1)
load_scene(env, s0); H1 = hidden_state.capture(env)
c = env.robots[0].composite_controller
flags1 = {n: p.new_update for n, p in c.part_controllers.items()}
load_scene(env, s1)
load_scene(env, s0); H2 = hidden_state.capture(env)
bad = [(k, d) for k, d in cmp(H1, H2) if "buffers" not in k]
print("RESULT", json.dumps(dict(task=task, diffs=bad, new_update_after_load=flags1), default=str))
