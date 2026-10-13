import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robosuite, robocasa
from robosuite.controllers import load_composite_controller_config
from demo_utils import load_demo, n_episodes
from oracle import reset_to, set_state
from atomic_steps import TASKS, prepare
task = sys.argv[1]
k = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"),
    has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, use_object_obs=True,
    obj_instance_split="target", layout_and_style_ids=list(zip(range(1, 11), range(1, 11))))
N = n_episodes(task); ok = 0
for ep in [int(i * N / 10) for i in range(10)]:
    d = load_demo(task, ep); reset_to(k, d["xml"], d["ep_meta"]); S = prepare(k, d, set_state)
    reset_to(k, d["xml"], d["ep_meta"])
    steps = TASKS[task]; segs = d["segs"]
    starts = [segs[s["seg"]]["start"] for s in steps]
    fired, cur = [], 0
    false_at_start = []
    for i, s in enumerate(steps):
        set_state(k, d["states"][starts[i]]); false_at_start.append(not s["check"](k, S))
    reset_to(k, d["xml"], d["ep_meta"])
    for f in range(len(d["states"])):
        set_state(k, d["states"][f])
        while cur < len(steps) and steps[cur]["check"](k, S):
            fired.append(f); cur += 1
    set_state(k, d["states"][starts[min(1, len(steps)-1)]]); instr = [s["instr"](k, S) for s in steps]
    good = cur == len(steps)
    ok += good
    print(json.dumps(dict(ep=ep, all_fired=good, fired_at=fired, step_starts=starts, false_at_start=false_at_start, S=S, instr=instr)))
print("SUMMARY", task, ok, "/ 10")
