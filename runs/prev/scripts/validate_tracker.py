import sys, json, os, numpy as np, gymnasium as gym
sys.path.insert(0, os.path.dirname(__file__))
import robocasa  # noqa
from demo_utils import load_demo
from oracle import reset_to, set_state, build_subtask_specs, SubtaskTracker

task = sys.argv[1]; eps = [int(x) for x in sys.argv[2].split(",")]
import robosuite
from robosuite.controllers import load_composite_controller_config
k = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"),
                   has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, use_object_obs=True,
                   obj_instance_split="target", layout_and_style_ids=list(zip(range(1, 11), range(1, 11))), translucent_robot=False)
out = []
for ep in eps:
    d = load_demo(task, ep)
    reset_to(k, d["xml"], d["ep_meta"])
    specs, P = build_subtask_specs(k, d)
    set_state(k, d["states"][0])
    tr = SubtaskTracker(k, specs, P)
    for f in range(len(d["states"])):
        set_state(k, d["states"][f]); tr.update(f)
    got = [h[0] for h in tr.history]
    want = [s["demo_end"] for s in specs]
    out.append(dict(ep=ep, n=len(specs), reached=len(got), lag=[g - w for g, w in zip(got, want)],
                    success_end=bool(k._check_success()),
                    specs=[(s["stage"], s.get("obj"), s.get("new_support_names"), s.get("joints")) for s in specs]))
    print(json.dumps(out[-1], default=str))
print("SUMMARY", task, "all reached:", sum(o["reached"] == o["n"] for o in out), "/", len(out))
