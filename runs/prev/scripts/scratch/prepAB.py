import os
import sys, json, glob
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
import robocasa.utils.object_utils as OU
from atomic_eval import load_scene
from oracle import set_state
from atomic_steps import TASKS
from robot_home import robot_to_scene_start
OUT = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/20261013_pretrain3"; SC = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scenes_seen5_v2"
H = json.load(open(os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/horizons_original_plus.json"))
R = {}
for p in glob.glob(f"{OUT}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r: R[r["tag"]] = r
plan = []
for t, objs in (("ArrangeVegetables", ["vegetable1", "vegetable2"]), ("GatherVegetables", ["veg1", "veg2"])):
    b = gym.make(f"robocasa/{t}", split="target", enable_render=False).unwrapped; e = b.env
    for s in range(20):
        f = f"{OUT}/{t}/{t}_s{s}_single_states.npz"
        recs = [x for x in np.load(f, allow_pickle=True)["recs"] if x["k"] == 1]
        if not recs: continue
        rec = recs[0]; S = dict(rec["S"]); budget = int(H[t] - rec["t"])
        init = load_scene(e, f"{SC}/{t}_s{s}.npz")
        # A: hand-off objects, robot back to the scene's initial joint values
        set_state(e, rec["state"])
        infoA = robot_to_scene_start(e, init)
        a_ok = dict(first_on_board=bool(OU.check_obj_in_receptacle(e, S["first"], "cutting_board")),
                    second_on_board=bool(OU.check_obj_in_receptacle(e, S["second"], "cutting_board")),
                    second_in_sink=bool(OU.obj_inside_of(e, S["second"], e.sink)))
        instrA = TASKS[t][1]["instr"](e, S)
        # B: initial scene, sentence names the object the single-horizon run placed second
        load_scene(e, f"{SC}/{t}_s{s}.npz")
        instrB = TASKS[t][0]["instr"](e, dict(first=S["second"]))
        b_ok = dict(target_in_sink=bool(OU.obj_inside_of(e, S["second"], e.sink)), on_board=bool(OU.check_obj_in_receptacle(e, S["second"], "cutting_board")))
        plan.append(dict(task=t, seed=s, first=S["first"], second=S["second"], names=[OU.get_obj_lang(e, obj_name=o) for o in (S["first"], S["second"])],
                         start_t=int(rec["t"]), budget=budget, A=dict(**infoA, **a_ok, instr=instrA), B=dict(**b_ok, instr=instrB)))
json.dump(plan, open(f"{OUT}/plan_AB.json", "w"), ensure_ascii=False, indent=1)
for p in plan:
    print(p["task"][:6], p["seed"], p["names"], "budget", p["budget"], "| A coll", p["A"]["collision"], "moved", p["A"]["max_obj_moved"], "first_on", p["A"]["first_on_board"], "second_on", p["A"]["second_on_board"], "sink", p["A"]["second_in_sink"], "| B target_in_sink", p["B"]["target_in_sink"])
print("A sentence example:", plan[0]["A"]["instr"]); print("B sentence example:", plan[0]["B"]["instr"])
