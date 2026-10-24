import sys, os, json, glob, collections
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
import robocasa.utils.object_utils as OU
from atomic_eval import load_scene
from oracle import set_state
OUT = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/20261013_pretrain3"
R = {}
for p in glob.glob(f"{OUT}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r: R[r["tag"]] = r
OBJ = dict(ArrangeVegetables=["vegetable1", "vegetable2"], GatherVegetables=["veg1", "veg2"], CondimentCollection=["condiment1", "condiment2"])
out = []
for t, objs in OBJ.items():
    b = gym.make(f"robocasa/{t}", split="target", enable_render=False).unwrapped; e = b.env
    done = (lambda o: OU.check_obj_in_receptacle(e, o, "cutting_board")) if "Veg" in t else (lambda o: OU.obj_inside_of(e, o, e.cab))
    for tag, r in sorted(R.items()):
        if r["task"] != t or r["success"]: continue
        kind = "chain" if r["tag"].endswith("_aseen_chain") and r["reached"] == 1 else ("solo" if r["tag"].endswith("_aseen_solo_k1") else None)
        if kind is None: continue
        load_scene(e, r["scene"]); set_state(e, np.load(f"{OUT}/{t}/{tag}_end.npz", allow_pickle=True)["state"])
        first = r["S_final"].get("first"); second = r["S_final"].get("second")
        d = dict(task=t, kind=kind, seed=r["seed"], first_still_done=bool(done(first)), second_placed=bool(done(second)),
                 far_first=bool(OU.gripper_obj_far(e, obj_name=first)), far_second=bool(OU.gripper_obj_far(e, obj_name=second)))
        if t == "ArrangeVegetables": d["far_board"] = bool(OU.gripper_obj_far(e, obj_name="cutting_board"))
        out.append(d)
json.dump(out, open(f"{OUT}/step2_failure_conditions.json", "w"), indent=1)
G = collections.defaultdict(list)
for d in out: G[(d["task"], d["kind"])].append(d)
for k, g in sorted(G.items()):
    cats = collections.Counter(("first knocked off" if not d["first_still_done"] else "") + ("" if d["first_still_done"] else "") +
                               ("; second placed" if d["second_placed"] else "; second NOT placed") for d in g)
    print(k, "n=%d" % len(g), "first_still_done", sum(d["first_still_done"] for d in g), "second_placed", sum(d["second_placed"] for d in g),
          "far_second", sum(d["far_second"] for d in g), ("far_board %d" % sum(d.get("far_board", False) for d in g)) if k[0] == "ArrangeVegetables" else "", dict(cats))
