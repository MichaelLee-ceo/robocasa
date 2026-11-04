"""Replay a closed-loop episode's committed path (out/<tag>.json) to see where it put the kettle (2026-10-29).
At every logged chunk the chosen candidate's first chunk is re-queried from the policy with the same seed and executed,
as closedloop.py did, without the imagining. Rendering is not bitwise reproducible (../README.md), so the replay is
checked against the log: the stage must change at the same chunks. Per chunk in stage 1 it records the KettleBoiling
stage-1 debug (burner under the kettle, its distance, on the stove).
PackIdenticalLunches: at every stage change it records which tupperware holds each food item (the official check
wants one vegetable and one meat in each); --stop_k stops the replay when that stage is reached.
usage: python -I replay.py --tag KettleBoiling_s5_single --port 5751 --out replay [--stop_k K]"""
import argparse, copy, json, os, sys
import gymnasium as gym

R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # runs/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [f"{R}/scripts", f"{R}/20261026_imagine", f"{R}/20261026_critic_data", HERE]
from atomic_eval import load_scene, rollout  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
from groot_eval import Client  # noqa: E402
from imagine import py_state, set_py_state  # noqa: E402
from collect import Recorder  # noqa: E402
from closedloop import scene, seed  # noqa: E402
import hidden_state  # noqa: E402
import robocasa.utils.object_utils as OU  # noqa: E402

PIL_ITEMS, PIL_BOXES = ("vegetable0", "vegetable1", "meat0", "meat1"), ("tupperware0", "tupperware1")


def placement(kenv):
    """item -> the tupperware it is in (None if neither)"""
    return {o: next((b for b in PIL_BOXES if OU.check_obj_in_receptacle(kenv, o, b)), None) for o in PIL_ITEMS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--port", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stop_k", type=int, default=None)
    a = ap.parse_args()
    ref = json.load(open(os.path.join(HERE, "out", f"{a.tag}.json")))
    task = ref["task"]; steps = TASKS[task]
    base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
    kenv = base.env
    client = Client(port=a.port, timeout_ms=600000)
    start = load_scene(kenv, scene(task, a.tag))
    instr = kenv.get_ep_meta()["lang"]
    node = dict(state=start, ps=py_state(kenv), hidden=hidden_state.capture(kenv), S={}, k=0, t=0)
    rows, match, places = [], True, []
    for c in ref["log"]:
        k, t, pick = node["k"], node["t"], c["pick"]
        if (t, k) != (c["t"], c["k"]):
            match = False
            break
        S = copy.deepcopy(node["S"])
        set_py_state(kenv, node["ps"])
        r, _ = rollout(Recorder(client), base, kenv, S, steps, node["state"], 16, "task", instr, k0=k, k_stop=k + 1,
                       seed_fn=lambda ci, t=t, pick=pick: seed(a.tag, t, pick, ci), hidden=node["hidden"])
        node = dict(state=kenv.sim.get_state().flatten().copy(), ps=py_state(kenv), hidden=hidden_state.capture(kenv, obs=True), S=S,
                    k=max(k, int(r["reached"])), t=t + int(r["steps"]))
        if task == "PackIdenticalLunches" and node["k"] > k:
            places.append(dict(t=node["t"], k=node["k"], placement=placement(kenv)))
            print(json.dumps(places[-1]), flush=True)
        if a.stop_k is not None and node["k"] >= a.stop_k:
            break
        if node["k"] >= 1 and len(steps) > 1 and "debug" in steps[1]:
            rows.append(dict(t=node["t"], k=node["k"], **steps[1]["debug"](kenv, S)))
        if r["success"]:
            break
    res = dict(tag=a.tag, success=ref["success"], match=match, replayed_to=node["t"], stage1=rows, placements=places)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.tag}.json"), "w") as f:
        json.dump(res, f)
    print(json.dumps(dict(res, stage1=rows[:1] + rows[-1:])), flush=True)


if __name__ == "__main__":
    main()
