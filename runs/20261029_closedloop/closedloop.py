"""Closed-loop chunk steering with the simulator as oracle (2026-10-29): how close does choosing chunks come to the
stage-level imagine target (../20261026_imagine), and how many chunks actually need a choice?

One episode from the frozen scene's start, executed for real, up to `horizon` steps. At every chunk boundary:
  - the current node (sim state, python fixture state, stage memory S, controller state via hidden_state.capture, which is
    exact here because every node is reached by stepping) is imagined N times: candidate i = the policy's chunk with noise
    seed i, then the plain policy until the current stage's check holds or L[k] steps pass (one continuation each);
  - critical = some candidates finish the stage and some do not. Then the chosen candidate is the earliest finisher
    (steered). Otherwise candidate 0 (what the plain policy would draw);
  - the chosen candidate's first chunk (the actions recorded while imagining) is executed for real, and the episode moves
    on 16 steps. Imagining does not count against the horizon.
Logged per chunk: t, stage k, how many of the N candidates finished, steered or not, chosen candidate, their steps.
Per episode: success, steps, chunks, steered chunks, policy calls of the imagining. Checked live: the executed chunk
reaches exactly the imagined state after 16 steps (it raises otherwise).
usage: python -I closedloop.py --tag KettleBoiling_s2_single --port 5751 --out DIR [--N 8] [--horizon 1000]"""
import argparse, copy, hashlib, json, os, sys, time
import numpy as np
import gymnasium as gym

R = "/home/michaellee/mclee/robocasa/runs"
sys.path[:0] = [f"{R}/scripts", f"{R}/20261026_imagine", f"{R}/20261026_critic_data"]
from atomic_eval import load_scene, rollout  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
from groot_eval import Client  # noqa: E402
from imagine import py_state, set_py_state  # noqa: E402
from collect import Recorder  # noqa: E402
import hidden_state  # noqa: E402

L_STAGE = {"KettleBoiling": [416, 272], "SteamInMicrowave": [432, 720, 576, 416], "PackIdenticalLunches": [416, 576, 608, 496]}
def scene(task, tag):
    """the frozen scene of an episode, from the critic data's job templates (s20-29 live in scenes_seen5_v2_s30)"""
    return next(j["scene"] for j in json.load(open(f"{R}/20261026_critic_data/jobs/{task}.json")) if j["tag"] == tag)


def seed(tag, t, i, ci):
    return int.from_bytes(hashlib.sha256(f"{tag}|closedloop|t{t}|c{i}|{ci}".encode()).digest()[:8], "little")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--port", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--horizon", type=int, default=1000)
    a = ap.parse_args()
    task = a.tag.split("_s")[0]; si = int(a.tag.split("_s")[1].split("_")[0])
    steps = TASKS[task]; L = L_STAGE[task]
    base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
    kenv = base.env
    client = Client(port=a.port, timeout_ms=600000)
    start = load_scene(kenv, scene(task, a.tag))
    instr = kenv.get_ep_meta()["lang"]
    node = dict(state=start, ps=py_state(kenv), hidden=hidden_state.capture(kenv), S={}, k=0, t=0)
    log, success, calls, t0 = [], False, 0, time.time()
    os.makedirs(a.out, exist_ok=True)
    while node["t"] + 16 <= a.horizon and node["k"] < len(steps):
        k, t = node["k"], node["t"]
        budget = min(L[k], a.horizon - t)
        cands = []
        for i in range(a.N):
            rec, S, st = Recorder(client), copy.deepcopy(node["S"]), []
            set_py_state(kenv, node["ps"])
            r, _ = rollout(rec, base, kenv, S, steps, node["state"], budget, "task", instr, k0=k, k_stop=k + 1,
                           seed_fn=lambda ci, i=i: seed(a.tag, t, i, ci), step_states=st, hidden=node["hidden"])
            calls += len(rec.chunks)
            cands.append(dict(done=bool(r["reached"] > k or r["success"]), steps=int(r["steps"]), chunk=rec.chunks[0],
                              s16=st[min(16, len(st) - 1)]))
        done = [c["done"] for c in cands]
        critical = 0 < sum(done) < a.N
        pick = min((i for i in range(a.N) if done[i]), key=lambda i: cands[i]["steps"]) if critical else 0
        # execute the chosen chunk for real: the recorded actions, from the exact node
        S = copy.deepcopy(node["S"])
        set_py_state(kenv, node["ps"])
        r, _ = rollout(Recorder(client, first=cands[pick]["chunk"]), base, kenv, S, steps, node["state"], 16, "task", instr,
                       k0=k, k_stop=k + 1, seed_fn=lambda ci: seed(a.tag, t, pick, ci), hidden=node["hidden"])
        exact = bool(np.array_equal(kenv.sim.get_state().flatten(), cands[pick]["s16"]))  # execution == imagination
        if not exact:
            raise RuntimeError(f"t{t}: the executed chunk diverged from the imagined one")
        log.append(dict(t=t, k=k, n_done=int(sum(done)), steered=critical, pick=pick, plain_done=done[0], steps=[c["steps"] for c in cands]))
        print(json.dumps(log[-1]), flush=True)
        node = dict(state=kenv.sim.get_state().flatten().copy(), ps=py_state(kenv), hidden=hidden_state.capture(kenv), S=S,
                    k=max(k, int(r["reached"])), t=t + int(r["steps"]))
        if r["success"]:
            success = True
            break
    res = dict(tag=a.tag, task=task, success=success, final_check=bool(kenv._check_success()), steps=node["t"], stage=node["k"],
               chunks=len(log), steered=sum(c["steered"] for c in log), steered_plain_would_fail=sum(c["steered"] and not c["plain_done"] for c in log),
               policy_calls=calls, N=a.N, horizon=a.horizon, wall_s=round(time.time() - t0, 1), log=log)
    with open(os.path.join(a.out, f"{a.tag}.json"), "w") as f:
        json.dump(res, f)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)


if __name__ == "__main__":
    main()
