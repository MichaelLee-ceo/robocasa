"""Restore vs continuous at the closed loop's nodes (2026-10-29): after committing a steered chunk, does the next node
continue exactly like the imagined candidate did?
The episode's committed path is replayed as in replay.py. At each chosen steered chunk (pick p, node at t):
  A: candidate p run continuously from the node, as the closed loop imagined it (seeds seed(tag, t, p, ci));
  B: its first chunk committed as closedloop.py does, the node captured (hidden_state.capture) and restored, then the
     continuation with the same seeds (ci + 1).
A's states after its first chunk must equal B's bit for bit. On a mismatch the first differing chunk is classified with
the per-chunk digests: observation sent (image keys only = render noise, see ../README.md), action, or sim state.
usage: python -I lockin_check.py --tag KettleBoiling_s8_single --port 5751 --out lockin [--max_chunk 12]"""
import argparse, copy, json, os, sys
import numpy as np
import gymnasium as gym

R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # runs/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [f"{R}/scripts", f"{R}/20261026_imagine", f"{R}/20261026_critic_data", HERE]
from atomic_eval import load_scene, rollout  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
from groot_eval import Client  # noqa: E402
from imagine import py_state, set_py_state  # noqa: E402
from collect import Recorder  # noqa: E402
from closedloop import L_STAGE, scene, seed  # noqa: E402
import hidden_state  # noqa: E402


def first_diff(dA, dB):
    """first chunk i where A's chunk 1 + i and B's chunk i differ: (i, what)"""
    for i, (a, b) in enumerate(zip(dA[1:], dB)):
        if a[0] != b[0]:
            keys = sorted(k for k in a[3] if a[3][k] != b[3].get(k))
            return i, "obs " + ",".join(keys)
        if a[1] != b[1]:
            return i, "action"
        if a[2] != b[2]:
            return i, "state"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--port", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max_chunk", type=int, default=12)
    a = ap.parse_args()
    ref = json.load(open(os.path.join(HERE, "out", f"{a.tag}.json")))
    task = ref["task"]; steps = TASKS[task]; L = L_STAGE[task]
    base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
    kenv = base.env
    client = Client(port=a.port, timeout_ms=600000)
    start = load_scene(kenv, scene(task, a.tag))
    instr = kenv.get_ep_meta()["lang"]
    node = dict(state=start, ps=py_state(kenv), hidden=hidden_state.capture(kenv), S={}, k=0, t=0)
    rows = []
    for j, c in enumerate(ref["log"][:a.max_chunk]):
        k, t, p = node["k"], node["t"], c["pick"]
        if (t, k) != (c["t"], c["k"]):
            rows.append(dict(chunk=j, t=t, note=f"replay diverged from the log: (t, k) = {(t, k)}, log {(c['t'], c['k'])}"))
            break
        budget = min(L[k], ref["horizon"] - t)
        if c["steered"]:  # A: continuous
            recA, SA, stA, dA = Recorder(client), copy.deepcopy(node["S"]), [], []
            set_py_state(kenv, node["ps"])
            rA, _ = rollout(recA, base, kenv, SA, steps, node["state"], budget, "task", instr, k0=k, k_stop=k + 1,
                            seed_fn=lambda ci: seed(a.tag, t, p, ci), step_states=stA, digest=dA, hidden=node["hidden"])
        # commit (as closedloop.py): the candidate's first chunk, from the exact node
        recC = Recorder(client)
        S = copy.deepcopy(node["S"])
        set_py_state(kenv, node["ps"])
        r, _ = rollout(recC if not c["steered"] else Recorder(client, first=recA.chunks[0]), base, kenv, S, steps, node["state"],
                       16, "task", instr, k0=k, k_stop=k + 1, seed_fn=lambda ci: seed(a.tag, t, p, ci), hidden=node["hidden"])
        node = dict(state=kenv.sim.get_state().flatten().copy(), ps=py_state(kenv), hidden=hidden_state.capture(kenv, obs=True), S=S,
                    k=max(k, int(r["reached"])), t=t + int(r["steps"]))
        if c["steered"]:  # B: restore at the new node, continue with the same seeds
            stB, dB = [], []
            SB = copy.deepcopy(node["S"])
            set_py_state(kenv, node["ps"])
            rB, _ = rollout(Recorder(client), base, kenv, SB, steps, node["state"], budget - 16, "task", instr, k0=node["k"],
                            k_stop=k + 1, seed_fn=lambda ci: seed(a.tag, t, p, ci + 1), step_states=stB, digest=dB,
                            hidden=node["hidden"])
            n = min(len(stA) - 16, len(stB))
            same = [bool(np.array_equal(stA[16 + i], stB[i])) for i in range(n)]
            fd = first_diff(dA, dB)
            row = dict(chunk=j, t=t, pick=p, logged_steps=c["steps"][p], A_done=bool(rA["reached"] > k or rA["success"]),
                       A_steps=int(rA["steps"]), B_done=bool(rB["reached"] > k or rB["success"]), B_steps=16 + int(rB["steps"]),
                       node_equal=same[0] if same else None, states_equal=all(same), first_state_diff=(same.index(False) if False in same else None),
                       first_chunk_diff=fd[0], diff_kind=fd[1])
            rows.append(row)
            print(json.dumps(row), flush=True)
        if r["success"]:
            break
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.tag}.json"), "w") as f:
        json.dump(dict(tag=a.tag, rows=rows), f)


if __name__ == "__main__":
    main()
