"""Pre-flight for the closed loop and imagine on the 4-stage tasks (2026-10-30): restore fidelity through the policy.
One episode per run, from its frozen scene:
  1. reset: load_scene in a fresh env vs load_scene after a 64-step rollout in the same env. Hidden state (controllers,
     clocks, last observation), python fixture state and sim state must be identical, else plain baselines run one
     after another by a worker inherit the previous rollout.
  2. the plain path, node by node as closedloop.py commits (candidate 0's seeds, capture(kenv, obs=True) after every
     chunk), until success or the horizon. At every node:
     - restore (atomic_eval.restore_node, the path of every rollout): the sim state and the python fixture state equal
       the node's; _check_success() and the stage checks before and after are recorded;
     - success bookkeeping: the reward during the chunk, the official check at its end, the stage index.
  3. every --every_cl chunks, closed-loop A/B: candidate 0 run continuously for 64 steps (A) vs its first chunk committed,
     the node captured and restored, then continued with the same seeds (B), and continued by replaying A's own later
     chunks (B2: physics and python fixture state, free of render noise). Compared at every chunk boundary: sim
     state, the observation sent (first difference classified: state keys / images only = render noise / action / sim
     state), python fixture state, stage memory S and _check_success().
  4. every --every_im chunks, imagine A/B through imagine.candidate: one 64-step candidate vs a 16-step candidate, its
     child node as imagine_episode builds it, and a 48-step candidate with the following seeds.
  Each test point records the state: stage, base moved in the chunk, base yaw vs start, objects held, finger contact
  with a fixture, fixture doors.
Output: preflight/<tag>.json (summary: preflight_summary.py).
usage: python -I preflight.py --tag SteamInMicrowave_s24_single --port 5751 [--every_cl 4] [--every_im 8]"""
import argparse, copy, json, os, sys, time
import numpy as np
import gymnasium as gym

R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # runs/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [f"{R}/scripts", f"{R}/20261026_imagine", f"{R}/20261026_critic_data", HERE]
import atomic_eval  # noqa: E402
from atomic_eval import load_scene, rollout, restore_node  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
from groot_eval import Client  # noqa: E402
import imagine  # noqa: E402
from imagine import py_state, set_py_state, progress_ref  # noqa: E402
from collect import Recorder  # noqa: E402
from closedloop import L_STAGE, scene, seed  # noqa: E402
import robocasa.utils.object_utils as OU  # noqa: E402
import hidden_state  # noqa: E402


def diff(a, b, path=""):
    """paths where two nested structures (dicts, lists, arrays, scalars) differ"""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b), key=str) for p in
                ([f"{path}/{k} missing"] if (k in a) != (k in b) else diff(a[k], b[k], f"{path}/{k}"))]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path} len {len(a)} vs {len(b)}"]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return [] if a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f") else [path]
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return []
    return [] if a == b else [path]


def first_diff(dA, dB, off):
    """first chunk i where A's chunk off + i and B's chunk i differ: (i, kind)"""
    for i, (a, b) in enumerate(zip(dA[off:], dB)):
        if a[0] != b[0]:
            keys = sorted(k for k in a[3] if a[3][k] != b[3].get(k))
            return i, ("images only" if all(k.startswith("video") for k in keys) else "obs " + ",".join(keys))
        if a[1] != b[1]:
            return i, "action"
        if a[2] != b[2]:
            return i, "state"
    return None, None


def checks(kenv, steps, S):
    return dict(success=bool(kenv._check_success()), stages=[bool(st["check"](kenv, copy.deepcopy(S))) for st in steps])


def describe(base, kenv, ref_yaw, prev_pos):
    """the robot / object situation at a node"""
    o = base.get_observation(kenv._get_observations(force_update=False))
    pos = np.asarray(o["state.base_position"], float).ravel()
    q = np.asarray(o["state.base_rotation"], float).ravel()  # robot0_base_quat, robosuite order x y z w; the base only yaws
    yaw = float(np.degrees(2 * np.arctan2(q[2], q[3])))
    held = sorted(n for n in kenv.objects if OU.check_obj_grasped(kenv, n))
    r = kenv.robots[0]
    fc = bool(hidden_state.finger_contacts(kenv, r, "right")[4])
    doors = {}
    for n, fx in kenv.fixtures.items():
        if hasattr(fx, "is_open") and type(fx).__name__ in ("Microwave", "SingleCabinet", "HingeCabinet", "Fridge", "FridgeFrenchDoor", "FridgeSideBySide"):
            try:
                doors[n] = bool(fx.is_open(kenv))
            except Exception:
                pass
    return dict(base_moved_mm=None if prev_pos is None else round(float(np.linalg.norm(pos - prev_pos)) * 1000, 1),
                yaw_deg=round(((yaw - ref_yaw + 180) % 360) - 180, 1) if ref_yaw is not None else 0.0, _yaw=yaw, _pos=pos,
                held=held, finger_fixture_contact=fc, doors_open=sorted(n for n, v in doors.items() if v))


class Replay:
    """client that answers with recorded chunks in order (no policy call): replays a rollout's actions exactly"""
    def __init__(self, chunks):
        self.todo, self.chunks, self.req0 = list(chunks), [], None

    def get_action(self, req):
        c = self.todo[len(self.chunks)]
        self.chunks.append(c)
        return dict({k: np.asarray(v)[None] for k, v in c.items()}, _seed_echo=req.get("_seed"))


def hooked(rec):
    """boundary_hook recording (t, python fixture state, S, _check_success) at every chunk boundary"""
    return lambda t, kenv, S, rewards: rec.append(dict(t=t, ps=py_state(kenv), S=copy.deepcopy(S), success=bool(kenv._check_success())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "preflight"))
    ap.add_argument("--every_cl", type=int, default=4); ap.add_argument("--every_im", type=int, default=8)
    ap.add_argument("--stop", type=int, default=None, help="stop the plain path at this step (default: the horizon)")
    a = ap.parse_args()
    task = a.tag.split("_s")[0]; steps = TASKS[task]; L = L_STAGE[task]
    H = json.load(open(f"{R}/horizons_original_plus.json"))[task]
    base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
    kenv = base.env
    client = Client(port=a.port, timeout_ms=600000)
    path = scene(task, a.tag)
    t0 = time.time()

    # 1. reset
    start = load_scene(kenv, path)
    instr = kenv.get_ep_meta()["lang"]
    h1, ps1, s1 = hidden_state.capture(kenv, obs=True), py_state(kenv), kenv.sim.get_state().flatten().copy()
    rollout(Recorder(client), base, kenv, {}, steps, start, 64, "task", instr, seed_fn=lambda ci: seed(a.tag, -1, 0, ci))
    start2 = load_scene(kenv, path)
    h2, ps2, s2 = hidden_state.capture(kenv, obs=True), py_state(kenv), kenv.sim.get_state().flatten().copy()
    reset = dict(hidden=diff(h1, h2), ps=diff(ps1, ps2), state=diff(s1, s2) + diff(start, start2))
    print(json.dumps(dict(reset=reset)), flush=True)

    # 2-4. the plain path
    node = dict(state=start2, ps=ps2, hidden=hidden_state.capture(kenv), S={}, k=0, t=0)  # t = 0: forced, as a plain run
    d0 = describe(base, kenv, None, None); ref_yaw, prev_pos = d0["_yaw"], d0["_pos"]
    nodes, cl, im = [], [], []
    j = 0
    while node["t"] + 16 <= (a.stop or H):
        k, t = node["k"], node["t"]
        desc = describe(base, kenv, ref_yaw, prev_pos) if j else d0
        prev_pos = desc["_pos"]
        desc = {kk: v for kk, v in desc.items() if not kk.startswith("_")}
        sf = lambda ci, t=t: seed(a.tag, t, 0, ci)
        # 3. closed-loop A/B
        if j % a.every_cl == 0 and j:
            recA, stA, dA, hA, SA = Recorder(client), [], [], [], copy.deepcopy(node["S"])
            set_py_state(kenv, node["ps"])
            rA, _ = rollout(recA, base, kenv, SA, steps, node["state"], 64, "task", instr, k0=k, seed_fn=sf, step_states=stA,
                            digest=dA, hidden=node["hidden"], boundary_hook=hooked(hA))
            S1 = copy.deepcopy(node["S"])
            set_py_state(kenv, node["ps"])
            r1, _ = rollout(Recorder(client, first=recA.chunks[0]), base, kenv, S1, steps, node["state"], 16, "task", instr, k0=k,
                            k_stop=k + 1, seed_fn=sf, hidden=node["hidden"])
            n1 = dict(state=kenv.sim.get_state().flatten().copy(), ps=py_state(kenv), hidden=hidden_state.capture(kenv, obs=True), S=S1)
            stB, dB, hB, SB = [], [], [], copy.deepcopy(S1)
            set_py_state(kenv, n1["ps"])
            rB, _ = rollout(Recorder(client), base, kenv, SB, steps, n1["state"], 48, "task", instr, k0=int(r1["reached"]),
                            seed_fn=lambda ci: sf(ci + 1),
                            step_states=stB, digest=dB, hidden=n1["hidden"], boundary_hook=hooked(hB))
            # B2: the same restore, then A's own later chunks replayed (no policy call), so the physics and the python
            # fixture state are compared independently of render noise (review 2026-10-30)
            stR, hR, SR, rR = [], [], copy.deepcopy(S1), dict(success=None, reached=None)
            if len(recA.chunks) > 1:  # A stopped after its first chunk only if it succeeded there
                set_py_state(kenv, n1["ps"])
                rR, _ = rollout(Replay(recA.chunks[1:]), base, kenv, SR, steps, n1["state"], 16 * (len(recA.chunks) - 1), "task",
                                instr, k0=int(r1["reached"]), seed_fn=lambda ci: sf(ci + 1), step_states=stR, hidden=n1["hidden"],
                                boundary_hook=hooked(hR))
            nR = min(len(stA) - 16, len(stR))
            replay = dict(states_equal=nR > 0 and all(np.array_equal(stA[16 + i], stR[i]) for i in range(nR)), steps=nR,
                          ps=[(y["t"], diff(x["ps"], y["ps"])) for x, y in zip(hA[1:], hR) if diff(x["ps"], y["ps"])],
                          S_end=diff(SA, SR), success=(rA["success"], rR["success"]), reached=(rA["reached"], rR["reached"]))
            n = min(len(stA) - 16, len(stB))
            same = [bool(np.array_equal(stA[16 + i], stB[i])) for i in range(n)]
            fdi, fdk = first_diff(dA, dB, 1)
            bnd = [dict(t=y["t"], state_equal=bool(len(stA) > 16 + y["t"] and len(stB) > y["t"] and np.array_equal(stA[16 + y["t"]], stB[y["t"]])),
                        ps=diff(x["ps"], y["ps"]), S=diff(x["S"], y["S"]), success=(x["success"], y["success"]))
                   for x, y in zip(hA[1:], hB)]
            row = dict(t=t, k=k, **desc, states_equal=all(same), first_state_diff=same.index(False) if False in same else None,
                       first_chunk_diff=fdi, diff_kind=fdk, node_ps=diff(hA[0]["ps"], n1["ps"]) if hA else ["no boundary"],
                       replay=replay,
                       boundary_ps=[b["ps"] for b in bnd if b["ps"]], boundary_S=[b["S"] for b in bnd if b["S"]],
                       ps_diff_while_state_equal=[(b["t"], b["ps"] + b["S"]) for b in bnd if b["state_equal"] and (b["ps"] or b["S"])],
                       boundary_success_differs=[b["t"] for b in bnd if b["success"][0] != b["success"][1]],
                       A=dict(steps=rA["steps"], success=rA["success"], reached=rA["reached"]),
                       B=dict(steps=16 + rB["steps"], success=rB["success"], reached=rB["reached"]))
            cl.append(row); print(json.dumps(dict(cl=row)), flush=True)
        # 4. imagine A/B
        if j % a.every_im == 0 and j:
            nd = dict(state=node["state"], ps=node["ps"], hidden=node["hidden"], S=node["S"])
            dig = []
            orig = imagine.rollout
            imagine.rollout = lambda *x, **kw: orig(*x, **dict(kw, digest=dig))
            try:
                imagine.restore(kenv, nd["state"], nd["ps"]); ref = progress_ref(task, k, kenv)
                dig.clear(); cA = imagine.candidate(client, base, kenv, task, steps, nd, k, 64, instr, sf, ref); dA = list(dig)
                imagine.restore(kenv, nd["state"], nd["ps"])
                dig.clear(); c1 = imagine.candidate(client, base, kenv, task, steps, nd, k, 16, instr, sf, ref)
                if c1["reached"] > k or c1["success"]:
                    row = dict(t=t, k=k, **desc, skipped="the stage finished in the first chunk")
                else:
                    child = dict(state=c1["end_state"], ps=c1["ps"], hidden=c1["hidden"], S=c1["S"])
                    imagine.restore(kenv, child["state"], child["ps"])
                    dig.clear(); cB = imagine.candidate(client, base, kenv, task, steps, child, k, 48, instr, lambda ci: sf(ci + 1), ref)
                    dB = list(dig)
                    fdi, fdk = first_diff(dA, dB, 1)
                    row = dict(t=t, k=k, **desc, end_equal=bool(np.array_equal(cA["end_state"], cB["end_state"])),
                               first_chunk_diff=fdi, diff_kind=fdk, ps=diff(cA["ps"], cB["ps"]), S=diff(cA["S"], cB["S"]),
                               A=dict(steps=cA["steps"], done=cA["done"], success=cA["success"]),
                               B=dict(steps=16 + cB["steps"], done=cB["done"], success=cB["success"]))
            finally:
                imagine.rollout = orig
            im.append(row); print(json.dumps(dict(im=row)), flush=True)
        # the plain step: commit candidate 0's chunk as closedloop.py does
        S = copy.deepcopy(node["S"])
        set_py_state(kenv, node["ps"])
        r, _ = rollout(Recorder(client), base, kenv, S, steps, node["state"], 16, "task", instr, k0=k, k_stop=k + 1,
                       seed_fn=sf, hidden=node["hidden"])
        node = dict(state=kenv.sim.get_state().flatten().copy(), ps=py_state(kenv), hidden=hidden_state.capture(kenv, obs=True), S=S,
                    k=max(k, int(r["reached"])), t=t + int(r["steps"]))
        # 2. restore at the new node, through the same path as every rollout (restore_node raises if the sim state
        # differs). The checks can differ because they see freshly recomputed positions after a restore, not the
        # one-substep-old ones a step leaves; they are recorded, not treated as errors.
        before = checks(kenv, steps, S)
        set_py_state(kenv, node["ps"]); restore_node(kenv, node["state"], node["hidden"])
        after = checks(kenv, steps, S)
        nodes.append(dict(t=node["t"], k=node["k"], reward=bool(r["success"]), official_end=before["success"],
                          restore_state_equal=bool(np.array_equal(kenv.sim.get_state().flatten(), node["state"])),
                          restore_ps=diff(node["ps"], py_state(kenv)), restore_checks=diff(before, after)))
        if nodes[-1]["restore_ps"] or not nodes[-1]["restore_state_equal"] or nodes[-1]["reward"] != nodes[-1]["official_end"]:
            print(json.dumps(dict(node=nodes[-1])), flush=True)
        j += 1
        if r["success"] or node["k"] >= len(steps):
            break
    res = dict(tag=a.tag, task=task, reset=reset, nodes=nodes, cl=cl, im=im, final=dict(t=node["t"], k=node["k"]),
               wall_s=round(time.time() - t0, 1))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.tag}.json"), "w") as f:
        json.dump(res, f, default=str)
    print(json.dumps(dict(done=a.tag, final=res["final"], n_cl=len(cl), n_im=len(im), wall_s=res["wall_s"])), flush=True)


if __name__ == "__main__":
    main()
