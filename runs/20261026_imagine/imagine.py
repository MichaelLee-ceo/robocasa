"""Stage-level "imagine, then choose" with the simulator as a perfect world model (oracle upper bound), KettleBoiling.

Imagination: from a saved sim state, each of N candidates runs the frozen policy closed-loop with its own action noise
until the stage's official condition holds (with the earlier stages still holding) or L steps pass.
Choice: a best-first search over these stage-level candidates (imagine_episode): finishers of a stage become nodes of
the next stage, earliest first; if none finished, the candidate with the most progress (generic per stage kind, see
progress()) continues the stage. A branch whose next stage cannot be finished is abandoned for the next-earliest finisher
of the previous stage (backtracking). The found path is committed by setting the env to its end state; the episode clock
counts only the committed steps (imagining takes no episode time; its cost is logged separately).
condition "baseline": a plain rollout from the scene start with policy_seed ps (same seeding as runs/20261025_seen480).
Units: 1 step = 0.05 s; chunk = 16 steps. usage: python imagine.py --jobs jobs.json --worker W --n_workers N --out DIR --port P"""
import argparse, copy, glob, hashlib, json, os, sys, time, traceback
import numpy as np
import gymnasium as gym

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from atomic_eval import load_scene, rollout, policy_seed_fn  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
from groot_eval import Client, write_video  # noqa: E402
from oracle import set_state  # noqa: E402
from stage_trace import STAGE_SPECS, _burner_dist, _nearest_burner, _knob_onness  # noqa: E402
import robocasa.utils.object_utils as OU  # noqa: E402
import hidden_state  # noqa: E402

FIXTURE_ON = {"stove": 0.35}  # official on-threshold of the fixture value used by progress() (default 1.0)


from hidden_state import py_state, set_py_state, _plain, _like  # noqa: E402,F401  (moved there 2026-10-30)


def restore(kenv, state, ps):
    """sim state + python fixture state; the python state first, because set_state runs update_state."""
    set_py_state(kenv, ps)
    set_state(kenv, state)


def progress(task, k, e, ref):
    """generic stage progress in [0, 1] at the current state; ref = values captured at the decision point.
    pnp stage: per object 1 if at the target, else 0.5 * (held or lifted >= 3 cm) + 0.5 * share of the start distance
    to the target already covered; averaged over the stage's objects. fixture stage: the official value over its
    on-threshold (KettleBoiling: the knob of the burner under the kettle, as the official condition requires)."""
    sp = STAGE_SPECS[task][k]
    if sp["kind"] == "pnp":
        ps = []
        for o in sp["objs"]:
            if sp["at_target"](e, o):
                ps.append(1.0); continue
            z = float(e.sim.data.body_xpos[e.obj_body_id[o]][2])
            held = OU.check_obj_grasped(e, o) or z >= ref["z"][o] + 0.03
            d0 = max(ref["dist"][o], 1e-3)
            ps.append(0.5 * held + 0.5 * float(np.clip(1 - sp["dist"](e, o) / d0, 0, 1)))
        return float(np.mean(ps))
    if task == "KettleBoiling":
        return float(min(_knob_onness(e, _nearest_burner(e, "obj")[0]) / FIXTURE_ON["stove"], 1.0))
    return float(min(sp["value"](e) / FIXTURE_ON.get(sp["fixture"], 1.0), 1.0))


def progress_ref(task, k, e):
    sp = STAGE_SPECS[task][k]
    if sp["kind"] != "pnp":
        return {}
    return dict(z={o: float(e.sim.data.body_xpos[e.obj_body_id[o]][2]) for o in sp["objs"]},
                dist={o: float(sp["dist"](e, o)) for o in sp["objs"]})


def candidate(client, base, kenv, task, steps, nd, k, budget, instr, seed_fn, ref):
    """one imagined continuation from node nd (sim state, python fixture state, stage-check memory S, controller
    state). The node's controller state is the one its producing rollout ended with, so a committed path executes
    exactly as imagined (runs before 2026-10-27 did not restore it; see ../scripts/hidden_state.py). The node also keeps
    its last observation (capture(obs=True), 2026-10-29), so the next stage's rollouts continue from it like a continuous
    run instead of starting with a forced, fresher observation."""
    import copy
    S = copy.deepcopy(nd["S"])
    set_py_state(kenv, nd["ps"])  # rollout() then sets the sim state (and runs update_state)
    r, frames = rollout(client, base, kenv, S, steps, nd["state"], budget, "task", instr, k0=k, k_stop=k + 1, seed_fn=seed_fn,
                        hidden=nd["hidden"])
    done = bool(r["reached"] > k or r["success"])
    # prev_ok (whether every earlier stage check still holds at the end) is logged only. Until 2026-10-30 a finisher also
    # needed it, unlike the closed loop, the critic data and the stage history of plain runs. On SteamInMicrowave it
    # rejected about 20 % of the stage-2 (close the door) finishers, even one with official success, mostly because an
    # earlier check's release condition (gripper 25 cm from the bowl) fails while the gripper pushes the door.
    prev_ok = all(steps[j]["check"](kenv, copy.deepcopy(S)) for j in range(k)) if (done and k < len(steps) - 1) else True
    return dict(done=done, raw_done=done, prev_ok=prev_ok, success=bool(r["success"]), steps=int(r["steps"]),
                reached=int(r["reached"]), progress=round(progress(task, k, kenv, ref), 3),
                end_state=kenv.sim.get_state().flatten().copy(), ps=py_state(kenv), hidden=hidden_state.capture(kenv, obs=True),
                S=S, frames=frames)


def imagine_episode(client, base, kenv, j, a):
    """stage-level best-first search over the policy's own imagined rollouts.
    Node = (sim state, stage k, steps used t). Expanding a node imagines N candidates of stage k from its state (each up
    to L steps). Children: the candidates that finished the stage (at most `keep` of them, earliest first) as nodes of
    stage k+1, and one continuation node of stage k from the candidate with the most progress (only if none finished).
    The frontier is ordered by (deeper stage first, fewer steps used, more progress), so when a branch fails at the next
    stage the search falls back to the next-earliest finisher of the previous stage: backtracking. Stops at a node that
    completes the task (official _check_success), or when the candidate budget or the horizon runs out. The first
    descent without backtracking is the greedy choice (logged as `backtracked`)."""
    import heapq
    task, tag, N, H, keep, cap = j["task"], j["tag"], j["N"], j["horizon"], j.get("keep", 3), j.get("max_candidates", 40)
    steps = TASKS[task]
    s0 = load_scene(kenv, j["scene"])
    instr = kenv.get_ep_meta()["lang"]
    vdir = f"{a.out}/videos/{tag}"; os.makedirs(vdir, exist_ok=True)
    nodes = [dict(id=0, parent=None, via=None, state=s0, ps=py_state(kenv), hidden=hidden_state.capture(kenv), S={}, k=0, t=0,
                  progress=0.0, success=False, first_choice=True)]
    frontier = [((0, 0, 0.0, 0), 0)]
    expansions, n_cands, imagined_steps, policy_calls, goal = [], 0, 0, 0, None
    while frontier and n_cands + N <= cap:
        _, nid = heapq.heappop(frontier)
        nd = nodes[nid]
        if nd["success"] or nd["k"] >= len(steps):
            goal = nid; break
        L = min(j["L"][nd["k"]], H - nd["t"])
        if L < 16:
            continue
        restore(kenv, nd["state"], nd["ps"]); ref = progress_ref(task, nd["k"], kenv)
        ei, cands = len(expansions), []
        for i in range(N):
            seed_fn = lambda ci, ei=ei, i=i: int.from_bytes(hashlib.sha256(f"{tag}|search|{ei}|{i}|{ci}".encode()).digest()[:8], "little")
            c = candidate(client, base, kenv, task, steps, nd, nd["k"], L, instr, seed_fn, ref)
            imagined_steps += c["steps"]; policy_calls += -(-c["steps"] // 16); n_cands += 1
            if c["frames"]:
                write_video(f"{vdir}/e{ei}_c{i}.mp4", c["frames"], fps=10)
            c.pop("frames"); cands.append(c)
        fin = sorted([i for i, c in enumerate(cands) if c["done"]], key=lambda i: cands[i]["steps"])[:keep]
        kids = []
        for rank, i in enumerate(fin):
            c = cands[i]
            kids.append(dict(via=(ei, i), state=c["end_state"], ps=c["ps"], hidden=c["hidden"], S=c["S"], k=max(nd["k"] + 1, c["reached"]),
                             t=nd["t"] + c["steps"], progress=0.0, success=c["success"], first_choice=nd["first_choice"] and rank == 0))
        if not fin:
            i = int(np.argmax([c["progress"] for c in cands]))
            c = cands[i]
            kids.append(dict(via=(ei, i), state=c["end_state"], ps=c["ps"], hidden=c["hidden"], S=c["S"], k=nd["k"], t=nd["t"] + c["steps"],
                             progress=c["progress"], success=c["success"], first_choice=nd["first_choice"]))
        for kd in kids:
            kd.update(id=len(nodes), parent=nid); nodes.append(kd)
            heapq.heappush(frontier, ((-kd["k"], kd["t"], -kd["progress"], kd["id"]), kd["id"]))
        expansions.append(dict(node=nid, k=nd["k"], t=nd["t"], L=L, finishers=fin,
                               cands=[dict(done=c["done"], raw_done=c["raw_done"], prev_ok=c["prev_ok"], steps=c["steps"],
                                           success=c["success"], reached=c["reached"], progress=c["progress"]) for c in cands]))
    if goal is None:  # also accept a finished node still on the frontier when the budget ran out
        goal = next((n["id"] for n in nodes if n["success"] or n["k"] >= len(steps)), None)
    best = goal if goal is not None else max(nodes, key=lambda n: (n["k"], n["progress"], -n["t"]))["id"]
    path, n = [], nodes[best]
    while n["parent"] is not None:
        path.append(list(n["via"])); n = nodes[n["parent"]]
    path.reverse()
    restore(kenv, nodes[best]["state"], nodes[best]["ps"])
    final_ok = bool(kenv._check_success())  # diagnostic only: the official check at the end state of the last chunk
    # success = the official check held at some step, as for the plain baseline (rollout's per-step reward). Until
    # 2026-10-30 this also required final_ok, the official check after restoring the end state. That can fail when
    # success held only mid-chunk, or because the restore's extra update_state() sees freshly recomputed contacts and
    # flips the microwave's turned_on (seen in ../20261029_closedloop/preflight.py). 4 searches (SteamInMicrowave s5, s22;
    # PackIdenticalLunches s7, s24) were counted as failures although a committed candidate had succeeded.
    return dict(success=goal is not None, final_check=final_ok, steps=nodes[best]["t"],
                stage=nodes[best]["k"], path=path, backtracked=not nodes[best]["first_choice"],
                n_expansions=len(expansions), n_candidates=n_cands, expansions=expansions,
                imagined_steps=imagined_steps, policy_calls=policy_calls)


def baseline_episode(client, base, kenv, j, a):
    task, tag = j["task"], j["tag"]
    s = load_scene(kenv, j["scene"])
    instr = kenv.get_ep_meta()["lang"]
    r, frames = rollout(client, base, kenv, {}, TASKS[task], s, j["horizon"], "task", instr,
                        seed_fn=policy_seed_fn(tag, j["policy_seed"]))
    write_video(f"{a.out}/videos/{tag}_ps{j['policy_seed']}.mp4", frames, fps=10)
    return dict(success=r["success"], steps=r["steps"], history=r["history"], policy_calls=-(-r["steps"] // 16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1); ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))[a.worker::a.n_workers]
    key = lambda r: (r["tag"], r["condition"], r.get("policy_seed"))
    def rows(p):  # a line cut off by a kill is skipped (its job runs again)
        for l in open(p):
            try:
                yield json.loads(l)
            except json.JSONDecodeError:
                pass
    done = {key(r) for p in glob.glob(f"{a.out}/results_w*.jsonl") for r in rows(p) if "error" not in r}
    client = Client(port=a.port, timeout_ms=600000)
    envs = {}
    os.makedirs(f"{a.out}/videos", exist_ok=True)
    for j in jobs:
        if key(j) in done:
            continue
        t0 = time.time()
        try:
            if j["task"] not in envs:
                for e_ in envs.values():
                    e_.close()
                envs = {j["task"]: gym.make(f"robocasa/{j['task']}", split="target", enable_render=True).unwrapped}
            base = envs[j["task"]]; kenv = base.env
            r = (imagine_episode if j["condition"] == "imagine" else baseline_episode)(client, base, kenv, j, a)
            r.update({kk: j[kk] for kk in ("tag", "task", "condition", "policy_seed", "N", "L") if kk in j})
        except Exception as e:
            r = dict(tag=j["tag"], condition=j["condition"], policy_seed=j.get("policy_seed"), error=repr(e),
                     tb=traceback.format_exc()[-1500:])
        r["wall_s"] = round(time.time() - t0, 1)
        with open(f"{a.out}/results_w{a.worker}.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps({kk: r.get(kk) for kk in ("tag", "condition", "policy_seed", "success", "steps", "imagined_steps", "error", "wall_s")}), flush=True)


if __name__ == "__main__":
    main()
