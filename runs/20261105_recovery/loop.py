"""The recovery loop from a known failure point (2026-11-06): GR00T executes; a supervisor steps in and keeps checking.

Detection is assumed solved: the run starts at a failure point of phase 0 (phase0/points.json: the failed stage and the
chunk where an oracle supervisor would step in), reached by replaying the plain run's recorded actions (exact).
From there, every chunk the supervisor observes the stages (oracle: stages.track; a VLM replaces it later) and keeps a
memory per stage (when it began, which interventions were tried, and, for the reset arm, the state at its start):
- at the start, the failed stage is stuck: the first intervention of the arm's ladder, at once;
- the current stage is done: move on (instruction back to the task sentence, attempts reset);
- an earlier stage is undone for UNDONE_N chunk ends in a row at which the hand is clear of every object (no contact,
  and farther than 0.25 m, the distance the stage checks use for "away"): that stage is current again and stuck. While
  the hand is near an object a "not done" may only mean "the hand is too close" (SteamInMicrowave: the bowl is in the
  microwave, but the hand is near it while pressing start), so those chunk ends do not count (from 2026-11-06, the
  retrace and reset arms; the earlier arms counted every chunk end, which only re-triggers their own interventions);
- the current stage is stuck: no completion within its budget (RETRY_W chunks after an intervention; for later stages
  the 90th percentile of their completions in phase 0): the next intervention of the ladder.
Interventions:
  hold        the stage's sentence (stages.instructions) instead of the task sentence, fresh samples
  back        reverse the last BACK (or 2 x BACK) executed chunks: motion deltas negated, gripper commands reversed in
              time; then fresh samples with the stage's sentence or the task sentence
  resample    fresh samples, task sentence (nothing else changes)
  retrace     the robot plays back the path it actually followed since the stage began (or since its last retrace of
              this stage, if that arrived), joint by joint at the original speed, to where it was then (retrace.py:
              recorded joint states, closed loop; what it carried goes back with it and is put down where it was
              picked up; what fell stays); then fresh samples with the task sentence and the stage's whole stall
              window again. The retrace's time is not counted, as a restart: the env runs with a free horizon and
              GR00T keeps the official horizon on its own time (success = official success at a chunk end within it;
              a success reached while retracing counts and is flagged)
  reset       (not used: a teleport of the robot only, restore_robot; it leaves held objects in the air) the robot set
              back to where it was when the stage began; one hold-still chunk; then as retrace
Arms (ladders):
  control  resample once, then the policy alone: what a new noise draw does from the failure state
  full     hold; back + hold; back + task; far back + hold; far back + task
  hold     hold (kept until the stage is done)
  back     back + task; back + task; far back + task; far back + task
  retrace  retrace, again each time the stage is stuck (the stage as the unit: go back to its start and retry it)
  reset    reset, likewise
After the ladder is used up the last instruction stays until the stage is done (retrace and reset repeat instead). The run ends at the official success
or at the horizon (as the official eval).
Output: loop/<arm>/<point id>_p<pseed>.json (+ .pkl.gz with the chunks); one json line on stdout.
usage: python -I loop.py <point id, e.g. SteamInMicrowave_s6> <arm> <policy seed>   (port from POLICY_PORT)"""
import gzip, json, math, os, pickle, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"), HERE]
import numpy as np  # noqa: E402
import robocasa.utils.object_utils as OU  # noqa: E402
import envkit, events, policy, stages  # noqa: E402
import retrace as rt  # noqa: E402

BACK, RETRY_W, STALL_Q, UNDONE_N = 4, 12, 90, 3
MOTION = ("action.end_effector_position", "action.end_effector_rotation", "action.base_motion")
LADDERS = {  # (kind, chunks to reverse in units of BACK, use the stage's sentence)
    "control": [("resample", 0, False)],
    "full": [("hold", 0, True), ("back", 1, True), ("back", 1, False), ("back", 2, True), ("back", 2, False)],
    "hold": [("hold", 0, True)],
    "back": [("back", 1, False), ("back", 1, False), ("back", 2, False), ("back", 2, False)],
    "reset": [("reset", 0, False)],
    "retrace": [("retrace", 0, False)],
}
REPEAT = {"reset", "retrace"}  # arms whose last rung repeats
RETRACE_MAX = 400  # chunks at most for one retrace (a safety cap: its time is not counted)
FREE_HORIZON = 10 ** 6  # retrace arm: the env's horizon; the official one is enforced on GR00T's own time


def hand_clear(e):
    """the hand touches no object and is farther than 0.25 m from every object"""
    g = e.robots[0].gripper["right"]
    return all(OU.gripper_obj_far(e, o) and not e.check_contact(g, e.objects[o]) for o in e.objects)


def stall_limits():
    """chunks: the STALL_Q percentile of each stage's completion time in phase 0"""
    dur = {}
    for l in open(os.path.join(HERE, "phase0", "plain.jsonl")):
        r = json.loads(l)
        fd = {int(k): v for k, v in r.get("first_done", {}).items()}
        for k in range(r.get("n_stages", 0)):
            if k in fd and (k == 0 or k - 1 in fd):
                dur.setdefault((r["task"], k), []).append((fd[k] - (fd[k - 1] if k else 0)) / 16)
    return {k: math.ceil(np.percentile(v, STALL_Q)) for k, v in dur.items()}


def reverse_chunks(chunks):
    """the given executed chunks undone: steps in reverse order, motion deltas negated, gripper reversed in time"""
    steps = []
    for c in chunks:
        a = c["actions"]
        steps += [{k: np.array(v[i]) for k, v in a.items()} for i in range(c["n"])]
    steps = steps[::-1]
    for s in steps:
        for k in MOTION:
            s[k] = -s[k]
    out = []
    for i in range(0, len(steps), envkit.N_ACTION_STEPS):
        part = steps[i:i + envkit.N_ACTION_STEPS]
        while len(part) < envkit.N_ACTION_STEPS:  # pad a short last chunk by holding still (zero motion, same gripper)
            part.append({k: (np.zeros_like(v) if k in MOTION else v) for k, v in part[-1].items()})
        out.append({k: np.stack([s[k] for s in part]) for k in part[0]})
    return out


ROBOT = ("robot0_", "mobilebase0_", "gripper0_")


def restore_robot(ep, snap):
    """put only the robot back as it was in snap (an envkit Snapshot): its joints (arm, base, torso, fingers), their
    velocities and warm start, the actuator controls, and its Python state (kenv.robots[0]: the controllers, the OSC goal,
    the gripper ramp; this subgraph reaches no fixture and not the env). Objects, fixtures, time and the observation
    clock stay as they are now. The observation is refreshed by the next step."""
    e = ep.kenv
    m, d, s = e.sim.model, e.sim.data, snap.mjdata
    for j in range(m.njnt):
        n = m.joint_id2name(j)
        if n and n.startswith(ROBOT):
            q, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
            d.qpos[q] = s.qpos[q]
            d.qvel[v], d.qacc_warmstart[v], d.qfrc_applied[v] = s.qvel[v], s.qacc_warmstart[v], s.qfrc_applied[v]
    d.ctrl[:] = s.ctrl  # every actuator is the robot's (asserted in check_restore_robot.py)
    node, old = snap.graph.memo[id(e.robots[0])], snap.graph.root
    snap.graph.root = node
    try:
        snap.graph.write_back()
    finally:
        snap.graph.root = old
    warm = d.qacc_warmstart.copy()
    e.sim.forward()  # derived arrays (kinematics, contacts) for the checks; it also overwrites the solver's warm start,
    d.qacc_warmstart[:] = warm  # which is integration state: put it back


def hold_still(last):
    """one chunk that holds the robot where it is: zero motion, the given gripper command and control mode"""
    return {k: np.repeat(np.zeros_like(v[-1:]) if k in MOTION else np.asarray(v)[-1:], envkit.N_ACTION_STEPS, 0)
            for k, v in last.items()}


def run(pid, arm, pseed, port=5581):
    t0 = time.time()
    point = next(p for p in json.load(open(os.path.join(HERE, "phase0", "points.json"))) if p["id"] == pid)
    task, seed = point["task"], point["seed"]
    rec = pickle.load(gzip.open(os.path.join(HERE, "phase0", "records", f"{task}_s{seed}.pkl.gz")))
    limits = stall_limits()
    client = policy.Client(port)
    free = arm == "retrace"  # the retrace's own time is not counted (as a restart): GR00T keeps its official time
    H = envkit.get_task_horizon(task)
    ep = envkit.Episode(task, seed, horizon=FREE_HORIZON if free else None)
    starts = {}  # the supervisor's memory: stage -> where it began (reset: the state and the chunk before it;
    rec_path = rt.Recorder(ep) if arm == "retrace" else None  # retrace: the index of the recorded path)

    def remember(k):
        if arm == "retrace":
            starts[k] = len(rec_path.path) - 1
        else:
            last = ep.chunks[-1]["actions"] if ep.chunks else {q: np.asarray(v)[:1] for q, v in rec["chunks"][0]["actions"].items()}
            starts[k] = (ep.snapshot(), last)

    reached = -1
    for i, c in enumerate(rec["chunks"][:point["chunk"] + 1]):
        while arm in REPEAT and reached < min(rec["tracks"][i]["stage"], point["stage"]):  # tracks[i]: after i chunks
            reached += 1
            remember(reached)
        ep.step(c["actions"])
    tr = stages.track(ep)
    assert not tr["done"][point["stage"]], f"{pid}: stage {point['stage']} is done at the failure point"
    tag = f"loop/{pid}/p{pseed}"
    cur, attempts, instr, stream = point["stage"], 0, None, f"{tag}/start"
    start, budget = len(ep.chunks), 0  # stuck now
    executed = list(range(len(ep.chunks)))  # the plain run's own chunks (reversible)
    log, done_at, kinds = [], {}, LADDERS[arm]
    fixed_at, low = None, 0
    back_to = starts.get(cur)  # retrace: where the next retrace of the current stage goes back to
    free_steps, succ, succ_retrace, t_fail = 0, None, False, ep.t

    def ptime():  # GR00T's own time: the retraces are not counted
        return ep.t - free_steps

    def policy_success(c):  # official: success at a chunk end; a chunk past H ends at H, as the official eval cuts it
        e = min(c["n"], H - (c["t"] - free_steps)) - 1
        return e >= 0 and c["step_success"][e]

    while not ep.done and (succ is None if free else not ep.success) and ptime() < H:
        n = len(ep.chunks)
        s = tr["stage"]
        if s > cur:  # moved on
            log.append(dict(chunk=n, event="stage_done", stage=cur, attempts=attempts))
            if cur == point["stage"] and fixed_at is None:
                fixed_at = round((ptime() - t_fail) / envkit.N_ACTION_STEPS)
            for k in range(cur + 1, s + 1) if arm in REPEAT else ():
                remember(k)
            cur, start, attempts, instr, stream = s, n, 0, None, f"{tag}/s{s}"
            budget, executed, back_to = limits.get((task, cur), 40), [], starts.get(s)
        if arm in REPEAT:
            low = (low + 1 if hand_clear(ep.kenv) else low) if s < cur else 0
        else:
            low = low + 1 if s < cur else 0
        if s < cur and low >= UNDONE_N:  # an earlier stage undone (debounced): current again, and stuck
            log.append(dict(chunk=n, event="undone", stage=s, was=cur))
            cur, start, attempts, budget, low, back_to = s, n, 0, 0, 0, starts.get(s)
        if n - start >= budget:  # stuck: the next intervention
            if attempts >= len(kinds) and arm not in REPEAT:
                log.append(dict(chunk=n, event="ladder_done", stage=cur))
                budget = 10 ** 6
            else:
                kind, back, hold = kinds[min(attempts, len(kinds) - 1)]
                attempts += 1
                entry = dict(chunk=n, event="intervene", stage=cur, attempt=attempts, kind=kind, back=back * BACK,
                             instruction="stage" if hold else "task")
                if back:
                    mine = [ep.chunks[i] for i in executed[-back * BACK:]]
                    for c in reverse_chunks(mine):
                        if ep.done:
                            break
                        ep.step(c)
                    entry["reversed"] = len(mine)
                if kind == "reset":
                    snap, last = starts[cur]
                    h0 = events._eef(ep.kenv)
                    restore_robot(ep, snap)
                    entry["hand_moved_cm"] = round(100 * float(np.linalg.norm(events._eef(ep.kenv) - h0)), 1)
                    ep.step(hold_still(last))
                if kind == "retrace":
                    n0 = len(ep.chunks)
                    res = rt.retrace(ep, rec_path, back_to, RETRACE_MAX)
                    entry.update(res)
                    free_steps += res["steps"]
                    if any(c["step_success"][-1] for c in ep.chunks[n0:]):  # success while retracing
                        succ, succ_retrace = ptime(), True
                        log.append(entry)
                        break
                    if res["arrived"]:
                        back_to = len(rec_path.path) - 1
                instr = stages.instructions(ep)[cur] if hold else None
                stream = f"{tag}/a{cur}.{attempts}"
                start, executed = len(ep.chunks), []
                budget = limits.get((task, cur), 40) if kind in ("reset", "retrace") else RETRY_W
                log.append(entry)
                if ep.done:
                    break
        ep.step(policy.act(client, ep.obs, policy.chunk_seed(stream, 0, len(ep.chunks)), instruction=instr))
        executed.append(len(ep.chunks) - 1)
        if free and succ is None and policy_success(ep.chunks[-1]):
            succ = min(ptime(), H)
        tr = stages.track(ep)
        for k, d in enumerate(tr["done"]):
            if d and k not in done_at:
                done_at[k] = len(ep.chunks)
    success = (succ is not None) if free else bool(ep.success)
    if fixed_at is None and (stages.track(ep)["done"][point["stage"]] or success):
        fixed_at = round((min(ptime(), H) - t_fail) / envkit.N_ACTION_STEPS)
    out = dict(id=pid, task=task, seed=seed, arm=arm, pseed=pseed, stage=point["stage"], kind=point["kind"],
               start_chunk=point["chunk"] + 1, chunks_left=point["chunks_left"], fixed=fixed_at is not None,
               fixed_after=fixed_at, success=success, success_step=succ if free else ep.success_step,
               success_in_retrace=succ_retrace, free_steps=free_steps,
               chunks=len(ep.chunks), final_stage=tr["stage"], stages_done={str(k): v for k, v in sorted(done_at.items())},
               interventions=sum(1 for x in log if x["event"] == "intervene"), log=log,
               seconds=round(time.time() - t0, 1))
    d = os.path.join(HERE, "loop", arm)
    os.makedirs(d, exist_ok=True)
    with gzip.open(os.path.join(d, f"{pid}_p{pseed}.pkl.gz"), "wb") as f:
        pickle.dump(dict(out=out, start_chunk=point["chunk"] + 1, chunks=ep.chunks), f)
    json.dump(out, open(os.path.join(d, f"{pid}_p{pseed}.json"), "w"), indent=1)
    ep.close()
    return out


if __name__ == "__main__":
    pid, arm, pseed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    claim = os.path.join(HERE, "loop", arm, f"{pid}_p{pseed}.claim")  # several runners on one queue: first one wins
    os.makedirs(os.path.dirname(claim), exist_ok=True)
    try:
        os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        print(json.dumps(dict(id=pid, arm=arm, pseed=pseed, skipped="claimed by another runner")), flush=True)
        sys.exit(0)
    try:
        r = run(pid, arm, pseed, int(os.environ.get("POLICY_PORT", 5581)))
        r["log"] = r["log"][:30]
    except Exception as x:
        import traceback
        r = dict(id=pid, arm=arm, pseed=pseed, error=f"{type(x).__name__}: {x}", trace=traceback.format_exc()[-800:])
    print(json.dumps(r, default=str), flush=True)
