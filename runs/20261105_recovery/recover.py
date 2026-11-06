"""Recovery from known failure points (2026-11-05): once a failure is identified, does a recovery fix the stage?

Each instance is a recorded envkit policy-test run (../../envkit/tests/results/policy) and the chunk where its failure is
visible. A trial replays the run exactly up to that chunk, applies one recovery, and runs up to WINDOW chunks; it
succeeds when the failing stage is done (stages.track, the simulator's check: here only the recovery is evaluated, not
the detection). No restore and no lookahead are used inside the trial.

Recoveries (the GR00T call per chunk):
  continue     composite model, original instruction (control)
  resample     composite model, original instruction, 8 samples per chunk, keep the medoid (MBR on the commanded
               end-effector path and gripper)
  stage_instr  composite model, the stage's sentence (stages.instructions)
  atomic       atomic_seen model, the atomic task's sentence (only for stages whose skill it was trained on)
  backoff      the last 3 executed chunks reversed with the gripper open, then continue
  restart      a fresh attempt from the episode start in the same scene (no replay; stage 0 only; up to RESTART_WINDOW
               chunks): the ceiling of any "go back and try again" recovery, the policy's own success in this scene
Output: one json line per trial (recover/results.jsonl); the trial's actions in recover/records/ (replayable).
usage: python -I recover.py <instance id> <recovery> <policy seed>   (servers: composite 5581/5582, atomic 5591)"""
import gzip, json, os, pickle, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"))
import numpy as np  # noqa: E402
import envkit, policy, stages  # noqa: E402

WINDOW, MBR_N, BACKOFF_K, RESTART_WINDOW = 40, 8, 3, 60
REC = os.path.join(HERE, "..", "..", "envkit", "tests", "results", "policy")
OUT = os.path.join(HERE, "recover")
INSTANCES = {  # chunk = the last executed chunk when the failure is visible (events.py, or by inspection)
    "DS1_short": dict(task="DeliverStraw", seed=1, chunk=23, stage=0, kind="drawer pulled to 0.39 and let go"),
    "DS2_wrong": dict(task="DeliverStraw", seed=2, chunk=12, stage=0, kind="pulled another drawer"),
    "DS0_wander": dict(task="DeliverStraw", seed=0, chunk=15, stage=0, kind="never reaches the drawer"),
    "GTB0_short": dict(task="GetToastedBread", seed=0, chunk=15, stage=0, kind="lever pressed to 0.75 and let go"),
    "GTB1_short": dict(task="GetToastedBread", seed=1, chunk=63, stage=0, kind="lever pressed to 0.27"),
    "GTB2_short": dict(task="GetToastedBread", seed=2, chunk=59, stage=0, kind="lever pressed to 0.50 and let go"),
    "KB2_air": dict(task="KettleBoiling", seed=2, chunk=8, stage=0, kind="grasps at air beside the kettle"),
    "SUCS0_drop": dict(task="SetUpCuttingStation", seed=0, chunk=48, stage=1, kind="steak dropped"),
}
ATOMIC_SKILLS = {"OpenDrawer", "PickPlaceCounterToStove", "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter",
                 "PickPlaceToasterToCounter", "TurnOnMicrowave", "TurnOnSinkFaucet", "TurnOffStove", "CoffeeSetupMug",
                 "OpenCabinet", "PickPlaceCounterToCabinet", "TurnOnElectricKettle"}  # atomic_seen tasks that are our stages


def drawer_side(e):
    """which side of the robot the drawer is on, in the mobile base frame (the atomic template names it)"""
    m = e.sim.model
    bid = m.body_name2id("mobilebase0_base")
    p, R = e.sim.data.xpos[bid], e.sim.data.xmat[bid].reshape(3, 3)
    d = e.sim.data.xpos[m.jnt_bodyid[m.joint_name2id(f"{e.drawer.name}_slidejoint")]]
    lateral = float((R.T @ (d - p))[1])
    return ("left" if lateral > 0 else "right"), round(lateral, 3)


def atomic_sentence(ep, stage):
    skill = stages.STAGES[ep.task][stage]["skill"]
    if skill not in ATOMIC_SKILLS:
        return None, {}
    if skill == "OpenDrawer":
        side, lateral = drawer_side(ep.kenv)
        return f"Open the {side} drawer.", {"drawer_lateral": lateral}
    return stages.instructions(ep)[stage], {}


def mbr(client, obs, seeds, instruction):
    """the medoid of several sampled chunks (smallest summed L2 distance on the commanded path and gripper)"""
    cands = [policy.act(client, obs, s, instruction=instruction) for s in seeds]
    vec = [np.concatenate([c["action.end_effector_position"].ravel(), c["action.end_effector_rotation"].ravel(),
                           c["action.gripper_close"].ravel()]) for c in cands]
    cost = [sum(float(np.linalg.norm(v - w)) for w in vec) for v in vec]
    return cands[int(np.argmin(cost))]


def backoff_chunks(ep, k=BACKOFF_K):
    """the last k executed chunks reversed in time, with the motion deltas negated and the gripper open"""
    steps = []
    for c in ep.chunks[-k:]:
        a = c["actions"]
        steps += [{key: np.array(v[i]) for key, v in a.items()} for i in range(c["n"])]
    steps = steps[::-1]
    for s in steps:
        for key in ("action.end_effector_position", "action.end_effector_rotation", "action.base_motion"):
            s[key] = -s[key]
        s["action.gripper_close"] = np.zeros_like(s["action.gripper_close"])
    out = []
    for i in range(0, len(steps), envkit.N_ACTION_STEPS):
        part = steps[i:i + envkit.N_ACTION_STEPS]
        out.append({key: np.stack([s[key] for s in part]) for key in part[0]})
    return out


def trial(iid, arm, pseed):
    t0 = time.time()
    inst = INSTANCES[iid]
    rec = pickle.load(gzip.open(os.path.join(REC, f"{inst['task']}_s{inst['seed']}.pkl.gz")))
    ep = envkit.Episode(inst["task"], inst["seed"])
    s = inst["stage"]
    if arm == "restart":
        assert s == 0, "restart is defined for stage 0 only"
    else:
        for c in rec["chunks"][:inst["chunk"] + 1]:
            ep.step(c["actions"])
    tr0 = stages.track(ep)
    assert not tr0["done"][s], f"{iid}: stage {s} is already done at the failure point"
    comp = policy.Client(5581 + pseed % 2)
    instr, client, extra = None, comp, {}
    if arm == "stage_instr":
        instr = stages.instructions(ep)[s]
    elif arm == "atomic":
        instr, extra = atomic_sentence(ep, s)
        if instr is None:
            ep.close()
            return dict(id=iid, arm=arm, pseed=pseed, skipped="no atomic_seen task for this stage")
        client = policy.Client(5591)
    if arm == "backoff":
        for ch in backoff_chunks(ep):
            if not ep.done:
                ep.step(ch)
    tag, calls, done_at = f"recover/{iid}", 0, None
    for i in range(RESTART_WINDOW if arm == "restart" else WINDOW):
        if ep.done:
            break
        ci = len(ep.chunks)
        if arm == "resample":
            chunk = mbr(client, ep.obs, [policy.chunk_seed(tag, pseed * 1000 + k, ci) for k in range(MBR_N)], instr)
            calls += MBR_N
        else:
            chunk = policy.act(client, ep.obs, policy.chunk_seed(tag, pseed, ci), instruction=instr)
            calls += 1
        ep.step(chunk)
        if stages.track(ep)["done"][s]:
            done_at = i + 1
            break
    os.makedirs(os.path.join(OUT, "records"), exist_ok=True)
    with gzip.open(os.path.join(OUT, "records", f"{iid}_{arm}_p{pseed}.pkl.gz"), "wb") as f:
        pickle.dump(dict(instance=inst, arm=arm, pseed=pseed, instruction=instr, start_chunk=0 if arm == "restart" else inst["chunk"] + 1,
                         chunks=ep.chunks), f)
    out = dict(id=iid, task=inst["task"], kind=inst["kind"], arm=arm, pseed=pseed, instruction=instr,
               success=done_at is not None, chunks_to_success=done_at, policy_calls=calls,
               seconds=round(time.time() - t0, 1), **extra)
    ep.close()
    return out


if __name__ == "__main__":
    iid, arm, pseed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    try:
        r = trial(iid, arm, pseed)
    except Exception as x:
        import traceback
        r = dict(id=iid, arm=arm, pseed=pseed, error=f"{type(x).__name__}: {x}", trace=traceback.format_exc()[-800:])
    print(json.dumps(r), flush=True)
