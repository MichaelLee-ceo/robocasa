"""VLM state-understanding pilot: query packets from recorded envkit runs, with the simulator's ground truth kept apart.

For each episode (a policy-test record), replay it exactly and, every STRIDE chunks, make one query: the last 4 chunk
ends 2 chunks apart, from the 3 cameras (vlm.Query). Packets go to pilot_vlm/packets/<qid>/ (qid is a hash, so a
packet reveals neither the episode nor its time order); the ground truth goes to pilot_vlm/truth.jsonl: the stage flags
(stages.track), the oracle events (events.Events) in the window, and the raw values (target grasped, drawer opening,
toaster lever).
usage: python -I make_vlm_pilot.py <task> <seed> [max_chunk]   (GPU rendering: CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID)"""
import gzip, hashlib, json, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"))
import envkit, events, stages  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # the repository root
from coordinator.vlm import pilot as vlm  # noqa: E402  (was envkit/vlm.py)
import robocasa.utils.object_utils as OU  # noqa: E402

STRIDE, SPAN, STEP = 8, 4, 2
OUT = os.path.join(HERE, "pilot_vlm")


def raw(e, task, stage):
    c = events.TARGETS.get(task, {}).get(stage)
    if not c:
        return {}
    if c[0] == "pick":
        return {"grasped": bool(OU.check_obj_grasped(e, c[1])), "target": c[1]}
    tg = c[1](e)[0]
    return {"fixture_value": round(events._target_value(e, tg), 3), "fixture": tg["attr"]}


def main(task, seed, max_chunk=80):
    rec = pickle.load(gzip.open(os.path.join(HERE, "..", "..", "envkit", "tests", "results", "policy", f"{task}_s{seed}.pkl.gz")))
    ep = envkit.Episode(task, seed)
    ev = events.Events(ep)
    instr = ep.instruction
    frames, truth, queries = {}, [], []
    for i, c in enumerate(rec["chunks"][:max_chunk]):
        ep.step(c["actions"])
        ev.update(ep)
        frames[i] = [(ep.t / 20.0, i, cam, ep.obs[cam][0]) for cam in vlm.CAMERAS]
        if i >= (SPAN - 1) * STEP and (i + 1) % STRIDE == 0:
            idx = [i - k * STEP for k in reversed(range(SPAN))]
            qid = hashlib.sha1(f"{task}|{seed}|{i}".encode()).hexdigest()[:10]
            # stage sentences are made now: some name the scene's current state (e.g. the burner under the kettle)
            q = vlm.Query(qid, instr, stages.instructions(ep), [f for j in idx for f in frames[j]])
            tr = stages.track(ep)
            truth.append(dict(qid=qid, task=task, seed=seed, chunk=i, t=ep.t, stages_done=tr["done"], stage=tr["stage"],
                              official=tr["official"],
                              events=[x for x in ev.events if idx[0] < x["chunk"] + 1 <= i + 1],
                              raw=raw(ep.kenv, task, tr["stage"])))
            queries.append(q)
    ep.close()
    vlm.PacketVLM(os.path.join(OUT, "packets")).write(queries)
    with open(os.path.join(OUT, "truth.jsonl"), "a") as f:
        for t in truth:
            f.write(json.dumps(t) + "\n")
    print(json.dumps(dict(task=task, seed=seed, queries=len(queries))))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *(int(a) for a in sys.argv[3:4]))
