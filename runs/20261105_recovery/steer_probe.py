"""Steerability probe: does the composite GR00T act differently when the instruction changes, at a failure point?

For one failure point of recover.py: replay to the failure chunk, snapshot, then for every candidate instruction in
steer/instructions.json and every policy seed, restore and run CHUNKS chunks with that instruction (composite model).
Per chunk end it records the hand-to-target distance, the target's state (drawer opening, lever, grasp), the base
displacement and whether the stage is done. The first chunk's actions are kept to compare instructions directly.
Output: steer/probe/<id>.jsonl (one line per instruction x seed) and steer/probe/<id>_actions.pkl.gz.
usage: python -I steer_probe.py <failure id> [seeds=4] [chunks=8]   (servers: composite 5581/5582)
env: STEER_DIR (default steer: reads <dir>/instructions.json, writes <dir>/probe/), STEER_KS (comma list: only these
candidates, written to <id>_k<k>.jsonl / _actions_k<k>.pkl.gz so that several processes can share a failure point)"""
import gzip, json, os, pickle, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"), HERE]
import numpy as np  # noqa: E402
import robocasa.utils.object_utils as OU  # noqa: E402
import envkit, events, policy, stages  # noqa: E402
import recover  # noqa: E402

DIR = os.path.join(HERE, os.environ.get("STEER_DIR", "steer"))
OUT = os.path.join(DIR, "probe")
PICK = {"KettleBoiling": "obj", "SetUpCuttingStation": "meat"}


def target_pos(e, task):
    m, d = e.sim.model, e.sim.data
    if task == "DeliverStraw":
        return np.array(d.xpos[m.jnt_bodyid[m.joint_name2id(f"{e.drawer.name}_slidejoint")]])
    if task == "GetToastedBread":
        tg = events._toaster_target(e)[0]
        return np.array(d.xpos[m.jnt_bodyid[m.joint_name2id(tg["joint"])]])
    return np.array(d.xpos[m.body_name2id(e.objects[PICK[task]].root_body)])


def measure(ep, s, base0):
    e, task = ep.kenv, ep.task
    h = events._eef(e)
    bid = e.sim.model.body_name2id("mobilebase0_base")
    out = dict(dist=round(float(np.linalg.norm(h - target_pos(e, task))), 3),
               base_moved=round(float(np.linalg.norm(e.sim.data.xpos[bid][:2] - base0)), 3),
               done=bool(stages.track(ep)["done"][s]))
    if task == "DeliverStraw":
        out["value"] = round(float(e.drawer.get_door_state(env=e)["door"]), 3)
    elif task == "GetToastedBread":
        out["value"] = round(float(events._target_value(e, events._toaster_target(e)[0])), 3)
    else:
        name = PICK[task]
        out["value"] = float(bool(OU.check_obj_grasped(e, name)))  # 1 = grasped
        out["obj_z"] = round(float(e.sim.data.xpos[e.sim.model.body_name2id(e.objects[name].root_body)][2]), 3)
    return out


if __name__ == "__main__":
    iid = sys.argv[1]
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n_chunks = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    inst = recover.INSTANCES[iid]
    cands = json.load(open(os.path.join(DIR, "instructions.json")))[iid]
    ks = [int(x) for x in os.environ["STEER_KS"].split(",")] if os.environ.get("STEER_KS") else list(range(len(cands)))
    tag = f"_k{'-'.join(map(str, ks))}" if os.environ.get("STEER_KS") else ""
    rec = pickle.load(gzip.open(os.path.join(recover.REC, f"{inst['task']}_s{inst['seed']}.pkl.gz")))
    ep = envkit.Episode(inst["task"], inst["seed"])
    for c in rec["chunks"][:inst["chunk"] + 1]:
        ep.step(c["actions"])
    s = inst["stage"]
    bid = ep.kenv.sim.model.body_name2id("mobilebase0_base")
    base0 = np.array(ep.kenv.sim.data.xpos[bid][:2])
    m0 = measure(ep, s, base0)
    snap = ep.snapshot()
    os.makedirs(OUT, exist_ok=True)
    first = {}
    with open(os.path.join(OUT, f"{iid}{tag}.jsonl"), "w") as f:
        for k in ks:
            cand = cands[k]
            for p in range(n_seeds):
                t0 = time.time()
                ep.restore(snap)
                client = policy.Client(5581 + (k + p) % 2)
                rows = []
                for i in range(n_chunks):
                    if ep.done:
                        break
                    ci = len(ep.chunks)
                    chunk = policy.act(client, ep.obs, policy.chunk_seed(f"steer/{iid}", p, ci), instruction=cand["text"])
                    if i == 0:
                        first[(k, p)] = chunk
                    ep.step(chunk)
                    rows.append(measure(ep, s, base0))
                f.write(json.dumps(dict(id=iid, k=k, family=cand["family"], text=cand["text"], pseed=p, start=m0,
                                        chunks=rows, seconds=round(time.time() - t0, 1))) + "\n")
                f.flush()
    with gzip.open(os.path.join(OUT, f"{iid}_actions{tag}.pkl.gz"), "wb") as g:
        pickle.dump(first, g)
    ep.close()
    print("done", iid)
