"""Steer v2, step 2: rank candidate instructions without the simulator, from GR00T's own predicted chunks (as PALO
selects language by the frozen policy's predicted actions).

At a failure point: for every candidate instruction, sample N chunks (the same N seeds for every instruction) and
compute each chunk's commanded end-effector path from the actions and the observation alone (cmd_path(): the OSC arm's
critically damped response and the calibrated base model, from ../20261030_vlm_critic/commanded.py at commit 84e894a;
end error 1.7-5 cm median there). Score = how much closer the commanded path brings the hand to the target, using the
target's true position (oracle pointing: a VLM would mark it).
Output: steer/v2/rank/<id>.json  [{k, family, text, toward_end, toward_best, close_frac, effect}]
usage: python -I steer_rank.py <id> [n=8]   (servers: composite 5581/5582)"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"), HERE]
import gzip, pickle  # noqa: E402
import numpy as np  # noqa: E402
import envkit, events, policy  # noqa: E402
import recover  # noqa: E402
from steer_probe import target_pos  # noqa: E402

STEP_POS, STEP_ROT, KP, SUB, DT, BASE_XY, BASE_YAW = 0.05, 0.5, 150.0, 25, 0.002, 0.028, 0.058


def quat2mat(q):  # x y z w
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def track(rel, w=np.sqrt(KP)):
    x, v, out = np.zeros(rel.shape[1]), np.zeros(rel.shape[1]), [np.zeros(rel.shape[1])]
    for r in rel:
        goal = x + r
        for _ in range(SUB):
            v = v + (w * w * (goal - x) - 2 * w * v) * DT
            x = x + v * DT
        out.append(x.copy())
    return np.array(out)


def obs_get(obs, suffix):
    k = next(k for k in obs if k.endswith(suffix))
    return np.asarray(obs[k], float).reshape(-1)


def cmd_path(obs, chunk):
    """world positions (n + 1, 3) of the commanded end-effector path"""
    bp, Rb = obs_get(obs, "base_position")[:3], quat2mat(obs_get(obs, "base_rotation")[:4])
    p = obs_get(obs, "end_effector_position_relative")[:3]
    a_pos = np.clip(np.asarray(chunk["action.end_effector_position"], float), -1, 1)
    base_mode = np.asarray(chunk["action.control_mode"], float).reshape(-1) > 0.5
    a_base = np.asarray(chunk["action.base_motion"], float)
    bps, Rbs = [bp.copy()], [Rb.copy()]
    for i in range(len(base_mode)):
        b, R = bps[-1], Rbs[-1]
        if base_mode[i]:
            b = b + R @ np.array([BASE_XY * a_base[i, 0], BASE_XY * a_base[i, 1], 0.0])
            R = R @ rotz(BASE_YAW * a_base[i, 2])
        bps.append(b); Rbs.append(R)
    P = p + track(STEP_POS * a_pos * (~base_mode)[:, None])
    return np.array(bps) + np.einsum("nij,nj->ni", np.array(Rbs), P)


if __name__ == "__main__":
    iid = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    inst = recover.INSTANCES[iid]
    cands = json.load(open(os.path.join(HERE, "steer", "v2", "instructions.json")))[iid]
    rec = pickle.load(gzip.open(os.path.join(recover.REC, f"{inst['task']}_s{inst['seed']}.pkl.gz")))
    ep = envkit.Episode(inst["task"], inst["seed"])
    for c in rec["chunks"][:inst["chunk"] + 1]:
        ep.step(c["actions"])
    e = ep.kenv
    tgt, hand = target_pos(e, inst["task"]), events._eef(e)
    d0 = float(np.linalg.norm(hand - tgt))
    ci = len(ep.chunks)
    chunks, out = {}, []
    for k, cand in enumerate(cands):
        client = policy.Client(5581 + k % 2)
        chunks[k] = [policy.act(client, ep.obs, policy.chunk_seed(f"steer2/{iid}", p, ci), instruction=cand["text"])
                     for p in range(n)]
    start_err = float(np.linalg.norm(cmd_path(ep.obs, chunks[0][0])[0] - hand))
    keys = ("action.end_effector_position", "action.end_effector_rotation", "action.gripper_close",
            "action.base_motion", "action.control_mode")
    vec = lambda c: np.concatenate([np.asarray(c[q], float).ravel() for q in keys])  # noqa: E731
    noise = np.mean([np.linalg.norm(vec(chunks[0][a]) - vec(chunks[0][b])) for a in range(n) for b in range(a + 1, n)])
    for k, cand in enumerate(cands):
        paths = [cmd_path(ep.obs, c) for c in chunks[k]]
        dist = np.array([np.linalg.norm(P - tgt, axis=1) for P in paths])  # (n, 17)
        out.append(dict(k=k, family=cand["family"], text=cand["text"],
                        toward_end=round(float(np.mean(d0 - dist[:, -1])), 4),
                        toward_best=round(float(np.mean(d0 - dist.min(1))), 4),
                        close_frac=round(float(np.mean([np.mean(np.asarray(c["action.gripper_close"]) > 0.5) for c in chunks[k]])), 3),
                        effect=round(float(np.mean([np.linalg.norm(vec(chunks[k][p]) - vec(chunks[0][p])) for p in range(n)]) / noise), 2)))
    os.makedirs(os.path.join(HERE, "steer", "v2", "rank"), exist_ok=True)
    json.dump(dict(id=iid, d0=round(d0, 3), start_err=round(start_err, 4), n=n, rows=out),
              open(os.path.join(HERE, "steer", "v2", "rank", f"{iid}.json"), "w"), indent=1)
    ep.close()
    print(iid, "d0", round(d0, 3), "start_err", round(start_err, 4))
    for r in sorted(out, key=lambda r: -r["toward_end"]):
        print(f"  {r['family']:4s} toward {r['toward_end']:+.3f} best {r['toward_best']:+.3f} close {r['close_frac']:.2f} "
              f"effect {r['effect']:.2f}  {r['text'][:90]}")
