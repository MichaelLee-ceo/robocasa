"""Question 2: can a chunk boundary where steering helps be told apart WITHOUT simulating? Step 1: per-point labels and
signals.

Points (../20261028_critical/data.py): confirm (32 located critical chunks, 8 candidates x 4 continuations), control (32
random chunks, same protocol) and coarse (768 uniform points, 8 x 2; the replication set). Every point is a state on
the plain policy's main run (snapshot state == main-run state at t, checked), so the main run's saved states, trace and
events give the robot's state and recent past at the point.

Labels (outcomes of the 8 candidates; NOT observable at deployment):
  V        completion rate over all rollouts of the point
  var_b    between-candidate variance of P(stage done), luck removed (confirm_analysis.py for 8 x 4; for 8 x 2 the
           unbiased covariance of the two continuations across candidates, effect_by_event.points)
  cv       cross-validated pick gain. 8 x 4: mean over the 6 ways to pick on 2 continuations and score on the other 2
           (more stable than the single split of confirm_analysis.py). 8 x 2: effect_by_event.cv_gain
  perm_p   (8 x 4 only) permutation test that the candidates differ: variance of the 8 candidates' rates against 4000
           shuffles of the 32 outcomes among candidates
  steerable = perm_p < 0.05 (the candidates differ beyond luck; every such point has var_b >= 0.056, i.e. a perfect
           best-of-8 picker gains >= 22 pts); steer05 = var_b >= 0.05 is the effect-size variant (sensitivity check)

Signals (observable at deployment; the policy's 8 sampled chunks, the robot's proprioception and gripper contacts):
  candidate disagreement, first chunk of each candidate (16 steps x 12 dims, every dim z-scored with its std over all
  coarse candidate chunks of the task; ../20261028_critical/distribution.py):
    var, var_arm, var_grip, var_base, var_mode   mean variance across the 8, all dims or one action group
    bimodal  share of the spread explained by the best split of the 8 into two groups (two distinct options)
    minority size of the smaller group of that split; pc1 share of the variance along the first principal direction
    outlier  largest distance to the centroid / median distance; growth variance of the last 4 steps / first 4
    pos_rel  spread of the summed eef-position command / its mean length (disagreement in direction; disagreement.py)
    grip_end variance across candidates of the gripper command at the chunk's last step (open vs closed after it)
    act_mag  mean length of the candidates' summed eef-position command (how far they intend to move; raw units)
    base_frac, close_frac  share of candidate-steps in base mode / with the gripper commanded closed
  robot state at t (main-run sim state = [time, qpos, qvel]; robot joints are qpos 0-12 in every scene, checked):
    gap       finger opening (m); arm_speed |arm joint velocity| (rad/s); base_speed |base x/y velocity| (m/s)
    arm_moved, base_moved, gap_moved  change of arm joints / base position / finger opening over the last 16 steps
  contacts at t (main-run trace, ../20261025_seen480/analysis): touch (gripper touches any task object or fixture),
    touch16 (any such contact in the last 16 steps), grasp (an object is grasped, check_obj_grasped)
  past events (../20261028_critical/events.jsonl, events at f <= t only): since_event, steps since the last touch / hold /
    lift / at / release / close / open / fx_touch / fx_move of the stage (since the stage start if none)
  stage: k and kind (pnp / fixture)
References (NOT deployable): t_stage (steps since the stage started), t (episode step), to_event (steps to the next
grasp / place / fixture event of the stage that this chunk or a later one causes; 9999 if none).
Writes features.jsonl (one row per point and dataset). usage: python -I features.py [workers]"""
import gzip, itertools, json, multiprocessing, os, sys, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CRIT = os.path.normpath(os.path.join(HERE, "..", "20261028_critical"))
SEEN = os.path.normpath(os.path.join(HERE, "..", "20261025_seen480"))
sys.path.insert(0, CRIT)
import data  # noqa: E402
from distribution import DIM_GROUP, actions, shape  # noqa: E402
from disagreement import measures  # noqa: E402
from effect_by_event import cv_gain  # noqa: E402

DS = ("confirm", "control", "coarse")
KIND = {"KettleBoiling": ["pnp", "fixture"], "SteamInMicrowave": ["pnp", "pnp", "fixture", "fixture"],
        "PackIdenticalLunches": ["pnp"] * 4}  # ../scripts/stage_trace.py STAGE_SPECS
PAST = {"touch", "hold", "lift", "at", "release", "close", "open", "fx_touch", "fx_move"}
NEXT = {"touch", "hold", "close", "at", "release", "open", "fx_touch", "fx_move"}


def labels(y, rng):
    """y: candidates x continuations of done"""
    p, M = y.mean(1), y.shape[1]
    out = dict(V=float(y.mean()), n_cont=M)
    if M == 4:
        out["var_b"] = float(p.var(ddof=1) - np.mean(p * (1 - p)) / (M - 1))
        g = []
        for a in itertools.combinations(range(4), 2):
            b = [i for i in range(4) if i not in a]
            f, s = y[:, list(a)].mean(1), y[:, b].mean(1)
            g.append(s[f == f.max()].mean() - s.mean())
        out["cv"] = float(np.mean(g))
        obs, flat = p.var(), y.ravel()
        null = np.array([rng.permutation(flat).reshape(y.shape).mean(1).var() for _ in range(4000)])
        out["perm_p"] = float((null >= obs - 1e-12).mean())
        out["steerable"] = out["perm_p"] < 0.05
        out["steer05"] = out["var_b"] >= 0.05
    else:  # 8 x 2 (coarse)
        n = len(y); d0, d1 = y[:, 0] - y[:, 0].mean(), y[:, 1] - y[:, 1].mean()
        out["var_b"] = float((d0 * d1).sum() / (n - 1))
        out["cv"] = cv_gain(y)
    return out


def episode(task, tag, pts):
    """features of every point of one main run; pts: [(ds, t, k, row)]"""
    S = np.load(f"{SEEN}/run/{task}/{tag}_steps.npz")["states"]
    tr = json.load(gzip.open(f"{SEEN}/analysis/{task}/{tag}_trace.json.gz"))
    ev = next(r for r in map(json.loads, open(f"{CRIT}/events.jsonl")) if r["tag"] == tag)
    nq = json.loads(str(np.load(data.scenes()[tag], allow_pickle=True)["fp"]))["nq"]
    q, v = S[:, 1:1 + nq], S[:, 1 + nq:]
    gw = np.array(tr["grip_width"])
    contact = np.zeros(len(gw), bool)
    for o in tr["obj"].values():
        contact |= np.array(o["touch"], bool)
    for f in tr["fx_touch"].values():
        contact |= np.array(f, bool)
    grasp = np.zeros(len(gw), bool)
    for o in tr["obj"].values():
        grasp |= np.array(o["grasp"], bool)
    starts = [0] + [h[0] for h in ev["history"]]
    rng = np.random.default_rng(zlib.crc32(tag.encode()))
    out = []
    for ds, t, k, r in pts:
        y = data.outcomes(r)
        chunks = [np.load(f"{data.roll(r)}/{tag}_t{t}_c{i}_m0.npz") for i in range(len(y))]
        A = actions(chunks)  # 8 x 16 x 12 raw
        m = measures(chunks)
        P = A[:, :, 0:3].sum(1)
        t0 = max(t - 16, 0)
        sk = starts[k]
        past = [e["t"] for e in ev["events"] if e["k"] == k and e["type"] in PAST and e["t"] <= t]
        nxt = [e["t"] - t for e in ev["events"] if e["k"] == k and e["type"] in NEXT and e["t"] > t]
        row = dict(task=task, ds=ds, tag=tag, t=t, k=k, kind=KIND[task][k], **labels(y, rng),
                   pos_rel=m["pos_rel"], grip_end=m["grip_end"], act_mag=float(np.linalg.norm(P, axis=1).mean()),
                   base_frac=float(A[:, :, 11].mean()), close_frac=float(A[:, :, 6].mean()),
                   gap=float(gw[t]), arm_speed=float(np.linalg.norm(v[t, 4:11])), base_speed=float(np.linalg.norm(v[t, 0:2])),
                   arm_moved=float(np.linalg.norm(q[t, 4:11] - q[t0, 4:11])), base_moved=float(np.linalg.norm(q[t, 0:2] - q[t0, 0:2])),
                   gap_moved=float(gw[t] - gw[t0]), touch=bool(contact[t]), touch16=bool(contact[t0:t + 1].any()), grasp=bool(grasp[t]),
                   since_event=int(t - max(past)) if past else int(t - sk), t_stage=int(t - sk), to_event=int(min(nxt)) if nxt else 9999)
        out.append((row, A))
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobs = {}
    for task in data.TASKS:
        for ds in DS:
            for (tag, t), r in data.rows(task, ds).items():
                jobs.setdefault((task, tag), []).append((ds, t, r["k"], r))
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = [x for g in pool.starmap(episode, [(task, tag, p) for (task, tag), p in sorted(jobs.items())]) for x in g]
    std = {}
    for task in data.TASKS:  # per task, every action dim's std over all coarse candidate chunks (as distribution.py)
        A = np.concatenate([A for row, A in res if row["task"] == task and row["ds"] == "coarse"])
        std[task] = A.reshape(-1, 12).std(0) + 1e-6
    rows = []
    for row, A in res:
        Z = A / std[row["task"]]
        row.update(shape(Z.reshape(len(Z), -1), np.tile(DIM_GROUP, 16)))
        vz = Z.var(0)
        row["growth"] = float(vz[-4:].mean() / (vz[:4].mean() + 1e-9))
        rows.append(row)
    rows.sort(key=lambda r: (r["task"], r["ds"], r["tag"], r["t"]))
    with open(os.path.join(HERE, "features.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    for task in data.TASKS:
        print(task, {ds: sum(r["task"] == task and r["ds"] == ds for r in rows) for ds in DS})


if __name__ == "__main__":
    main()
