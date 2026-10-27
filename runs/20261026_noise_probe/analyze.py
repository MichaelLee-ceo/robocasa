"""Summarize the noise probe (results_w*.jsonl): per probe point, outcome over the N noise candidates and the spread of
the first action chunk across candidates.
Spread of the first chunk, in the policy's normalized action units:
- pos_disp_spread: mean pairwise distance between candidates' summed end-effector position commands over the chunk,
  divided by the mean length of that summed command (0 = all candidates move the same way, ~1 = as different as the
  motion itself);
- grip_disagree: fraction of the 16 steps where candidates disagree on gripper close/open (some close, some open);
- base_spread: same as pos_disp_spread for the base motion command.
usage: python analyze.py"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rs = [json.loads(l) for p in glob.glob(f"{HERE}/results_w*.jsonl") for l in open(p)]
rs = [r for r in rs if "error" not in r]
pts = sorted({(r["tag"], r["t0"]) for r in rs}, key=lambda x: (x[0].split("_s")[1].split("_")[0].zfill(2)))


def pair_spread(v):
    """mean pairwise distance between rows / mean row norm."""
    n = len(v)
    d = [np.linalg.norm(v[i] - v[k]) for i in range(n) for k in range(i + 1, n)]
    return float(np.mean(d) / max(np.mean(np.linalg.norm(v, axis=1)), 1e-9)) if d else float("nan")


rows = []
for tag, t0 in pts:
    g = [r for r in rs if r["tag"] == tag and r["t0"] == t0]
    fc = [r["first_chunk"] for r in g]
    pos = np.array([np.sum(np.array(f["action.end_effector_position"]), 0) for f in fc])
    base = np.array([np.sum(np.array(f["action.base_motion"])[:, :3], 0) for f in fc])
    grip = np.array([np.array(f["action.gripper_close"])[:, 0] > 0.5 for f in fc])
    dis = float(np.mean([0 < grip[:, s].sum() < len(grip) for s in range(grip.shape[1])]))
    stage0 = sum(any(k == 1 for _, k in r["stage_steps"]) for r in g)
    lifted = sum(r["kettle_lift_m"] >= 0.05 for r in g)
    succ = sum(r["success"] for r in g)
    same_obs = len({r["obs0_fp"] for r in g}) == 1
    rows.append(dict(point=f"{tag.split('_')[1]} t0={t0}", role=g[0]["role"], n=len(g), success=succ, stage0=stage0,
                     lifted=lifted, pos_disp_spread=round(pair_spread(pos), 2), base_spread=round(pair_spread(base), 2),
                     grip_disagree=round(dis, 2), same_obs=same_obs,
                     end_steps=sorted(r["end_step"] for r in g if r["success"])))
print(f"{'point':14s} {'role':8s} {'n':>3s} {'succ':>5s} {'S0':>4s} {'lift':>5s} {'posSpr':>7s} {'baseSpr':>8s} {'gripDis':>8s} same_obs")
for r in rows:
    print(f"{r['point']:14s} {r['role']:8s} {r['n']:3d} {r['success']:5d} {r['stage0']:4d} {r['lifted']:5d} {r['pos_disp_spread']:7.2f} "
          f"{r['base_spread']:8.2f} {r['grip_disagree']:8.2f} {r['same_obs']}  ok@{r['end_steps'][:6]}")
json.dump(rows, open(f"{HERE}/summary.json", "w"), indent=1)
