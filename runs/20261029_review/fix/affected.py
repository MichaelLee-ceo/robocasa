"""Which collected decision points does the settled-v5 restore change? For every row in use, the gripper ramp its own
settle version set at t (v1-v3: the old +-1 rule, v4: the joint-limit +-1 rule; t = 0: +1 or -1 although the truth is 0)
vs the ramp hidden_state.gripper_ramp infers now (t = 0: exactly 0). Also flags points that start in base mode-like
motion (base moved in the last step), whose arm-goal lag no restore can fix. No rendering, no stepping.
usage: python -I affected.py <task> out.jsonl [shard/n_shards]"""
import glob, json, os, sys
import numpy as np

R = "/home/michaellee/mclee/robocasa"
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", f"{R}/runs/20261026_imagine", f"{R}/runs/20261026_critic_data"]
from analyze_stages import replay_env  # noqa: E402
from atomic_eval import load_scene  # noqa: E402
from oracle import set_state  # noqa: E402
import hidden_state  # noqa: E402

D = f"{R}/runs/20261026_critic_data"
MAIN = f"{R}/runs/20261025_seen480/run"
DIRS = {"KettleBoiling": ["run_v2/s0", "run_v2/s1", "run_v2/s2", "run_v2/s3", "run_v2/s4", "run_KettleBoiling_fix",
                          "run_dense_KettleBoiling", "run_confirm_KettleBoiling"],
        "SteamInMicrowave": ["run_SteamInMicrowave_v2", "run_SteamInMicrowave_fix", "run_SteamInMicrowave_fix3",
                             "run_dense_SteamInMicrowave", "run_dense2_SteamInMicrowave", "run_confirm_SteamInMicrowave"],
        "PackIdenticalLunches": ["run_PackIdenticalLunches", "run_PackIdenticalLunches_fix", "run_PackIdenticalLunches_fix3",
                                 "run_dense_PackIdenticalLunches", "run_dense2_PackIdenticalLunches", "run_confirm_PackIdenticalLunches"]}
OPEN_V4 = 0.0785


def rows_of(d):
    out = []
    for p in sorted(glob.glob(f"{D}/{d}/results_w*.jsonl")):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                out.append(r)
    return out


def main():
    task, outp = sys.argv[1], sys.argv[2]
    shard, nshard = map(int, sys.argv[3].split("/")) if len(sys.argv) > 3 else (0, 1)  # e.g. 2/6: every 6th episode from the 3rd
    jobs = {}
    for p in glob.glob(f"{D}/jobs*.json"):
        for j in json.load(open(p)):
            if j.get("task") == task:
                jobs.setdefault((j["tag"], j["t"]), j)
    pts = []
    for d in DIRS[task]:
        for r in rows_of(d):
            pts.append((d, r))
    kenv = replay_env(task)
    kenv = getattr(kenv, "env", kenv)
    out = open(outp, "w")
    by_tag = {}
    for d, r in pts:
        by_tag.setdefault(r["tag"], []).append((d, r))
    for ti, (tag, lst) in enumerate(sorted(by_tag.items())):
        if ti % nshard != shard:
            continue
        job = jobs[(tag, lst[0][1]["t"])]
        load_scene(kenv, job["scene"])
        ep = np.load(f"{MAIN}/{task}/{tag}_steps.npz")["states"]
        rb = kenv.robots[0]
        idx = rb._ref_gripper_joint_pos_indexes["right"]
        bidx = [kenv.sim.model.get_joint_qpos_addr(j) for j in rb.robot_model.joints if "mobilebase" in j or "base" in j][:3]
        for d, r in lst:
            t = r["t"]
            set_state(kenv, ep[t])
            q = kenv.sim.data.qpos[idx]
            gap = abs(q[0] - q[1])
            dgap = gap - abs(ep[t - 1][1 + idx[0]] - ep[t - 1][1 + idx[1]]) if t else 0.0
            ver = r.get("hidden") or "settled"
            if t == 0:
                old = 1.0 if gap < (OPEN_V4 if ver == "settled-v4" else hidden_state.OPEN_OLD) else -1.0
                new = 0.0
            else:
                thr = OPEN_V4 if ver == "settled-v4" else hidden_state.OPEN_OLD
                old = 1.0 if gap < thr and not dgap > 0.0005 else -1.0
                new = hidden_state.gripper_ramp(kenv, rb, "right", ep[t - 1])
            f1, f2, n_open, n_close, fixture = hidden_state.finger_contacts(kenv, rb, "right")
            base_moved = bool(t and np.abs(ep[t][1 + np.array(bidx)] - ep[t - 1][1 + np.array(bidx)]).max() > 1e-4) if bidx else None
            rec = dict(task=task, dir=d, tag=tag, t=t, k=r["k"], N=r["N"], M=r["M"], L=r["L"], hidden=ver,
                       gap_mm=round(1000 * float(gap), 2), dgap_mm=round(1000 * float(dgap), 3), old=old, new=round(float(new), 3),
                       diff_steps=int(round(abs(new - old) / 0.2)), held=bool(f1 & f2), obj=bool(f1 | f2), fixture=bool(fixture),
                       base_moved=base_moved, wall_s=r.get("wall_s"))
            out.write(json.dumps(rec) + "\n")
        out.flush()
        print(tag, len(lst), flush=True)


if __name__ == "__main__":
    main()
