"""Ground-truth check of hidden_state.gripper_ramp on official demos (CPU): the true PandaGripper ramp after f steps is
clip(sum 0.2*sign(gripper action)) from 0 at reset; compare with gripper_ramp(state_f, state_{f-1}).
usage: python -I demo_ramp.py <out.jsonl> <n_demos> <stride> task1,task2,..."""
import glob, json, multiprocessing, os, sys, time
from pathlib import Path
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")


def branch(kenv, r, prev):
    import hidden_state as H
    m, d = kenv.sim.model, kenv.sim.data
    idx = r._ref_gripper_joint_pos_indexes["right"]
    q = d.qpos[idx]; gap = abs(q[0] - q[1])
    dgap = 0.0
    if prev is not None:
        qp = prev[1 + np.asarray(idx)]; dgap = gap - abs(qp[0] - qp[1])
    f1, f2, n_open, n_close, fixture = H.finger_contacts(kenv, r, "right")
    if f1 & f2:
        b = "held"
    elif fixture and (f1 | f2):
        b = "obj+fix"
    elif fixture and abs(dgap) <= 0.0005:
        b = "fix_still_side" if (0.005 <= gap < H.OPEN_OLD and n_open != n_close) else "fix_still_old"
    elif fixture:
        b = "fix_moving_servo"
    elif f1 | f2:
        b = "obj1_servo"
    else:
        b = "free_servo"
    # names of touched non-robot root bodies
    names = set()
    for i in range(d.ncon):
        c = d.contact[i]
        for ga, gb in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            ba, bb = m.geom_bodyid[ga], m.geom_bodyid[gb]
            na = m.body_id2name(ba) or ""
            if "finger" in na or "pad" in na:
                nb = m.body_id2name(m.body_rootid[bb]) or ""
                if not nb.startswith(("robot", "gripper", "mobilebase", "mount")):
                    names.add(nb)
    return b, gap, dgap, sorted(names)


def work(job):
    task, n_demos, stride = job
    import robocasa  # noqa
    import robocasa.scripts.dataset_scripts.add_subtask_labels_to_lerobot as L
    from demo_utils import load_demo, ds_dir
    import pandas as pd
    from oracle import reset_to, set_state
    import hidden_state as H
    lerobot = Path(ds_dir(task))
    env = L.build_env(json.loads((lerobot / "extras" / "dataset_meta.json").read_text())["env_args"])
    out = []
    for ep in range(n_demos):
        d = load_demo(task, ep)
        df = pd.read_parquet(f"{lerobot}/data/chunk-000/episode_{ep:06d}.parquet")
        A = np.stack(df["action"].values)[:, 11]
        S = d["states"]
        if len(S) != len(A):
            out.append(dict(task=task, ep=ep, error=f"len states {len(S)} != actions {len(A)}")); continue
        c = [0.0]
        for a in A[:-1]:
            c.append(float(np.clip(c[-1] + 0.2 * np.sign(a), -1, 1)))
        reset_to(env, d["xml"], d["ep_meta"], S[0])
        r = env.robots[0]
        for f in range(1, len(S), stride):
            set_state(env, S[f])
            g = float(H.gripper_ramp(env, r, "right", S[f - 1]))
            b, gap, dgap, names = branch(env, r, S[f - 1])
            out.append(dict(task=task, ep=ep, f=f, true=round(c[f], 3), true_next=round(c[min(f + 1, len(c) - 1)], 3),
                            true_prev=round(c[f - 1], 3), rule=round(g, 3), branch=b,
                            gap_mm=round(1000 * gap, 2), dgap_mm=round(1000 * dgap, 3), touch=names))
    env.close()
    return out


if __name__ == "__main__":
    outp, nd, stride, tasks = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(",")
    t0 = time.time()
    with multiprocessing.get_context("spawn").Pool(len(tasks)) as pool, open(outp, "w") as fo:
        for res in pool.imap_unordered(work, [(t, nd, stride) for t in tasks]):
            for row in res:
                fo.write(json.dumps(row) + "\n")
            print(res[0]["task"] if res else "?", len(res), round(time.time() - t0), flush=True)
