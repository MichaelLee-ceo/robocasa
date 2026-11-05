"""On the plain policy's main runs (20261025_seen480, seed 0): at every chunk boundary t>0, classify the finger contacts
as gripper_ramp does and flag 'split holds' (each finger on a different free object, none on both), where the rule falls
to the servo target. usage: python -I main_split.py <task> <out.jsonl> <workers>"""
import json, multiprocessing, sys, glob
from pathlib import Path
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")


def work(job):
    task, s = job
    import robocasa  # noqa
    import robocasa.scripts.dataset_scripts.add_subtask_labels_to_lerobot as L
    from demo_utils import ds_dir
    from oracle import reset_to, set_state
    import hidden_state as H
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"),
                         has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, use_object_obs=True,
                         obj_instance_split="target", layout_and_style_ids=list(zip(range(1, 11), range(1, 11))), translucent_robot=False)
    sc = glob.glob(f"{R}/runs/scenes_seen*_v2*/{task}_s{s}.npz")[0]
    z = np.load(sc, allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"])
    ep = np.load(f"{R}/runs/20261025_seen480/run/{task}/{task}_s{s}_single_steps.npz")["states"]
    m = env.sim.model
    r = env.robots[0]
    out = []
    for t in range(16, len(ep), 16):
        set_state(env, ep[t])
        f1, f2, n_open, n_close, fixture = H.finger_contacts(env, r, "right")
        c = float(H.gripper_ramp(env, r, "right", ep[t - 1]))
        idx = r._ref_gripper_joint_pos_indexes["right"]
        q = env.sim.data.qpos[idx]; gap = abs(q[0] - q[1])
        nm = lambda S: sorted(m.body_id2name(b) for b in S)
        split = bool(f1 and f2 and not (f1 & f2))
        # does the gap stay put over the next 16 steps of the main run (a held stack stays at its width)?
        idxa = 1 + np.asarray(idx)
        fut = [abs(ep[min(t + k, len(ep) - 1)][idxa[0]] - ep[min(t + k, len(ep) - 1)][idxa[1]]) for k in (4, 8, 16)]
        out.append(dict(task=task, s=s, t=t, rule=c, gap_mm=round(1000 * gap, 2), f1=nm(f1), f2=nm(f2), fixture=bool(fixture),
                        held=bool(f1 & f2), split=split, fut_gap_mm=[round(1000 * x, 1) for x in fut]))
    env.close()
    return out


if __name__ == "__main__":
    task, outp, w = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seeds = sorted(int(p.split(f"{task}_s")[-1].split("_")[0]) for p in glob.glob(f"{R}/runs/20261025_seen480/run/{task}/{task}_s*_single_steps.npz"))
    with multiprocessing.get_context("spawn").Pool(w) as pool, open(outp, "w") as fo:
        for res in pool.imap_unordered(work, [(task, s) for s in seeds]):
            for row in res:
                fo.write(json.dumps(row) + "\n")
            fo.flush()
            print(res[0]["s"] if res else "?", len(res), sum(r["split"] for r in res), flush=True)
