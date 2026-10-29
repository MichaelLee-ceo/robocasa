"""Rollouts vs the main run's own continuation, at the points of a V pattern (CPU replay of the saved states).

For each point t of one (episode, stage): the main run's states t .. t+H and every rollout's states 0 .. H are replayed
(set_state, no rendering) and summarised at steps 0, 4, 8, 16, 32, 64 (when they exist):
  the task objects' height and gripper distance (the object nearest the gripper at t is shown first), gripper contact,
  finger gap, base xy displacement from t and base yaw change, stage-k check, and the rollouts' labels.
A restore artifact shows up as the rollouts departing from what the main run did in the first steps (object drops
while the main run holds it, the base moving another way, fingers moving against the command), at every point of the
pattern. usage: python compare_main.py <task> <tag> <t1,t2,...> [H=64]"""
import copy, json, os, sys
import numpy as np

R = "/home/michaellee/mclee/robocasa"
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", f"{R}/runs/20261028_critical", f"{R}/runs/20261026_imagine",
                f"{R}/runs/20261026_critic_data", os.path.dirname(os.path.abspath(__file__))]
OBJS = {"KettleBoiling": ["obj"], "SteamInMicrowave": ["vegetable", "bowl"], "PackIdenticalLunches": ["vegetable0", "vegetable1", "meat0", "meat1"]}
SHOW = (0, 4, 8, 16, 32, 64)


def probe(env, task):
    d = env.sim.data
    eef = np.array(d.site_xpos[env.robots[0].eef_site_id["right"]])
    grip = env.robots[0].gripper["right"]
    out = {}
    for o in OBJS[task]:
        p = np.array(d.body_xpos[env.obj_body_id[o]])
        out[o] = (round(float(p[2]), 3), round(float(np.linalg.norm(p - eef)), 3), bool(env.check_contact(grip, env.objects[o])))
    bid = env.sim.model.body_name2id("mobilebase0_base")
    m = np.array(d.body_xmat[bid]).reshape(3, 3)
    out["_base"] = (np.array(d.body_xpos[bid][:2]), float(np.degrees(np.arctan2(m[1, 0], m[0, 0]))))
    q = d.qpos
    out["_gap"] = round(1000 * abs(q[11] - q[12]), 1)
    return out


def main():
    import atomic_eval  # noqa: F401
    from atomic_eval import load_scene
    from atomic_steps import TASKS
    from analyze_stages import replay_env
    from collect import replay
    from common import SEEN, coarse, dense, rolls, roll_path, scenes
    from imagine import set_py_state
    from oracle import set_state
    task, tag, ts = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
    H = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    rows = {**coarse(task), **dense(task)}
    env = replay_env(task)
    load_scene(env, scenes()[tag])
    ms = np.load(f"{SEEN}/run/{task}/{tag}_steps.npz")["states"]
    steps = TASKS[task]
    for t in ts:
        r = rows[(tag, t)]
        k = r["k"]
        S0, _ = replay(env, ms, t, steps)
        p0 = probe(env, task)
        objs = sorted(OBJS[task], key=lambda o: p0[o][1])
        b0, y0 = p0["_base"]
        lab = [x for _, _, x in rolls(r)]
        print(f"=== {tag} t{t} S{k} {r['_dir']} hidden={r.get('hidden')} V={np.mean([x['done'] for x in lab]):.2f} "
              f"({sum(x['done'] for x in lab)}/{len(lab)}), steps of done: {sorted(x['steps'] for x in lab if x['done'])}")

        def summary(states, label):
            cells = []
            for s in SHOW:
                if s >= len(states) or s > H:
                    break
                set_state(env, states[s])
                p = probe(env, task)
                b, y = p["_base"]
                o = objs[0]
                cells.append(f"{s}:z{p[o][0]:.2f} d{p[o][1]:.2f}{'c' if p[o][2] else ''} g{p['_gap']:.0f} B{np.linalg.norm(b - b0):.2f}/{y - y0:+.0f}")
            return f"   {label:10s} [{objs[0]}] " + " | ".join(cells)
        set_py_state(env, r["py_state"])
        print(summary(ms[t:t + H + 1], "main"))
        for c, m, x in rolls(r):
            st = np.load(roll_path(r, c, m))["states"]
            print(summary(st, f"c{c}m{m} {'D' if x['done'] else '-'}{x['steps']}"))


if __name__ == "__main__":
    main()
