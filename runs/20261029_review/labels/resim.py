"""Re-simulate a collected rollout from its saved actions, without rendering, with the same restore as
collect.py:decision_point (load_scene, replay -> S, settle at t with the state at t-1, capture, set_py_state(row),
set_state, restore), stepping the robosuite env directly with the action conversion of RoboCasaGymEnv.step. The stage
checks then run on the env exactly as online (after env.step, i.e. on the kinematics/contacts MuJoCo computed before
the last substep's integration), and every state is compared with the saved one.
Use: settle a label mismatch of label_check.py (state replay) that may come from that one-substep difference.
usage: python resim.py <dir> <tag> <t> <c> <m> [version, default: the row's hidden tag]"""
import copy, json, os, sys
import numpy as np

R = "/home/michaellee/mclee/robocasa"
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", f"{R}/runs/20261028_critical", f"{R}/runs/20261026_imagine",
                f"{R}/runs/20261026_critic_data", os.path.dirname(os.path.abspath(__file__))]


def env_action(kenv, act):
    """RoboCasaGymEnv.step's conversion of one step of the policy's action dict to the env action vector."""
    from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
    from robosuite.controllers.composite.composite_controller import HybridMobileBase
    a = PandaOmronKeyConverter.unmap_action(dict(act))
    out = []
    for robot in kenv.robots:
        cc = robot.composite_controller
        pf = robot.robot_model.naming_prefix
        v = np.zeros(cc.action_limits[0].shape)
        for part_name in cc.part_controllers:
            s, e = cc._action_split_indexes[part_name]
            v[s:e] = a.pop(f"{pf}{part_name}")
        if isinstance(cc, HybridMobileBase):
            v[-1] = a.pop(f"{pf}base_mode")
        out.append(v)
    assert not a, a
    return np.concatenate(out)


def resim(env, task, scene, d, tag, t, c, m, row, version=None):
    """version: the settle of the row's code version (row['hidden']): 'settled-v4' = current; 'settled-v3' = old gripper
    rule; 'settled-basefix' = also the base controller's start pose overwritten; 'settled' = also the arm goal in world
    coordinates (as repro_drop.py emulates them)."""
    version = version or row.get("hidden")
    old_grip = version in ("settled", "settled-basefix", "settled-v3")
    import hidden_state
    from atomic_eval import load_scene
    from atomic_steps import TASKS
    from collect import replay
    from common import SEEN, DATA
    from imagine import set_py_state
    from oracle import set_state
    steps = TASKS[task]
    load_scene(env, scene)
    ep = np.load(f"{SEEN}/run/{task}/{tag}_steps.npz")["states"]
    S0, _ = replay(env, ep, t, steps)
    hidden_state.settle(env, ep[t - 1] if t else None)
    if old_grip:  # the gripper rule of the rows collected before settled-v4
        for r in env.robots:
            for arm, g in r.gripper.items():
                q = env.sim.data.qpos[r._ref_gripper_joint_pos_indexes[arm]]
                gap = abs(q[0] - q[1])
                closing = gap < abs(g.init_qpos[0] - g.init_qpos[1]) * 1.9 - 0.004
                if t and closing:
                    qp = ep[t - 1][1 + np.asarray(r._ref_gripper_joint_pos_indexes[arm])]
                    closing = not (gap - abs(qp[0] - qp[1]) > 0.0005)
                g.current_action = np.array([-1.0, 1.0]) * (1.0 if closing else -1.0)
    arm = env.robots[0].composite_controller.part_controllers["right"]
    bc = env.robots[0].composite_controller.part_controllers["base"]
    if version in ("settled", "settled-basefix"):
        bc.init_pos, bc.init_ori = bc.get_base_pose()
    if version == "settled":
        arm.goal_pos, arm.goal_ori = np.array(arm.ref_pos), np.array(arm.ref_ori_mat)
    H0 = hidden_state.capture(env)
    z = np.load(f"{DATA}/{d}/roll/{tag}_t{t}_c{c}_m{m}.npz")
    saved = z["states"]
    keys = [k for k in z.files if k.startswith("action.")]
    set_py_state(env, row["py_state"])
    set_state(env, ep[t])
    hidden_state.restore(env, H0)
    S = copy.deepcopy(S0)
    k = row["k"]; kk = k; success = False; states = [env.sim.get_state().flatten()]; i = 0; stop = None; hist = []
    n = len(saved) - 1
    for ch in range(len(z[keys[0]])):
        for s in range(16):
            if i >= n:
                break
            env.step(env_action(env, {kk_: z[kk_][ch, s] for kk_ in keys}))
            i += 1
            if env._check_success():
                success = True
            states.append(env.sim.get_state().flatten())
        while kk < len(steps) and steps[kk]["check"](env, S):
            hist.append((i, kk)); kk += 1
        if success or kk >= k + 1:
            stop = i
            break
        if i >= n:
            break
    done = bool(kk > k or success)
    prev_ok = all(steps[q]["check"](env, copy.deepcopy(S)) for q in range(k)) if done and k < len(steps) - 1 else True
    m_ = min(len(states), len(saved))
    diff = [float(np.abs(states[j] - saved[j]).max()) for j in range(m_)]
    return dict(done=done, prev_ok=prev_ok, steps=i, success=success, reached=kk, stop=stop, hist=hist,
                max_state_diff=max(diff), first_diff_step=next((j for j, x in enumerate(diff) if x > 0), None), n_saved=n)


def main():
    import atomic_eval  # noqa: F401
    from analyze_stages import replay_env
    from common import read_rows, rolls, scenes
    d, tag, t, c, m = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    version = sys.argv[6] if len(sys.argv) > 6 else None
    row = next(r for r in read_rows(d) if r["tag"] == tag and r["t"] == t)
    lab = next(x for cc, mm, x in rolls(row) if (cc, mm) == (c, m))
    env = replay_env(row["task"])
    r = resim(env, row["task"], scenes()[tag], d, tag, t, c, m, row, version=version)
    print(json.dumps(dict(stored={f: lab[f] for f in ("done", "prev_ok", "steps", "success", "reached")}, resim=r)))


if __name__ == "__main__":
    main()
