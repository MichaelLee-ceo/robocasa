"""Reproduce a saved rollout without rendering and compare it with the saved states.

Same scene, replay to t (collect.replay: python fixture state), hidden_state settle -> capture -> set_state + restore as
collect.py, then the rollout's saved chunks are stepped through the robosuite env directly (the action dict is mapped to
the env action exactly as RoboCasaGymEnv.step does; rendering does not touch the physics). Variants of settle:
  v4       the current hidden_state.settle
  v3       the gripper rule before bug 4 (fully open = 1.9 x the init gap, opening within 4 mm of it)
  basefix  v3 + the base controller's start pose overwritten (before bug 3)
  settled  basefix + the arm goal in world coordinates (before bug 2)
  true     no settle: continue the main run from t-? (not available: the main run's actions are not saved)
For each variant: max |state - saved| at steps 0/1/2/4/8/16/32, and the signals the scan flags (eef speed relative to
the base, finger gap, base displacement, held object height) next to the main run's.
  ramp=<r> the current settle, then the gripper ramp set to r (r = current_action[1]: +1 closing, -1 opening, 0 = reset)
usage: python -I repro.py <dir> <tag> <t> <c[,c...]> <m> [variant ...]   (default: the row's own variant and v4)"""
import glob, json, os, sys
import numpy as np

R = os.environ["ROBOCASA_ROOT"]
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", f"{R}/runs/20261026_imagine"]
DATA = f"{R}/runs/20261026_critic_data"
SEEN = f"{R}/runs/20261025_seen480"
HIDDEN2VAR = {"settled": "settled", "settled-basefix": "basefix", "settled-v3": "v3", "settled-v4": "v4", None: "settled"}
SCENE_JOBS = ["jobs.json", "jobs_SteamInMicrowave.json", "jobs_PackIdenticalLunches.json"]


def settle_variant(kenv, prev_state, variant):
    """hidden_state.settle (2026-10-29 version) with the earlier behaviours switchable."""
    if variant.startswith("ramp="):
        settle_variant(kenv, prev_state, "v4")
        r = float(variant.split("=")[1])
        for rob in kenv.robots:
            for g in rob.gripper.values():
                if hasattr(g, "current_action"):
                    g.current_action = np.array([-r, r])
        return
    keep_base_init = variant in ("v4", "v3")
    arm_goal_fix = variant in ("v4", "v3", "basefix")
    new_grip = variant == "v4"
    kenv.sim.data.qacc_warmstart[:] = 0
    for r in kenv.robots:
        cc = r.composite_controller
        cc.update_state()
        for c in cc.part_controllers.values():
            c.new_update = True
            c.update(force=True)
            init = (c.init_pos, c.init_ori) if getattr(c, "init_ori", None) is not None else None
            c.reset_goal()
            if init is not None and keep_base_init:
                c.init_pos, c.init_ori = init
            if arm_goal_fix and getattr(c, "input_ref_frame", None) == "base" and hasattr(c, "world_to_origin_frame"):
                c.goal_pos = c.world_to_origin_frame(np.array(c.ref_pos))
                c.goal_ori = c.goal_origin_to_eef_pose()[:3, :3]
        for arm, g in r.gripper.items():
            if not hasattr(g, "current_action"):
                continue
            q = kenv.sim.data.qpos[r._ref_gripper_joint_pos_indexes[arm]]
            gap = abs(q[0] - q[1])
            if new_grip:
                lo, hi = kenv.sim.model.jnt_range[[kenv.sim.model.joint_name2id(j) for j in g.joints]].T
                open_gap = max(abs(hi[0] - lo[1]), abs(lo[0] - hi[1]))
                closing = gap < open_gap - 0.0015
            else:
                open_gap = abs(g.init_qpos[0] - g.init_qpos[1]) * 1.9
                closing = gap < open_gap - 0.004
            if prev_state is not None and closing:
                qp = prev_state[1 + np.asarray(r._ref_gripper_joint_pos_indexes[arm])]
                closing = not (gap - abs(qp[0] - qp[1]) > 0.0005)
            g.current_action = np.array([-1.0, 1.0]) * (1.0 if closing else -1.0)


def env_action(kenv, a):
    from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
    d = PandaOmronKeyConverter.unmap_action({k: np.asarray(v) for k, v in a.items()})
    rob = kenv.robots[0]
    cc, pf = rob.composite_controller, rob.robot_model.naming_prefix
    act = np.zeros(cc.action_limits[0].shape)
    for part in cc.part_controllers:
        s, e = cc._action_split_indexes[part]
        act[s:e] = d.pop(f"{pf}{part}")
    act[-1] = d.pop(f"{pf}base_mode")
    return act


def step_through(kenv, start, H, saved, nsteps, hidden_state, set_state):
    set_state(kenv, start)
    hidden_state.restore(kenv, H)
    keys = [k for k in saved.files if k.startswith("action.")]
    states, ramps = [kenv.sim.get_state().flatten()], [np.array(kenv.robots[0].gripper["right"].current_action)]
    for ci in range(len(saved[keys[0]])):
        for i in range(16):
            if len(states) > nsteps:
                return np.array(states), np.array(ramps)
            kenv.step(env_action(kenv, {k: saved[k][ci, i] for k in keys}))
            states.append(kenv.sim.get_state().flatten())
            ramps.append(np.array(kenv.robots[0].gripper["right"].current_action))
    return np.array(states), np.array(ramps)


def signals(kenv, states, objs):
    import mujoco
    m, d = kenv.sim.model._model, kenv.sim.data._data
    bb = kenv.sim.model.body_name2id("mobilebase0_base")
    site = kenv.robots[0].eef_site_id["right"]
    gi = kenv.robots[0]._ref_gripper_joint_pos_indexes["right"]
    nq = kenv.sim.model.nq
    out = dict(eef_b=[], base=[], gap=[], obj={o: [] for o in objs})
    for s in states:
        d.qpos[:] = s[1:1 + nq]
        mujoco.mj_kinematics(m, d)
        Rb = d.xmat[bb].reshape(3, 3)
        out["eef_b"].append(Rb.T @ (d.site_xpos[site] - d.xpos[bb]))
        out["base"].append([d.xpos[bb][0], d.xpos[bb][1], np.degrees(np.arctan2(Rb[1, 0], Rb[0, 0]))])
        out["gap"].append(abs(s[1 + gi[0]] - s[1 + gi[1]]))
        for o in objs:
            out["obj"][o].append(np.array(d.xpos[kenv.obj_body_id[o]]))
    return {k: (np.array(v) if k != "obj" else {o: np.array(x) for o, x in v.items()}) for k, v in out.items()}


def main():
    import atomic_eval  # noqa: F401
    import hidden_state
    from oracle import reset_to, set_state
    from analyze_stages import replay_env
    from atomic_steps import TASKS
    from imagine import set_py_state
    d, tag, t, m = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[5])
    cands = [int(x) for x in sys.argv[4].split(",")]
    task = tag.split("_s")[0]
    row = next(json.loads(l) for p in glob.glob(f"{DATA}/{d}/results_w*.jsonl") for l in open(p)
               if f'"tag": "{tag}"' in l and f'"t": {t},' in l and '"error"' not in l)
    variants = sys.argv[6:] or sorted({HIDDEN2VAR[row.get("hidden")], "v4"})
    scene = next(j["scene"] for f in SCENE_JOBS for j in json.load(open(f"{DATA}/{f}")) if j["tag"] == tag)
    kenv = replay_env(task)
    z = np.load(scene, allow_pickle=True)
    reset_to(kenv, str(z["xml"]), json.loads(str(z["ep_meta"])))
    kenv.ignore_done = True
    ep = np.load(f"{SEEN}/run/{task}/{tag}_steps.npz")["states"]
    steps, S, k = TASKS[task], {}, 0
    for i in range(t + 1):  # collect.replay: python fixture state and stage memory at t
        set_state(kenv, ep[i])
        if i and i % 16 == 0:
            while k < len(steps) and steps[k]["check"](kenv, S):
                k += 1
    objs = list(kenv.obj_body_id)
    OBJ = os.environ.get("OBJS")
    show_objs = OBJ.split(",") if OBJ else objs
    bc = kenv.robots[0].composite_controller.part_controllers["base"]
    yaw = lambda M: float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    H_rep = hidden_state.capture(kenv)  # controllers as replay() leaves them; every variant settles from here
    NS = int(os.environ.get("NSTEPS", "32"))
    sel = [f for f in (0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32) if f <= NS]
    sig_main = signals(kenv, ep[t:t + NS + 1], objs)
    print(f"{d} {tag} t{t} m{m}: hidden={row.get('hidden')} k={row['k']} (replayed k={k})")
    print("step:                       " + " ".join(f"{f:>6d}" for f in sel))
    fmt = lambda sg, key, f_: " ".join(f"{f_(sg, f):>6s}" for f in sel if f < len(sg["gap"]))
    G = lambda sg, f: f"{1000 * sg['gap'][f]:.1f}"
    print(f"main      finger gap mm      {fmt(sig_main, 'gap', G)}")
    for o in show_objs:
        Z = lambda sg, f, o=o: f"{sg['obj'][o][f, 2]:.3f}"
        print(f"main      {o[:12]:12s} z     {fmt(sig_main, 'obj', Z)}")
    for c in cands:
        saved = np.load(f"{DATA}/{d}/roll/{tag}_t{t}_c{c}_m{m}.npz")
        ss = saved["states"]
        n = min(NS, len(ss) - 1)
        print(f"-- c{c}: done={row['cands'][c]['rollouts'][m]['done']} first chunk control_mode "
              + "".join(str(int(x > 0.5)) for x in saved["action.control_mode"][0, :, 0])
              + " gripper_close " + "".join(str(int(x > 0.5)) for x in saved["action.gripper_close"][0, :, 0]))
        sig_saved = signals(kenv, ss[:n + 1], objs)
        print(f"saved     finger gap mm      {fmt(sig_saved, 'gap', G)}")
        for v in variants:
            set_state(kenv, ep[t])
            hidden_state.restore(kenv, H_rep)
            settle_variant(kenv, ep[t - 1] if t else None, v)
            H = hidden_state.capture(kenv)
            set_py_state(kenv, row["py_state"])
            st, ramps = step_through(kenv, ep[t], H, saved, n, hidden_state, set_state)
            dd = [float(np.abs(st[f] - ss[f]).max()) for f in range(min(len(st), len(ss)))]
            sg = signals(kenv, st, objs)
            sg["ramp"] = ramps[:, 1]
            R_ = lambda sg, f: f"{sg['ramp'][f]:+.1f}"
            print(f"[{v:9s}] ramp at start {H['robots'][0]['grippers']['right']['current_action'][1]:+.1f}; base yaw {yaw(bc.get_base_pose()[1]):.0f}, "
                  f"start yaw {yaw(bc.init_ori):.0f}; max|state - saved| at 1/4/16/32: " + " ".join(f"{dd[f]:.2g}" for f in (1, 4, 16, 32) if f < len(dd)))
            print(f"[{v:9s}] gripper ramp     {fmt(sg, 'gap', R_)}")
            print(f"[{v:9s}] finger gap mm    {fmt(sg, 'gap', G)}")
            for o in show_objs:
                Z = lambda sg, f, o=o: f"{sg['obj'][o][f, 2]:.3f}"
                print(f"[{v:9s}] {o[:12]:12s} z {fmt(sg, 'obj', Z)}")


if __name__ == "__main__":
    main()
