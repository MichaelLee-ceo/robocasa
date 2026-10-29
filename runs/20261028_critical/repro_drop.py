"""Reproduce a collected rollout exactly (same env, replay, settle, capture/restore and rollout() as
../20261026_critic_data/collect.py), with a client that answers with the rollout's saved chunks, and compare its states
with the saved ones. Then the same with the current hidden_state.settle, to test a fix.
usage (renders: give it a GPU): python -I repro_drop.py <task> <tag> <t> <cand> [old_goal]"""
import json, os, sys
import numpy as np
import gymnasium as gym

R = os.environ["ROBOCASA_ROOT"]
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261026_imagine", f"{R}/runs/20261026_critic_data"]
import hidden_state  # noqa: E402
from atomic_eval import load_scene, rollout  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
from collect import replay  # noqa: E402
from imagine import set_py_state  # noqa: E402
from oracle import set_state  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
DS = os.environ.get("DATASET", "coarse")  # coarse / dense / confirm / control (data.py)


class Saved:
    def __init__(self, npz):
        keys = [k for k in npz.files if k.startswith("action.")]
        self.chunks = [{k: npz[k][c] for k in keys} for c in range(len(npz[keys[0]]))]
        self.i = 0

    def get_action(self, req):
        ch = self.chunks[min(self.i, len(self.chunks) - 1)]; self.i += 1
        return {k: np.asarray(v)[None] for k, v in ch.items()}


def main():
    task, tag, t, cand = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    old_goal = len(sys.argv) > 5 and sys.argv[5] == "old_goal"
    old_base = len(sys.argv) > 5 and sys.argv[5] == "old_base"  # the base controller's start pose overwritten, as before
    old_grip = len(sys.argv) > 5 and sys.argv[5] == "old_grip"  # the gripper ramp at "open", as the old rule inferred
    res = data.rows(task, DS)[(tag, t)]
    roll = f"{data.roll(res)}/{tag}_t{t}_c{cand}_m0.npz"
    j = dict(scene=data.scenes()[tag])
    base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
    kenv = base.env
    steps = TASKS[task]
    load_scene(kenv, j["scene"])
    ep = np.load(f"{R}/runs/20261025_seen480/run/{task}/{tag}_steps.npz")["states"]
    S0, _ = replay(kenv, ep, t, steps)
    if t:  # at t = 0 the state after load_scene is exact (settled-v5)
        hidden_state.settle(kenv, ep[t - 1])
    arm = kenv.robots[0].composite_controller.part_controllers["right"]
    print(f"origin_pos {np.round(arm.origin_pos, 3)}, ref_pos (world) {np.round(arm.ref_pos, 3)}, goal_pos after settle {np.round(arm.goal_pos, 3)}")
    if old_goal:  # what settle left before the 2026-10-28 fix
        arm.goal_pos, arm.goal_ori = np.array(arm.ref_pos), np.array(arm.ref_ori_mat)
    bc = kenv.robots[0].composite_controller.part_controllers["base"]
    yaw = lambda m: np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    cur = bc.get_base_pose()[1]
    print(f"base yaw now {yaw(cur):.0f} deg, controller's start yaw {yaw(bc.init_ori):.0f} deg")
    if old_base:
        bc.init_pos, bc.init_ori = bc.get_base_pose()
    gr = kenv.robots[0].gripper["right"]
    print(f"gripper ramp after settle: {gr.current_action}")
    if old_grip:
        gr.current_action = np.array([1.0, -1.0])
    H0 = hidden_state.capture(kenv)
    set_py_state(kenv, res["py_state"])
    saved = np.load(roll)
    states = []
    rollout(Saved(saved), base, kenv, dict(S0), steps, ep[t], 16 * (int(os.environ.get("VCHUNK", "1")) + 2), "task", kenv.get_ep_meta()["lang"], k0=j["k"], k_stop=j["k"] + 1,
            step_states=states, hidden=H0)
    ss = saved["states"]
    n = min(len(states), len(ss))
    d = [float(np.abs(np.asarray(states[f]) - ss[f]).max()) for f in range(n)]
    print(f"{'old settle' if old_goal or old_base or old_grip else 'current settle'}: max |state - saved state| at steps 0/4/8/16/32: "
          + " ".join(f"{d[f]:.2g}" for f in (0, 4, 8, 16, 32) if f < n))
    bid = kenv.sim.model.body_name2id("mobilebase0_base")
    bx = []
    for f in (0, 8, 16, 32):
        if f < len(states):
            set_state(kenv, states[f]); bx.append(tuple(np.round(kenv.sim.data.body_xpos[bid][:2], 2)))
    print(f"   base position at steps 0/8/16/32: {bx}")
    vc = int(os.environ.get("VCHUNK", "1"))  # chunk to compare: continuing vs restored at its start with settle
    if not old_goal and not old_base and len(states) > 16 * (vc + 1) + 1 and not os.environ.get("RUNDIR"):
        # settle vs the true hidden state: run chunk 2 from the state reached after chunk 1, once continuing (states above)
        # and once restored with set_state + settle, with the same actions
        sim2 = []
        rollout(Saved(saved), base, kenv, dict(S0), steps, ep[t], 16 * (int(os.environ.get("VCHUNK", "1")) + 2), "task", kenv.get_ep_meta()["lang"], k0=j["k"], k_stop=j["k"] + 1,
                step_states=sim2, hidden=H0)  # deterministic: same as states
        s0 = 16 * vc
        set_state(kenv, states[s0]); hidden_state.settle(kenv, states[s0 - 1])
        H16 = hidden_state.capture(kenv)
        sv = Saved(saved); sv.i = vc
        b = []
        rollout(sv, base, kenv, dict(S0), steps, states[s0], 32, "task", kenv.get_ep_meta()["lang"], k0=j["k"], k_stop=j["k"] + 1,
                step_states=b, hidden=H16)
        mode = saved["action.control_mode"][vc, :, 0]
        dd = [float(np.abs(np.asarray(b[f]) - np.asarray(states[s0 + f])).max()) for f in range(min(17, len(b)))]
        print(f"   chunk {vc + 1} (mode {mode.mean():.2f}) continuous vs restored at step {s0} + settle: max |state diff| after 1/4/8/16 steps "
              + " ".join(f"{dd[f]:.2g}" for f in (1, 4, 8, 16) if f < len(dd)))
        m = kenv.sim.model
        names = ["time"] + [f"qpos:{m.joint_id2name(j)}" for j in range(m.njnt) for _ in range({0: 7, 1: 4, 2: 1, 3: 1}[m.jnt_type[j]])] \
            + [f"qvel:{m.joint_id2name(j)}" for j in range(m.njnt) for _ in range({0: 6, 1: 3, 2: 1, 3: 1}[m.jnt_type[j]])]
        for f in (1, 8):
            diff = np.abs(np.asarray(b[f]) - np.asarray(states[s0 + f]))
            top = np.argsort(-diff)[:5]
            print(f"   step {f} largest diffs: " + ", ".join(f"{names[i] if i < len(names) else i} {diff[i]:.3g}" for i in top))
        ctl = kenv.robots[0].composite_controller.part_controllers
        print("   part controllers:", {k: type(v).__name__ for k, v in ctl.items()})
    obj = os.environ.get("OBJ") or next((o for o in ("bowl", "obj", "meat0", "vegetable0") if o in kenv.obj_body_id), None)
    if obj:
        zs = []
        for f in (0, 8, 16, 24, 32, 48):
            if f < len(states):
                set_state(kenv, states[f]); zs.append(round(float(kenv.sim.data.body_xpos[kenv.obj_body_id[obj]][2]), 3))
        print(f"   {obj} height at steps 0/8/16/24/32/48: {zs}")


if __name__ == "__main__":
    main()
