"""ScrubCuttingBoard: does rebuilding the scrub counter by replaying states (set_state + update_state, as
collect.replay / stage_trace do) give the same counter as a live run on the SAME trajectory?
For each demo: (1) play the demo's actions open-loop live (official playback path), tracking the live counter,
live per-step success and divergence from the recorded states; (2) replay the LIVE states with set_state and
compare the counter frame by frame; (3) replay the RECORDED states (what labeler_check did)."""
import sys, os, json, ast, glob
from pathlib import Path
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
os.environ.setdefault("MUJOCO_GL", "egl")
import robosuite, robocasa  # noqa
from oracle import reset_to, set_state
from demo_utils import load_demo
import robocasa.utils.object_utils as OU

_H5 = {"end_effector_position": (0, 3), "end_effector_rotation": (3, 6), "gripper_close": (6, 7), "base_motion": (7, 11), "control_mode": (11, 12)}
def reorder_lerobot_action(a, ds):  # copy of robocasa.utils.lerobot_utils.reorder_lerobot_action (that module needs lerobot)
    info = json.load(open(f"{ds}/meta/modality.json"))["action"]
    out = np.zeros_like(a)
    for k, v in info.items():
        h0, h1 = _H5[k]; out[:, h0:h1] = a[:, v["start"]:v["end"]]
    return out
import pandas as pd

task = "ScrubCuttingBoard"
lerobot = Path(sorted(glob.glob(f"{R}/datasets/v1.0/target/composite/{task}/*/lerobot"))[-1])
m = json.loads((lerobot / "extras" / "dataset_meta.json").read_text())
ea = ast.literal_eval(m["env_args"]) if isinstance(m["env_args"], str) else m["env_args"]
ek = dict(ea["env_kwargs"]); ek.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                                      translucent_robot=False, ignore_done=True)
env = robosuite.make(**ek)


def counters(e):
    p = np.array(e.board_contact_positions) if e.board_contact_positions else np.zeros((1, 2))
    return e.board_contact_timer, round(float(np.linalg.norm(p.max(0) - p.min(0))), 3)


def replay(d, states):
    reset_to(env, d["xml"], d["ep_meta"], states[0])
    env.board_contact_positions, env.board_contact_timer = [], 0
    tl, sf = [], []
    for f in range(len(states)):
        set_state(env, states[f])
        if env._check_success():
            sf.append(f)
        tl.append(env.board_contact_timer)
    return tl, sf, counters(env)


for ep in map(int, sys.argv[1:]):
    d = load_demo(task, ep)
    df = pd.read_parquet(f"{lerobot}/data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet")
    acts = reorder_lerobot_action(np.stack(df["action"].to_list()), lerobot)
    rec = d["states"]
    reset_to(env, d["xml"], d["ep_meta"], rec[0])
    env.board_contact_positions, env.board_contact_timer = [], 0
    live_states = [env.sim.get_state().flatten()]
    live_t, live_sf, div = [env.board_contact_timer], [], []
    for t in range(len(rec) - 1):
        _, rwd, _, _ = env.step(acts[t])
        s = env.sim.get_state().flatten()
        live_states.append(s)
        live_t.append(env.board_contact_timer)
        if rwd > 0:
            live_sf.append(t + 1)
        div.append(float(np.abs(s[1:] - rec[t + 1][1:]).max()))
    lc = counters(env)
    rl_t, rl_sf, rlc = replay(d, np.array(live_states))
    rr_t, rr_sf, rrc = replay(d, rec)
    first_diff = next((i for i in range(len(live_t)) if live_t[i] != rl_t[i]), None)
    print(json.dumps(dict(ep=ep, T=len(rec), live=dict(timer_sweep=lc, success_frames=len(live_sf), first_success=live_sf[0] if live_sf else None),
                          replay_live_states=dict(timer_sweep=rlc, success_frames=len(rl_sf), first_success=rl_sf[0] if rl_sf else None,
                                                  first_counter_mismatch=first_diff),
                          replay_recorded=dict(timer_sweep=rrc, success_frames=len(rr_sf), first_success=rr_sf[0] if rr_sf else None),
                          divergence_max=round(max(div), 4), divergence_at_50=round(div[min(49, len(div) - 1)], 5))), flush=True)
