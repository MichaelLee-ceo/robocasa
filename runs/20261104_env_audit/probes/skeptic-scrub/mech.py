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
    sb = env.obj_body_id["sponge"]; bb = env.obj_body_id["cutting_board"]
    sg = [env.sim.model.geom_id2name(g) for g in range(env.sim.model.ngeom) if env.sim.model.geom_bodyid[g] == sb]
    bg = [env.sim.model.geom_id2name(g) for g in range(env.sim.model.ngeom) if env.sim.model.geom_bodyid[g] == bb]
    nflip = 0; nboth = 0; dxy = []
    for t in range(len(rec) - 1):
        env.step(acts[t])
        c_live = env.check_contact(sg, bg); xy_live = np.array(env.sim.data.body_xpos[sb][:2])
        st = env.sim.get_state().flatten(); ws = np.array(env.sim.data.qacc_warmstart)
        env.sim.forward()
        c_fwd = env.check_contact(sg, bg); xy_fwd = np.array(env.sim.data.body_xpos[sb][:2])
        env.sim.set_state_from_flattened(st); env.sim.data.qacc_warmstart[:] = ws
        nflip += c_live != c_fwd; nboth += c_live or c_fwd; dxy.append(float(np.linalg.norm(xy_live - xy_fwd)))
    print("MECH", ep, "frames with sponge-board contact (either)", nboth, "contact flag differs live vs after forward", nflip,
          "sponge xy shift live vs forward: max %.4f mean %.5f" % (max(dxy), float(np.mean(dxy))), flush=True)
