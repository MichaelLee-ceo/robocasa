"""ScrubCuttingBoard official demos 50 / 151 / 478 (labeler_check: official success never reached on replay):
the official counters at the end of the replay, and how many contact frames the finger-1 grasp test dropped."""
import sys, glob, json
from pathlib import Path
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/stages-A")
from common import *
import robocasa.scripts.dataset_scripts.add_subtask_labels_to_lerobot as L
from demo_utils import load_demo
task = "ScrubCuttingBoard"
lerobot = Path(sorted(glob.glob(f"{R}/datasets/v1.0/target/composite/{task}/*/lerobot"))[-1])
env = L.build_env(json.loads((lerobot / "extras" / "dataset_meta.json").read_text())["env_args"])
for ep in map(int, sys.argv[1:]):
    d = load_demo(task, ep)
    reset_to(env, d["xml"], d["ep_meta"])
    touch_nogrip = 0; succ_frames = []
    sb = env.obj_body_id["sponge"]; bb = env.obj_body_id["cutting_board"]
    sg = [env.sim.model.geom_id2name(g) for g in range(env.sim.model.ngeom) if env.sim.model.geom_bodyid[g] == sb]
    bg = [env.sim.model.geom_id2name(g) for g in range(env.sim.model.ngeom) if env.sim.model.geom_bodyid[g] == bb]
    for f in range(len(d["states"])):
        set_state(env, d["states"][f])
        if env._check_success():
            succ_frames.append(f)
        if env.check_contact(sg, bg) and env.check_contact(env.robots[0].gripper["right"], env.objects["sponge"]) and not OU.check_obj_grasped(env, "sponge"):
            touch_nogrip += 1
    pos = np.array(env.board_contact_positions) if env.board_contact_positions else np.zeros((1, 2))
    eef = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
    print(f"ep{ep}: len {len(d['states'])} timer {env.board_contact_timer} sweep {float(np.linalg.norm(pos.max(0) - pos.min(0))):.3f} "
          f"grip-sponge dist end {float(np.linalg.norm(eef - env.sim.data.body_xpos[sb])):.3f} success frames {len(succ_frames)} "
          f"| frames sponge-board contact + gripper touching sponge but check_obj_grasped False: {touch_nogrip}", flush=True)
