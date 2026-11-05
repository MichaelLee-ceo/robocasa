"""Does the node restore used by imagine / closedloop / collect put back the TASK's own python counters
(WashLettuce.washed_time, RinseSinkBasin.washed_loc, ScrubCuttingBoard.board_contact_*)?"""
import sys, copy
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/stages-A")
from common import *
import hidden_state
from hidden_state import py_state, set_py_state
sys.path.insert(0, f"{R}/third_party/Isaac-GR00T")
from atomic_eval import restore_node
task = sys.argv[1]
env = make_env(task)
s0 = load(env, task, 0)
ATTR = {"WashLettuce": ["washed_time"], "RinseSinkBasin": ["washed_loc"],
        "ScrubCuttingBoard": ["board_contact_positions", "board_contact_timer", "sponge_contact_height"]}[task]
print(task, "at load:", {a: getattr(env, a) for a in ATTR}, "success", env._check_success())
node = dict(state=s0, ps=py_state(env), hidden=hidden_state.capture(env))
print("py_state fixtures captured:", sorted(node["ps"].keys()), "| env attrs in hidden:", [a for a in ATTR if a in str(node["hidden"].keys())])
# a previous candidate from this node left the task counters at 'done' (what a successful branch leaves behind)
if task == "WashLettuce":
    env.washed_time = 30
elif task == "RinseSinkBasin":
    env.washed_loc = [True, True, True]
else:
    env.board_contact_positions = [np.array([0.0, 0.0]), np.array([0.1, 0.1])] + [np.array([0.03 * i, 0.0]) for i in range(2, 5)]
    env.board_contact_timer = 5
# restore the node exactly as imagine.candidate / closedloop do
set_py_state(env, node["ps"])
restore_node(env, node["state"], node["hidden"])
print("after restore_node:", {a: getattr(env, a) for a in ATTR}, "official success at the scene start:", env._check_success(),
      "| sim state equal to node:", bool(np.array_equal(env.sim.get_state().flatten(), s0)))
eef = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
if task == "ScrubCuttingBoard":
    print("gripper-sponge dist at start", float(np.linalg.norm(eef - env.sim.data.body_xpos[env.obj_body_id["sponge"]])))
