"""Skeptic check of 'task-level python state is not captured/restored' with the REAL pipeline code paths:
collect.replay + hidden_state.settle/capture + atomic_eval.rollout (real env.step through the gym wrapper, no render),
a deterministic 'hold still' fake policy. WashLettuce, seen480 episode WashLettuce_s9_single.

Part 1 (collect.py decision_point pattern): candidate A then candidate B from the same decision point, both via
set_py_state(ps) + rollout(hidden=H0). Control: candidate B again with washed_time put back to its pre-A value.
Part 2 (closedloop.py pattern): a stepped node captured with capture(obs=True); 2 candidates + the real execution."""
import sys, json, copy, time
D = sys.argv[1]; sys.path.insert(0, D)
from harness import *
from fake import wrap
sys.path.insert(0, f"{ROOT}/runs/20261026_imagine"); sys.path.insert(0, f"{ROOT}/runs/20261026_critic_data")
import atomic_eval
from atomic_eval import load_scene, rollout
from atomic_steps import TASKS
from oracle import set_state
import hidden_state
from hidden_state import py_state, set_py_state
from collect import replay  # the real collect.py replay

TASK, TAG, ATTR = "WashLettuce", "WashLettuce_s9_single", "washed_time"
RUN = f"{ROOT}/runs/20261025_seen480/run"
job = next(j for j in json.load(open(f"{RUN}/jobs.json")) if j["tag"] == TAG)
states = np.load(f"{RUN}/{TASK}/{TAG}_steps.npz")["states"]
steps = TASKS[TASK]
env = make_env(TASK); base = wrap(env)

# pre-pass: where is the lettuce under water?
load_scene(env, job["scene"])
under = []
for i in range(len(states)):
    set_state(env, states[i]); env._check_success()
    under.append(bool(env.sink.check_obj_under_water(env, "lettuce")))
first_under = under.index(True)
t_node = (first_under // 16 + 1) * 16  # first chunk boundary inside the washing
print("first under water", first_under, "t_node", t_node, "T", len(states) - 1, "under at node", under[t_node], flush=True)

# restore_node logger (in-process wrapper; repo untouched)
_orig = atomic_eval.restore_node
LOG = []
def restore_node_logged(kenv, s, hidden=None):
    before = getattr(kenv, ATTR)
    _orig(kenv, s, hidden)
    LOG.append((before, getattr(kenv, ATTR)))
atomic_eval.restore_node = restore_node_logged


class Hold:
    """zero arm / base motion, gripper command kept as at the node."""
    def __init__(self, grip):
        self.grip = grip
    def get_action(self, req):
        out = {"action.end_effector_position": np.zeros((1, 16, 3)), "action.end_effector_rotation": np.zeros((1, 16, 3)),
               "action.gripper_close": np.full((1, 16, 1), self.grip), "action.base_motion": np.zeros((1, 16, 4)),
               "action.control_mode": np.zeros((1, 16, 1))}
        if "_seed" in req:
            out["_seed_echo"] = req["_seed"]
        return out


def cand(client, S0, k, start, hidden, L, attr_override=None, ps=None):
    bounds, st = [], []
    def hook(t, e, S, rew):
        bounds.append((int(t), int(getattr(e, ATTR)), int(np.argmax(rew > 0)) if np.any(rew > 0) else None))
    set_py_state(env, ps)
    if attr_override is not None:
        setattr(env, ATTR, attr_override)
    r, _ = rollout(client, base, env, copy.deepcopy(S0), steps, start, L, "task", "wash", k0=k, k_stop=k + 1,
                   seed_fn=lambda ci: ci, hidden=hidden, step_states=st, boundary_hook=hook)
    return dict(start_after_restore=LOG[-1][1], before_restore=LOG[-1][0], success=r["success"], steps=r["steps"],
                reached=r["reached"], bounds=bounds, end=int(getattr(env, ATTR))), np.array(st)

res = dict(task=TASK, tag=TAG, t_node=t_node)
# ---- Part 1: collect.py decision_point
load_scene(env, job["scene"])
S0, k = replay(env, states, t_node, steps)
hidden_state.settle(env, states[t_node - 1]); H0 = hidden_state.capture(env); ps = py_state(env)
grip = 1.0 if env.robots[0].gripper["right"].current_action[1] > 0 else 0.0
w_node = int(getattr(env, ATTR))
res["p1_node"] = dict(k_replay=k, washed_time=w_node, grip=grip, attr_in_py_state=any(ATTR in d for d in ps.values()),
                      attr_in_capture=ATTR in json.dumps(list(H0.keys())))
A, stA = cand(Hold(grip), S0, k, states[t_node], H0, 64, ps=ps)
B, stB = cand(Hold(grip), S0, k, states[t_node], H0, 64, ps=ps)
Bc, stBc = cand(Hold(grip), S0, k, states[t_node], H0, 64, attr_override=A["before_restore"], ps=ps)
res["p1_candA"] = A; res["p1_candB_leaked"] = B; res["p1_candB_clean"] = Bc
n = min(len(stB), len(stBc))
res["p1_B_vs_clean_states_identical"] = bool(np.array_equal(stB[:n], stBc[:n]))
res["p1_B_vs_clean_len"] = [len(stB), len(stBc)]

# ---- Part 2: closedloop.py stepped node (capture obs=True after a real executed chunk)
load_scene(env, job["scene"])
S0, k = replay(env, states, t_node - 16, steps)
hidden_state.settle(env, states[t_node - 17]); Hs = hidden_state.capture(env); ps0 = py_state(env)
set_py_state(env, ps0)
S1 = copy.deepcopy(S0)
r, _ = rollout(Hold(grip), base, env, S1, steps, states[t_node - 16], 16, "task", "wash", k0=k, k_stop=k + 1,
               seed_fn=lambda ci: ci, hidden=Hs)
node = dict(state=env.sim.get_state().flatten().copy(), ps=py_state(env), hidden=hidden_state.capture(env, obs=True), S=S1,
            k=max(k, int(r["reached"])), w=int(getattr(env, ATTR)), success=bool(env._check_success()))
res["p2_node"] = dict(washed_time=node["w"], success=node["success"], k=node["k"])
c0, s0 = cand(Hold(grip), node["S"], node["k"], node["state"], node["hidden"], 64, ps=node["ps"])
c1, s1 = cand(Hold(grip), node["S"], node["k"], node["state"], node["hidden"], 64, ps=node["ps"])
ex, sx = cand(Hold(grip), node["S"], node["k"], node["state"], node["hidden"], 16, ps=node["ps"])
res["p2_cand0"] = c0; res["p2_cand1"] = c1; res["p2_exec"] = ex
res["p2_exec_first16_equal_cand0"] = bool(np.array_equal(sx[:17], s0[:17]))
print("RESULT", json.dumps(res, default=str), flush=True)
