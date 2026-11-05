"""closedloop.py protocol per task: continuous run with node capture at t=48 (capture(obs=True)), a dirtying rollout,
then restore the node and continue with the same seeds; compare sim states, the requests sent, and task-level python
state, bit for bit. Also: what env-level (task) attributes change during steps and are not in capture()/py_state()."""
import sys, json, copy, time
sys.path.insert(0, sys.argv[1])
from harness import *
from fake import wrap, FakeClient, seedf
from atomic_eval import load_scene, rollout
from atomic_steps import TASKS
import hidden_state
from hidden_state import py_state, set_py_state

task = sys.argv[2]; scene_i = int(sys.argv[3]) if len(sys.argv) > 3 else 0
t00 = time.time()
env = make_env(task); base = wrap(env)
start = load_scene(env, scene_path(task, scene_i))
SKIP = {"_obs_cache", "sim_state_initial", "_ep_meta"}
tv0 = plain_vars(env, SKIP)
steps = TASKS[task]
node0 = dict(state=start, ps=py_state(env), hidden=hidden_state.capture(env), S={})
instr = env.get_ep_meta()["lang"]
T_NODE, T_END = 48, 96
snap = {}
bvars = {}
def hook(t, e, S, rew):
    bvars[t] = (plain_vars(e, SKIP), bool(e._check_success()))
    if t == T_NODE:
        snap.update(state=e.sim.get_state().flatten().copy(), ps=py_state(e), hidden=hidden_state.capture(e, obs=True), S=copy.deepcopy(S))
# continuous reference
cA = FakeClient(); refst = []
set_py_state(env, node0["ps"])
rollout(cA, base, env, {}, steps, node0["state"], T_END, "task", instr, k0=0, k_stop=None, seed_fn=seedf("A"), step_states=refst,
        hidden=node0["hidden"], boundary_hook=hook)
tv_end_ref = plain_vars(env, SKIP)
changed = diff_vars(tv0, tv_end_ref)
# dirtying rollout from the root with other seeds
cB = FakeClient(); set_py_state(env, node0["ps"])
rollout(cB, base, env, {}, steps, node0["state"], 64, "task", instr, seed_fn=seedf("B"), hidden=node0["hidden"])
tv_dirty = plain_vars(env, SKIP)
# restore the node and continue with A's seeds from chunk 3
cR = FakeClient(); rst = []
set_py_state(env, snap["ps"])
pre = plain_vars(env, SKIP)
rollout(cR, base, env, copy.deepcopy(snap["S"]), steps, snap["state"], T_END - T_NODE, "task", instr, seed_fn=seedf("A", off=T_NODE // 16),
        step_states=rst, hidden=snap["hidden"])
tv_end_rep = plain_vars(env, SKIP)
ref = np.array(refst[T_NODE:T_END + 1]); rep = np.array(rst)
n = min(len(ref), len(rep))
dev = np.abs(ref[:n] - rep[:n]).max(axis=1)
first = int(np.argmax(dev > 0)) if (dev > 0).any() else None
# requests: chunk 3.. of A vs chunk 0.. of R
req_eq = []
for a, b in zip(cA.reqs[T_NODE // 16:], cR.reqs):
    req_eq.append(all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a if k.startswith("state.")))
out = dict(task=task, scene=scene_i, n=n, exact=bool((dev == 0).all()), first_diverging_step=first, max_dev=float(dev.max()),
           state_requests_equal=req_eq,
           env_attrs_changed_by_steps=changed,
           env_attrs_dirty_vs_node_before_restore=diff_vars(bvars[T_NODE][0], pre),
           env_attrs_end_replay_vs_ref=diff_vars(tv_end_ref, tv_end_rep),
           wall=round(time.time() - t00, 1))
print("RESULT", json.dumps(out, default=str))
