"""Which components differ after the first step(s), when the same scene is loaded twice in a row? Also py_state, sim state
right after load_scene, and every plain env/task attribute."""
import sys, json, copy
HERE = sys.argv[1]; sys.path.insert(0, HERE)
from harness import *
from atomic_eval import load_scene, restore_node
import hidden_state
task = sys.argv[2]; nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
env = make_env(task)
acts = [action_vec(env, *a) for a in rand_actions(32, 7, p_base=0.4)]
m = env.sim.model
def names():
    out = ["time"]
    for j in range(m.njnt):
        n = m.joint_id2name(j); t = m.jnt_type[j]; k = {0: 7, 1: 4, 2: 1, 3: 1}[t]
        out += [f"qpos:{n}[{i}]" for i in range(k)]
    for j in range(m.njnt):
        n = m.joint_id2name(j); t = m.jnt_type[j]; k = {0: 6, 1: 3, 2: 1, 3: 1}[t]
        out += [f"qvel:{n}[{i}]" for i in range(k)]
    return out
def envattrs():
    return {k: copy.deepcopy(v) for k, v in vars(env).items() if hidden_state._plain(v) or (isinstance(v, np.ndarray) and v.size < 1000)}
def trial(path):
    st = load_scene(env, path)
    rec = dict(load=env.sim.get_state().flatten().copy(), ps=hidden_state.py_state(env), attrs=envattrs(),
               H=hidden_state.capture(env), ctrl=np.array(env.sim.data.ctrl), ws=np.array(env.sim.data.qacc_warmstart))
    restore_node(env, st, None); env._get_observations(force_update=True)
    rec["steps"] = []
    for a in acts[:nsteps]:
        env.step(a); rec["steps"].append(env.sim.get_state().flatten().copy())
    return rec
s0 = scene_path(task, 0)
A = trial(s0); B = trial(s0)
nm = names()
print("load_state_equal", np.array_equal(A["load"], B["load"]), "ctrl_eq", np.array_equal(A["ctrl"], B["ctrl"]), "ws_eq", np.array_equal(A["ws"], B["ws"]))
print("py_state diff", hidden_state._plain and [n for n in A["ps"] if repr(A["ps"][n]) != repr(B["ps"].get(n))])
for n in A["ps"]:
    if repr(A["ps"][n]) != repr(B["ps"].get(n)):
        print("  ", n, {k: (A["ps"][n][k], B["ps"][n].get(k)) for k in A["ps"][n] if repr(A["ps"][n][k]) != repr(B["ps"][n].get(k))})
ad = [k for k in set(A["attrs"]) | set(B["attrs"]) if repr(A["attrs"].get(k)) != repr(B["attrs"].get(k))]
print("env attr diff", {k: (repr(A["attrs"].get(k))[:80], repr(B["attrs"].get(k))[:80]) for k in ad})
for i in range(nsteps):
    d = np.abs(A["steps"][i] - B["steps"][i]); idx = np.argsort(-d)[:6]
    print("step", i + 1, "max", d.max(), [(nm[j] if j < len(nm) else j, float(d[j])) for j in idx if d[j] > 0])
