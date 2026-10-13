import sys, os, json, numpy as np, gymnasium as gym
sys.path.insert(0, os.path.expanduser('~/Desktop/robocasa/runs/scripts')); sys.path.insert(0, os.path.expanduser('~/Desktop/robocasa/third_party/Isaac-GR00T'))
import robocasa
from collections import defaultdict, deque
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from groot_eval import Client, batch
from demo_utils import load_demo
from oracle import reset_to, set_state
c = Client(port=5556, timeout_ms=600000)
base = gym.make("robocasa/WaffleReheat", split="target", enable_render=True).unwrapped; k = base.env
for dm, name in [(0,'close the microwave door'),(150,'close the microwave door'),(350,'close the microwave door'),(0,'place the bowl'),(150,'place the bowl')]:
    d = load_demo("WaffleReheat", dm); reset_to(k, d["xml"], d["ep_meta"])
    s = [x for x in d['segs'] if name in x['instr']][0]
    menv = MultiStepWrapper(base, video_delta_indices=np.array([0]), state_delta_indices=np.array([0]), n_action_steps=16, max_episode_steps=600)
    set_state(k, d["states"][s["start"]]); o = base.get_observation(k._get_observations(force_update=True))
    menv.obs = deque([o]*2, maxlen=2); menv.reward, menv.done = [], []; menv.info = defaultdict(lambda: deque(maxlen=2))
    obs = menv._get_obs(menv.video_delta_indices, menv.state_delta_indices); mw = k.microwave
    door0 = mw.get_joint_state(k, mw.door_joint_names)[mw.door_joint_names[0]]
    steps = 0; trace = []
    while steps < 600:
        obs["annotation.human.task_description"] = d["task_instr"]
        a = {kk: np.asarray(v)[0] for kk, v in c.get_action(batch(obs)).items()}
        # step one by one to catch the exact success step
        for t in range(16):
            ob, r, done, tr, info = base.step({kk: v[t] for kk, v in a.items()}); steps += 1
            q = mw.get_joint_state(k, mw.door_joint_names)[mw.door_joint_names[0]]
            if r > 0: break
        trace.append(round(float(q),2))
        menv.obs.append(ob); obs = menv._get_obs(menv.video_delta_indices, menv.state_delta_indices)
        if r > 0: break
    print(json.dumps(dict(demo=dm, start=name, door_at_start=round(float(door0),2), success=bool(r>0), steps=steps, door_at_end=round(float(q),2), door_trace_per_chunk=trace[:12])), flush=True)
