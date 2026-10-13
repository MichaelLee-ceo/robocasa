"""Composite task -> atomic steps experiment (composite_seen).

conditions (all start from demo states of the target dataset, so they are paired):
  baseline     : demo frame 0, original task instruction, original horizon; official env._check_success();
                 the atomic step checks only log progress (-> where the episode got stuck)
  chain_atomic : demo frame 0, instruction = atomic-template instruction of the current step; switch to the next step
                 when the step check passes; official env._check_success() decides final success
  reset_atomic : for every step k, start at the demo frame where step k begins, atomic instruction, budget
                 clip(3 x demo step length, 240, horizon); success = step check (or full task success)
  reset_task   : same as reset_atomic with the original task instruction
"""
import argparse, json, os, sys, time, traceback
from collections import defaultdict, deque
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/third_party/Isaac-GR00T"))
import robocasa  # noqa
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from groot_eval import Client, batch, write_video
from demo_utils import load_demo
from oracle import reset_to, set_state
from atomic_steps import TASKS, prepare


class FrameGrab(gym.Wrapper):
    def __init__(self, env, every=4):
        super().__init__(env); self.every, self.t, self.frames = every, 0, []

    def step(self, a):
        o, r, d, tr, info = self.env.step(a)
        self.t += 1
        if self.t % self.every == 0:
            self.frames.append(np.concatenate([o["video.robot0_agentview_left"], o["video.robot0_eye_in_hand"]], 1))
        return o, r, d, tr, info


def rollout(client, base, kenv, S, steps, start_state, horizon, mode, task_instr, k0=0, k_stop=None, save=None):
    fg = FrameGrab(base)
    menv = MultiStepWrapper(fg, video_delta_indices=np.array([0]), state_delta_indices=np.array([0]),
                            n_action_steps=16, max_episode_steps=horizon)
    set_state(kenv, start_state)
    o = base.get_observation(kenv._get_observations(force_update=True))
    menv.obs = deque([o] * 2, maxlen=2); menv.reward, menv.done = [], []
    menv.info = defaultdict(lambda: deque(maxlen=2))
    obs = menv._get_obs(menv.video_delta_indices, menv.state_delta_indices)
    k = k0; hist = []; instr_log = []; t = 0; success = False; done = False; debug = []
    def instr_for(kk):
        if mode == "task" or (mode == "atomic_then_task" and kk >= 1):
            return task_instr
        st = steps[min(kk, len(steps) - 1)]
        if mode == "atomic_aug" and "instr_aug" in st:
            return st["instr_aug"](kenv, S)
        return st["instr"](kenv, S)
    cur_instr = instr_for(k)
    while not done:
        obs["annotation.human.task_description"] = cur_instr
        if not instr_log or instr_log[-1][1] != obs["annotation.human.task_description"]:
            instr_log.append((t, obs["annotation.human.task_description"]))
        act = {kk: np.asarray(v)[0] for kk, v in client.get_action(batch(obs)).items()}
        obs, r, done, trunc, info = menv.step(act)
        t += len(info["rewards"])
        if np.any(info["rewards"] > 0):
            success = True
        while k < len(steps) and steps[k]["check"](kenv, S):
            hist.append((t, k)); k += 1
            if save is not None:  # state at the chunk boundary where step k-1 finished = correct start of step k
                save.append(dict(t=t, k=k, state=kenv.sim.get_state().flatten().copy(), S=dict(S)))
            if k < len(steps):
                cur_instr = instr_for(k)
            if k < len(steps) and "debug" in steps[k]:
                debug.append(dict(t=t, when="step_start", step=k, **steps[k]["debug"](kenv, S)))
        if success or (k_stop is not None and k >= k_stop):
            break
    for kk, st in enumerate(steps):
        if "debug" in st:
            debug.append(dict(t=t, when="end", step=kk, **st["debug"](kenv, S)))
    return dict(success=bool(success), steps=int(t), reached=int(k), n_steps=len(steps), history=hist,
                instr_log=instr_log, debug=debug), fg.frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1); ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=5555)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))[a.worker::a.n_workers]
    client = Client(port=a.port, timeout_ms=600000)
    res_path = os.path.join(a.out, f"results_w{a.worker}.jsonl")
    import glob as _glob
    done_tags = {json.loads(l)["tag"] for p_ in _glob.glob(os.path.join(a.out, "results_w*.jsonl")) for l in open(p_)
                 if "error" not in json.loads(l)}
    envs = {}
    def _done_now():
        return {json.loads(l)["tag"] for p_ in _glob.glob(os.path.join(a.out, "results_w*.jsonl")) for l in open(p_)
                if "error" not in json.loads(l)}
    for j in jobs:
        if j["tag"] in done_tags or j["tag"] in _done_now():
            continue
        t0 = time.time()
        try:
            task = j["task"]
            if task not in envs:
                for e in envs.values():
                    e.close()
                envs = {task: gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped}
            base = envs[task]; kenv = base.env
            if j["condition"] in ("reset_seed_atomic", "reset_seed_task"):
                src = np.load(os.path.join(a.out, task, f"{j['src_tag']}_states.npz"), allow_pickle=True)
                rec = [x for x in src["recs"] if x["k"] == j["step"]][0]
                base.reset(seed=j["seed"]); steps = TASKS[task]; S = dict(rec["S"])
                instr0 = kenv.get_ep_meta()["lang"]
                budget = int(j["horizon"] - rec["t"])
                r, frames = rollout(client, base, kenv, S, steps, rec["state"], budget,
                                    "atomic" if j["condition"] == "reset_seed_atomic" else "task", instr0,
                                    k0=j["step"], k_stop=j["step"] + 1)
                r.update(step=j["step"], budget=budget, start_t=int(rec["t"]), skill=steps[j["step"]]["skill"],
                         step_success=bool(r["reached"] > j["step"] or r["success"]), task_instr=instr0)
                os.makedirs(os.path.join(a.out, task), exist_ok=True)
                write_video(os.path.join(a.out, task, f"{j['tag']}.mp4"), frames, fps=10)
                r.update(j); r["wall_s"] = round(time.time() - t0, 1)
                with open(res_path, "a") as f:
                    f.write(json.dumps(r, default=str) + "\n")
                print(json.dumps({k: r.get(k) for k in ("tag", "success", "step_success", "steps", "wall_s")}), flush=True)
                continue
            if j["condition"] in ("chain_seed", "baseline_seed", "chain_then_task_seed", "chain_aug_seed", "baseline_seed_save"):
                base.reset(seed=j["seed"]); steps = TASKS[task]; S = {}
                start = kenv.sim.get_state().flatten().copy(); instr0 = kenv.get_ep_meta()["lang"]
                saved = [] if j["condition"] == "baseline_seed_save" else None
                r, frames = rollout(client, base, kenv, S, steps, start, j["horizon"],
                                    {"baseline_seed": "task", "baseline_seed_save": "task", "chain_seed": "atomic",
                                     "chain_then_task_seed": "atomic_then_task", "chain_aug_seed": "atomic_aug"}[j["condition"]], instr0, save=saved)
                if saved is not None:
                    os.makedirs(os.path.join(a.out, task), exist_ok=True)
                    np.savez_compressed(os.path.join(a.out, task, f"{j['tag']}_states.npz"), recs=np.array(saved, dtype=object))
                    r["saved_steps"] = [x["k"] for x in saved]
                r["atomic_instructions"] = [st["instr"](kenv, S) for st in steps]; r["skills"] = [st["skill"] for st in steps]
                r.update(task_instr=instr0)
                os.makedirs(os.path.join(a.out, task), exist_ok=True)
                write_video(os.path.join(a.out, task, f"{j['tag']}.mp4"), frames, fps=10)
                r.update(j); r["wall_s"] = round(time.time() - t0, 1)
                with open(res_path, "a") as f:
                    f.write(json.dumps(r, default=str) + "\n")
                print(json.dumps({k: r.get(k) for k in ("tag", "success", "reached", "steps", "wall_s")}), flush=True)
                continue
            d = load_demo(task, j["demo"])
            reset_to(kenv, d["xml"], d["ep_meta"]); S = prepare(kenv, d, set_state)
            reset_to(kenv, d["xml"], d["ep_meta"])  # fresh fixture python state (see README: microwave/coffee flags)
            steps = TASKS[task]; segs = d["segs"]
            starts = [segs[s["seg"]]["start"] for s in steps] + [segs[-1]["start"]]
            cond = j["condition"]
            if cond in ("baseline", "chain_atomic", "chain_task"):
                r, frames = rollout(client, base, kenv, S, steps, d["states"][0], j["horizon"],
                                    "atomic" if cond == "chain_atomic" else "task", d["task_instr"])
            else:
                k = j["step"]
                L = starts[k + 1] - starts[k]
                budget = int(np.clip(3 * L, 240, j["horizon"]))
                r, frames = rollout(client, base, kenv, S, steps, d["states"][starts[k]], budget,
                                    "atomic" if cond == "reset_atomic" else "task", d["task_instr"], k0=k, k_stop=k + 1)
                r.update(step=k, budget=budget, demo_step_len=int(L), skill=steps[k]["skill"],
                         step_success=bool(r["reached"] > k or r["success"]))
            set_state(kenv, d["states"][starts[min(1, len(steps) - 1)]])
            r["atomic_instructions"] = [s["instr"](kenv, S) for s in steps]
            r["skills"] = [s["skill"] for s in steps]
            os.makedirs(os.path.join(a.out, task), exist_ok=True)
            write_video(os.path.join(a.out, task, f"{j['tag']}.mp4"), frames, fps=10)
            r.update(task_instr=d["task_instr"], demo_len=d["length"], step_starts=starts[:-1])
        except Exception as e:
            r = dict(error=repr(e), tb=traceback.format_exc()[-2000:])
        r.update(j); r["wall_s"] = round(time.time() - t0, 1)
        with open(res_path, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        print(json.dumps({k: r.get(k) for k in ("tag", "success", "step_success", "reached", "steps", "wall_s", "error")}), flush=True)


if __name__ == "__main__":
    main()
