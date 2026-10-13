"""GR00T N1.5 eval runner for RoboCasa (client side; talks to scripts/run_eval.py --server).

Mirrors gr00t/eval/simulation.py (RoboCasaGymEnv + MultiStepWrapper, n_action_steps=16, obs batch dim 1)
but adds: per-episode seeds, early stop on success (metric unchanged: success = ever succeeded),
instruction override hooks, sim-state logging every chunk (for post-hoc subtask analysis), and video.
"""
import argparse, gzip, json, os, sys, time, traceback
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.expanduser("~/Desktop/robocasa/third_party/Isaac-GR00T"))
import robocasa  # noqa: F401  (registers robocasa/<Task> gym ids)
from robocasa.utils.dataset_registry_utils import get_task_horizon
from gr00t.eval.service import BaseInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper


class Client(BaseInferenceClient):
    def get_action(self, obs):
        return self.call_endpoint("get_action", obs)


def write_video(path, frames, fps=10):
    import av
    if not frames:
        return
    c = av.open(path, mode="w")
    s = c.add_stream("h264", rate=fps)
    s.height, s.width = frames[0].shape[:2]
    s.pix_fmt = "yuv420p"
    s.options = {"crf": "26"}
    for f in frames:
        for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def batch(obs):
    out = {}
    for k, v in obs.items():
        out[k] = (v,) if isinstance(v, str) else np.asarray(v)[None]
    return out


def run_episode(client, task, seed, horizon, out_dir, tag, instr_fn=None, reset_fn=None,
                n_action_steps=16, save_video=True, env=None):
    """instr_fn(raw_env, step, default_instr) -> instruction string (None = keep default).
    reset_fn(base_env) -> called after env.reset(seed) to overwrite the sim state (oracle reset)."""
    t0 = time.time()
    base = env if env is not None else gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
    menv = MultiStepWrapper(base, video_delta_indices=np.array([0]), state_delta_indices=np.array([0]),
                            n_action_steps=n_action_steps, max_episode_steps=horizon)
    obs, _ = menv.reset(seed=seed)
    if reset_fn is not None:
        raw = reset_fn(base)
        o = base.get_observation(raw)
        menv.obs.clear(); [menv.obs.append(o) for _ in range(menv.max_steps_needed + 1)]
        obs = menv._get_obs(menv.video_delta_indices, menv.state_delta_indices)
    kenv = base.env
    ep_meta = kenv.get_ep_meta()
    default_instr = obs["annotation.human.task_description"]
    states, frames, instr_log = [], [], []
    success, first_success_step, steps = False, None, 0
    done = False
    while not done:
        instr = default_instr
        if instr_fn is not None:
            i2 = instr_fn(kenv, steps, default_instr)
            if i2 is not None:
                instr = i2
        if not instr_log or instr_log[-1][1] != instr:
            instr_log.append((steps, instr))
        obs["annotation.human.task_description"] = instr
        act = client.get_action(batch(obs))
        act = {k: np.asarray(v)[0] for k, v in act.items()}
        states.append(kenv.sim.get_state().flatten().astype(np.float32))
        obs, r, done, trunc, info = menv.step(act)
        n = len(info["rewards"])
        if save_video:
            fr = obs["video.robot0_agentview_left"][-1], obs["video.robot0_eye_in_hand"][-1]
            frames.append(np.concatenate(fr, axis=1))
        if np.any(info["rewards"] > 0):
            success = True
            first_success_step = steps + int(np.argmax(info["rewards"] > 0)) + 1
        steps += n
        if success:
            break
    states.append(kenv.sim.get_state().flatten().astype(np.float32))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{tag}_states.npz"), states=np.stack(states),
                        chunk=n_action_steps)
    with gzip.open(os.path.join(out_dir, f"{tag}_model.xml.gz"), "wt") as f:
        f.write(kenv.sim.model.get_xml())
    if save_video:
        write_video(os.path.join(out_dir, f"{tag}.mp4"), frames, fps=max(1, 20 // n_action_steps * 4))
    res = dict(task=task, seed=seed, horizon=horizon, success=bool(success), steps=int(steps),
               first_success_step=first_success_step, wall_s=round(time.time() - t0, 1),
               instruction=default_instr, instr_log=instr_log,
               layout_id=ep_meta.get("layout_id"), style_id=ep_meta.get("style_id"))
    with open(os.path.join(out_dir, f"{tag}_ep_meta.json"), "w") as f:
        json.dump(ep_meta, f, default=str)
    if env is None:
        base.close()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="json file: list of {task, seed, horizon, tag}")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=5555)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))[a.worker::a.n_workers]
    client = Client(port=a.port, timeout_ms=600000)
    res_path = os.path.join(a.out, f"results_w{a.worker}.jsonl")
    import glob as _glob
    done_tags = {json.loads(l)["tag"] for p_ in _glob.glob(os.path.join(a.out, "results_w*.jsonl")) for l in open(p_)
                 if "error" not in json.loads(l)}
    for j in jobs:
        if j["tag"] in done_tags:
            continue
        try:
            r = run_episode(client, j["task"], j["seed"], j["horizon"], os.path.join(a.out, j["task"]), j["tag"])
        except Exception as e:
            r = dict(task=j["task"], seed=j["seed"], horizon=j["horizon"], error=repr(e), tb=traceback.format_exc()[-2000:])
        r.update(tag=j["tag"], condition=j.get("condition", "baseline"))
        with open(res_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps({k: r.get(k) for k in ("tag", "success", "steps", "wall_s", "error")}), flush=True)
