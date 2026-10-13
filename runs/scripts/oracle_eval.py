"""Oracle experiments for GR00T N1.5 on RoboCasa365 composite tasks. All conditions start from demo states
of the target dataset so they are paired on identical scenes.

conditions:
  baseline_demoinit : demo frame 0, original task instruction, full (original) horizon; subtask tracker only logs.
  planner           : demo frame 0, instruction = current oracle subtask instruction (switches when tracker
                      marks the subtask done), full horizon.
  reset_sub         : for each subtask k, reset to demo frame at start of k, subtask instruction, budget
                      = clip(3 x demo segment length, 240, horizon); success = tracker predicate for k.
  reset_task        : same as reset_sub but with the original task-level instruction.
"""
import argparse, json, os, sys, time, traceback
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/Desktop/robocasa/third_party/Isaac-GR00T"))
import robocasa  # noqa
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from groot_eval import Client, batch, write_video
from demo_utils import load_demo
from oracle import reset_to, set_state, build_subtask_specs, SubtaskTracker, SceneProbe
import robocasa.utils.object_utils as OU


def eef_pos(kenv):
    return kenv.sim.data.site_xpos[kenv.robots[0].eef_site_id["right"]].copy()


def rollout(client, base, kenv, specs, P, start_state, horizon, mode, task_instr, k0=0, k_stop=None,
            n_action_steps=16, frames=None):
    """mode: 'task' (task instr) or 'sub' (oracle subtask instr). Tracker starts at subtask k0;
    stops when tracker index reaches k_stop (None = all) or env success or horizon."""
    menv = MultiStepWrapper(base, video_delta_indices=np.array([0]), state_delta_indices=np.array([0]),
                            n_action_steps=n_action_steps, max_episode_steps=horizon)
    from collections import defaultdict, deque
    set_state(kenv, start_state)
    o = base.get_observation(kenv._get_observations(force_update=True))
    menv.obs = deque([o] * (menv.max_steps_needed + 1), maxlen=menv.max_steps_needed + 1)
    menv.reward, menv.done = [], []
    menv.info = defaultdict(lambda: deque(maxlen=menv.max_steps_needed + 1))
    obs = menv._get_obs(menv.video_delta_indices, menv.state_delta_indices)
    tr = SubtaskTracker(kenv, specs, P)
    tr.k = k0; tr._arm()
    log = []
    steps, success, done = 0, False, False
    while not done:
        instr = task_instr if mode == "task" else tr.instr
        obs["annotation.human.task_description"] = instr
        act = client.get_action(batch(obs))
        act = {kk: np.asarray(v)[0] for kk, v in act.items()}
        obs, r, done, trunc, info = menv.step(act)
        steps += len(info["rewards"])
        if np.any(info["rewards"] > 0):
            success = True
        k = tr.update(steps)
        tgt = specs[min(k, len(specs) - 1)].get("obj")
        grasped = [n for n in P.obj_names if P.grasped(n)]
        log.append(dict(step=steps, k=k, grasped=grasped,
                        d_tgt=None if tgt is None else float(np.linalg.norm(eef_pos(kenv) - P.obj_pos()[tgt]))))
        if frames is not None:
            frames.append(np.concatenate([obs["video.robot0_agentview_left"][-1], obs["video.robot0_eye_in_hand"][-1]], 1))
        if success or (k_stop is not None and k >= k_stop):
            break
    # undo check: re-evaluate completed subtasks' predicates at the end
    undone = []
    for (_, kk) in tr.history:
        tmp = SubtaskTracker(kenv, specs, P); tmp.anchor = tr.anchor
        if specs[kk]["stage"] == "place" and not tmp.done_k(kk):
            undone.append(kk)
    return dict(success=bool(success), steps=int(steps), reached=int(tr.k), n_sub=len(specs),
                history=tr.history, undone=undone, log=log[::2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=5555)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))[a.worker::a.n_workers]
    client = Client(port=a.port, timeout_ms=600000)
    res_path = os.path.join(a.out, f"results_w{a.worker}.jsonl")
    done_tags = {json.loads(l)["tag"] for l in open(res_path)} if os.path.exists(res_path) else set()
    envs = {}
    for j in jobs:
        if j["tag"] in done_tags:
            continue
        t0 = time.time()
        try:
            task = j["task"]
            if task not in envs:
                for e in envs.values():
                    e.close()
                envs = {task: gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped}
            base = envs[task]; kenv = base.env
            d = load_demo(task, j["demo"])
            reset_to(kenv, d["xml"], d["ep_meta"])
            specs, P = build_subtask_specs(kenv, d)
            # rebuild the scene: replaying demo states while building specs calls fixture.update_state(), which can leave
            # Python-level fixture state behind (e.g. Microwave._turned_on=True after a frame where the demo presses start)
            reset_to(kenv, d["xml"], d["ep_meta"])
            P = SceneProbe(kenv)
            frames = [] if j.get("video", True) else None
            cond = j["condition"]
            if cond in ("baseline_demoinit", "planner"):
                r = rollout(client, base, kenv, specs, P, d["states"][0], j["horizon"],
                            "task" if cond == "baseline_demoinit" else "sub", d["task_instr"], frames=frames)
            else:
                k = j["subtask"]; sp = specs[k]
                budget = int(np.clip(3 * sp["demo_len"], 240, j["horizon"]))
                r = rollout(client, base, kenv, specs, P, d["states"][sp["demo_start"]], budget,
                            "sub" if cond == "reset_sub" else "task", d["task_instr"], k0=k, k_stop=k + 1,
                            frames=frames)
                r["subtask_success"] = bool(r["reached"] > k or (sp["last"] and r["success"]))
                r.update(subtask=k, budget=budget, sub_instr=sp["instr"], sub_stage=sp["stage"], sub_skill=sp["skill"])
            r.update(specs=[{kk: v for kk, v in s.items() if kk in ("i", "instr", "skill", "stage", "obj", "demo_len", "new_support_names", "joints")} for s in specs])
            if frames:
                os.makedirs(os.path.join(a.out, task), exist_ok=True)
                write_video(os.path.join(a.out, task, f"{j['tag']}.mp4"), frames, fps=5)
            r.update(task_instr=d["task_instr"], demo_len=d["length"])
        except Exception as e:
            r = dict(error=repr(e), tb=traceback.format_exc()[-2000:])
        r.update(j); r["wall_s"] = round(time.time() - t0, 1)
        with open(res_path, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        print(json.dumps({k: r.get(k) for k in ("tag", "success", "subtask_success", "reached", "n_sub", "steps", "wall_s", "error")}), flush=True)


if __name__ == "__main__":
    main()
