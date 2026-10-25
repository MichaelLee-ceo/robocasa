"""Checks that a rerun episode's saved per-step states are usable: replays <tag>_steps.npz in a fresh env built like the
rollout's (gym robocasa/<task>, split target -- the env that generated the frozen scene; the dataset's env_args give a
different set of counter accessories), scene restored with oracle.reset_to, and
  1. the stage tracker, evaluated at chunk boundaries (every 16 control steps) as in the rollout, reproduces the
     online stage history, and the last saved state equals <tag>_end.npz;
  2. the official subtask labeler runs on the rollout (label_episode from add_subtask_labels_to_lerobot.py).
usage: python -I replay_check.py <run dir> <tag> [<tag> ...]"""
import glob, json, os, sys
import numpy as np
import gymnasium as gym
R = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{R}/runs/scripts")
import robocasa.scripts.dataset_scripts.add_subtask_labels_to_lerobot as L  # noqa: E402
from robocasa.utils.subtask_labeling import get_labeler  # noqa: E402
from oracle import reset_to, set_state  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
import atomic_eval  # noqa: E402,F401  (registers the robocasa gym envs, as in the rollout)

run, tags = sys.argv[1], sys.argv[2:]
res = {}
for p in glob.glob(f"{run}/results_w*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        res[r["tag"]] = r
for tag in tags:
    r = res[tag]
    task = r["task"]
    z = np.load(r["scene"], allow_pickle=True)
    xml, ep_meta = str(z["xml"]), json.loads(str(z["ep_meta"]))
    states = np.load(f"{run}/{task}/{tag}_steps.npz")["states"]
    end = np.load(f"{run}/{task}/{tag}_end.npz", allow_pickle=True)["state"]
    env = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped.env
    # 1. stage tracker replay
    reset_to(env, xml, ep_meta)
    steps, S, k, hist = TASKS[task], {}, 0, []
    for f in range(len(states)):
        set_state(env, states[f])
        env._check_success()
        if f > 0 and (f % 16 == 0 or f == len(states) - 1):
            while k < len(steps) and steps[k]["check"](env, S):
                hist.append([f, k]); k += 1
    print(f"{tag}: {len(states)} states for {r['steps']} steps | online history {r['history']} | replayed {hist} | "
          f"same: {hist == r['history']} | last state == end.npz: {np.array_equal(states[-1], end)}", flush=True)
    # 2. official labeler on the rollout
    labeler = get_labeler(task)
    out = L.label_episode(env, labeler, states, ep_meta, xml)
    idx = np.asarray(out.subtask_idx)
    starts = np.r_[0, np.nonzero(np.diff(idx))[0] + 1]
    segs = [(int(s), str(out.subtask_name[s]), str(out.subtask_stage[s])) for s in starts]
    print(f"   official labeler segments (start frame, subtask, stage): {segs}\n   validate(): {labeler.validate(out)}", flush=True)
    env.close()
