"""Is the camera rendering of a sim state reproducible? Runs KettleBoiling s1 (policy_seed 0, frozen scene) through
atomic_eval.rollout against a seeded server, with every camera render done twice on the same state and compared.
Writes <out>.json: per-chunk fingerprints (same format as atomic_eval's "digest") and every double-render mismatch.
usage: python -I render_check.py <port> <n_chunks> <out prefix>"""
import json, os, sys
import numpy as np
import gymnasium as gym
R = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{R}/runs/scripts")
import atomic_eval as A  # noqa: E402

port, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
task, tag = "KettleBoiling", "KettleBoiling_s1_single"
base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
kenv = base.env
start = A.load_scene(kenv, f"{R}/runs/scenes_seen5_v2/KettleBoiling_s1.npz")
orig, dig, mism, n_renders = kenv.sim.render, [], [], [0]


def render_twice(*a, **k):
    x = orig(*a, **k)
    y = orig(*a, **k)
    n_renders[0] += 1
    if not np.array_equal(x, y):
        d = np.abs(np.asarray(x, dtype=np.int32) - np.asarray(y, dtype=np.int32))
        mism.append(dict(chunk=len(dig), cam=k.get("camera_name"), n_pixels=int((d.reshape(d.shape[0], d.shape[1], -1).max(-1) > 0).sum()),
                         max_abs=int(d.max())))
    return x


kenv.sim.render = render_twice
client = A.Client(port=port, timeout_ms=600000)
r, _ = A.rollout(client, base, kenv, {}, A.TASKS[task], start, 16 * n, "task", kenv.get_ep_meta()["lang"],
                 seed_fn=A.policy_seed_fn(tag, 0), digest=dig)
json.dump(dict(digest=dig, mismatches=mism, n_renders=n_renders[0], success=r["success"], history=r["history"]),
          open(f"{out}.json", "w"))
print(json.dumps(dict(out=out, chunks=len(dig), renders=n_renders[0], mismatches=len(mism), first=mism[:5])), flush=True)
