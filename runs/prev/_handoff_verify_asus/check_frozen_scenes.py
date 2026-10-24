"""README §5.1 on a new server: reload every frozen scene in runs/scenes_seen5_v2 with atomic_eval.load_scene()
and compare layout / style / nq / objects / phys_sha1 with the fingerprint stored in the file.
On a phys_sha1-only mismatch, dump the raw arrays (body_mass, body_inertia, body_ipos, geom_friction) for comparison.
usage: python check_frozen_scenes.py --out <dir> [--shard i/n]"""
import argparse, glob, json, os, sys, traceback
import numpy as np
import gymnasium as gym

ROOT = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa"))
sys.path.insert(0, f"{ROOT}/runs/scripts")
from atomic_eval import scene_fp  # noqa: E402
from oracle import reset_to  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--scenes", default=f"{ROOT}/runs/scenes_seen5_v2")
ap.add_argument("--shard", default="0/1", help="i/n: this process checks tasks i, i+n, ... (sorted)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
paths = sorted(glob.glob(f"{a.scenes}/*.npz"))
_i, _n = map(int, a.shard.split("/"))
_tasks = sorted({os.path.basename(p)[:-4].rsplit("_s", 1)[0] for p in paths})[_i::_n]
paths = [p for p in paths if os.path.basename(p)[:-4].rsplit("_s", 1)[0] in _tasks]
rows, envs = [], {}
for p in paths:
    name = os.path.basename(p)[:-4]; task = name.rsplit("_s", 1)[0]
    row = dict(scene=name, task=task)
    try:
        if task not in envs:
            for e in envs.values():
                e.close()
            envs = {task: gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped}
        kenv = envs[task].env
        z = np.load(p, allow_pickle=True)
        reset_to(kenv, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); kenv.unset_ep_meta()
        ref, got = json.loads(str(z["fp"])), scene_fp(kenv)
        row["mismatch"] = [k for k in ("layout", "style", "nq", "objects", "phys_sha1") if got[k] != ref[k]]
        row["ref"] = {k: ref[k] for k in row["mismatch"]}; row["got"] = {k: got[k] for k in row["mismatch"]}
        if row["mismatch"] == ["phys_sha1"]:
            mm = kenv.sim.model
            np.savez_compressed(f"{a.out}/{name}_phys.npz", body_mass=mm.body_mass, body_inertia=mm.body_inertia,
                                body_ipos=mm.body_ipos, geom_friction=mm.geom_friction)
    except Exception as e:
        row["error"] = repr(e); row["tb"] = traceback.format_exc()[-1500:]
    rows.append(row)
    print(json.dumps({k: row.get(k) for k in ("scene", "mismatch", "error")}), flush=True)
json.dump(rows, open(f"{a.out}/scene_check_{_i}of{_n}.json", "w"), indent=1)
ok = sum(1 for r in rows if not r.get("error") and not r.get("mismatch"))
print(f"{ok}/{len(rows)} identical; errors {sum('error' in r for r in rows)}; "
      f"phys_sha1-only {sum(r.get('mismatch') == ['phys_sha1'] for r in rows)}; other mismatches "
      f"{sum(bool(r.get('mismatch')) and r['mismatch'] != ['phys_sha1'] for r in rows)}")
