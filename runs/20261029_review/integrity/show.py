"""Reproduce single findings from the saved files (read-only).
  python -I show.py gripper <dir> <tag> <t> [n_cands=3]   finger gap (mm) of the main run and of rollouts c<i>m0, steps 0..16,
                                                          with each candidate's first gripper commands
  python -I show.py joint <dir> <tag> <t> <joint name>    a joint's qpos at the start / max / end of every rollout vs the main run
  python -I show.py snapdiff <dirA> <dirB> <tag> <t>      per-key comparison of the two dirs' saved first request (snap req/*)
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, SEEN, HERE, scenes  # noqa: E402

TASK = lambda tag: tag.split("_s")[0]


def main_states(tag):
    return np.load(f"{SEEN}/run/{TASK(tag)}/{tag}_steps.npz")["states"]


def rows(d, tag, t):
    import glob
    for p in glob.glob(f"{DATA}/{d}/results_w*.jsonl"):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("tag") == tag and r.get("t") == t and "error" not in r:
                return r


def gripper(d, tag, t, n=3):
    ms = main_states(tag); g = lambda s: np.abs(s[..., 12] - s[..., 13]) * 1000  # finger joints: qpos 11, 12
    r = rows(d, tag, t)
    print(f"{d} {tag} t{t} hidden={r.get('hidden')} k={r['k']}")
    print("main run  gap t-4..t:", np.round(g(ms[max(t - 4, 0):t + 1]), 1), " t..t+16:", np.round(g(ms[t:t + 17]), 1))
    for c in r["cands"][:n]:
        z = np.load(f"{DATA}/{d}/roll/{tag}_t{t}_c{c['cand']}_m0.npz")
        print(f"c{c['cand']}m0 gap 0..16:", np.round(g(z["states"][:17]), 1), " cmd close (first 8):",
              z["action.gripper_close"][0, :8, 0], " done:", [x["done"] for x in c["rollouts"]])


def joint(d, tag, t, name):
    lay = json.load(open(os.path.join(HERE, "layout.json")))[scenes()[tag]]
    q = 1 + lay["joints"][name]["qpos"]
    ms = main_states(tag)
    r = rows(d, tag, t)
    print(f"main run {name} at t: {ms[t, q]:.4f}; max over the main run {ms[:, q].max():.4f}")
    for c in r["cands"]:
        for x in c["rollouts"]:
            s = np.load(f"{DATA}/{d}/roll/{tag}_t{t}_c{c['cand']}_m{x['m']}.npz")["states"]
            print(f"c{c['cand']}m{x['m']} start {s[0, q]:.4f} max {s[:, q].max():.4f} end {s[-1, q]:.4f}  states[0]==main[t]: {np.array_equal(s[0], ms[t])}")


def snapdiff(da, db, tag, t):
    za = np.load(f"{DATA}/{da}/snap/{tag}_t{t}.npz", allow_pickle=True)
    zb = np.load(f"{DATA}/{db}/snap/{tag}_t{t}.npz", allow_pickle=True)
    for k in sorted(za.files):
        if not k.startswith(("req/", "rgb/", "depth", "state")):
            continue
        a, b = za[k], zb[k]
        if a.dtype.kind in "OUS":
            print(k, "equal" if np.array_equal(a, b) else "differs"); continue
        d = np.abs(a.astype(float) - b.astype(float))
        px = (d.max(-1) > 0).sum() if d.shape[-1] == 3 else (d > 0).sum()
        print(f"{k:45s} {'equal' if not d.any() else f'{px} px/elements differ, max {d.max():.3g}'}")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "gripper":
        gripper(a[0], a[1], int(a[2]), int(a[3]) if len(a) > 3 else 3)
    elif cmd == "joint":
        joint(a[0], a[1], int(a[2]), a[3])
    elif cmd == "snapdiff":
        snapdiff(a[0], a[1], a[2], int(a[3]))
