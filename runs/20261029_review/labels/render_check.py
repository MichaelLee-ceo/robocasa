"""Is the first observation of a decision point the same for all its rollouts? Every rollout's row carries obs_fp, the
fingerprint of its first request (3 camera images, proprio, instruction). All rollouts of a point start from the same sim
state and the same restored controller state, so obs_fp must be equal unless the render differs.
  (1) per run dir: points with 1, 2-3, >= 4 distinct obs_fp among their rollouts;
  (2) start states (states[0]) across the rollouts of a point, and states over the first chunk of the two continuations
      of a candidate (same first chunk), on a sample: bit-identical means the physics and the restore are deterministic;
  (3) the same (point, candidate seed) collected in two runs: is the first chunk identical? (it is a deterministic function of
      the observation and the seed), and how large is the difference when not.
Writes render_check.json. usage: python render_check.py"""
import collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, TASKS3, coarse, dense, read_rows, rolls, roll_path  # noqa: E402

PAIRS = [("run_SteamInMicrowave_v2", "run_SteamInMicrowave_fix3"), ("run_SteamInMicrowave_v2", "run_SteamInMicrowave_fix"),
         ("run_PackIdenticalLunches", "run_PackIdenticalLunches_fix3"), ("run_PackIdenticalLunches", "run_PackIdenticalLunches_fix"),
         ("run_dense_SteamInMicrowave_bug3", "run_dense_SteamInMicrowave"), ("run_dense_PackIdenticalLunches_bug3", "run_dense_PackIdenticalLunches"),
         ("run_v2/s*", "run_confirm_KettleBoiling"), ("run_dense_KettleBoiling", "run_confirm_KettleBoiling")]


def main():
    out = {"distinct_obs_fp": {}, "determinism": [], "pairs": {}}
    for task in TASKS3:
        for src, rows in (("coarse", coarse(task)), ("dense", dense(task))):
            by = collections.defaultdict(list)
            for r in rows.values():
                by[(r["_dir"].split("/")[0], str(r.get("hidden")))].append(len({x["obs_fp"] for _, _, x in rolls(r)}))
            for k, v in by.items():
                v = np.array(v)
                out["distinct_obs_fp"][f"{task}|{k[0]}|{k[1]}"] = dict(points=len(v), one=int((v == 1).sum()), two_three=int(((v > 1) & (v < 4)).sum()),
                                                                       four_plus=int((v >= 4).sum()), max=int(v.max()))
                print(f"{task:21s} {k[0]:34s} {k[1]:16s} points {len(v):4d}: 1 distinct obs_fp {int((v == 1).sum()):4d}, 2-3 {int(((v > 1) & (v < 4)).sum()):4d}, >=4 {int((v >= 4).sum()):4d}")
    # (2) determinism of the physics and the restore, on coarse rows of the pool runs (M = 2)
    for d in ("run_SteamInMicrowave_fix3", "run_PackIdenticalLunches", "run_PackIdenticalLunches_fix3"):
        rows = read_rows(d)[:10]
        for r in rows:
            st0, pair = [], []
            for c in range(8):
                a, b = np.load(roll_path(r, c, 0)), np.load(roll_path(r, c, 1))
                sa, sb = a["states"], b["states"]
                st0 += [sa[0], sb[0]]
                n = min(17, len(sa), len(sb))
                pair.append(float(np.abs(sa[:n] - sb[:n]).max()))
            out["determinism"].append(dict(dir=d, tag=r["tag"], t=r["t"], distinct_obs_fp=len({x["obs_fp"] for _, _, x in rolls(r)}),
                                           start_diff=max(float(np.abs(s - st0[0]).max()) for s in st0), first_chunk_m0_vs_m1=max(pair)))
    dd = out["determinism"]
    print(f"determinism sample: {len(dd)} points; max start-state diff across rollouts {max(x['start_diff'] for x in dd)}; "
          f"max state diff over the shared first chunk (m0 vs m1) {max(x['first_chunk_m0_vs_m1'] for x in dd)}; "
          f"distinct obs_fp at these points: {[x['distinct_obs_fp'] for x in dd]}")
    # (3) same seed, two runs
    for a, b in PAIRS:
        A = {(r["tag"], r["t"]): r for r in read_rows(a)}; B = {(r["tag"], r["t"]): r for r in read_rows(b)}
        diffs, same, fp_same, n = [], 0, 0, 0
        flips = collections.Counter()
        for key in sorted(set(A) & set(B))[:40]:
            ra, rb = A[key], B[key]
            for c in range(8):
                pa, pb = roll_path(ra, c, 0), roll_path(rb, c, 0)
                if not (os.path.exists(pa) and os.path.exists(pb)):
                    continue
                za, zb = np.load(pa), np.load(pb)
                dk = {k: float(np.abs(za[k][0] - zb[k][0]).max()) for k in za.files if k.startswith("action.")}
                d = max(dk.values())
                diffs.append(d); n += 1; same += d == 0
                for k in ("action.gripper_close", "action.control_mode"):
                    flips[k] += dk[k] >= 0.5
                fa = next(x for cc, m, x in rolls(ra) if (cc, m) == (c, 0))["obs_fp"]
                fb = next(x for cc, m, x in rolls(rb) if (cc, m) == (c, 0))["obs_fp"]
                fp_same += fa == fb
        if n:
            out["pairs"][f"{a} vs {b}"] = dict(cands=n, identical=same, obs_fp_equal=fp_same, median=float(np.median(diffs)),
                                               p90=float(np.percentile(diffs, 90)), flips=dict(flips))
            print(f"{a:36s} vs {b:32s} cands {n:4d}: first chunk identical {same}/{n}, obs_fp equal {fp_same}/{n}, "
                  f"max |action diff| median {np.median(diffs):.3g} p90 {np.percentile(diffs, 90):.3g}; binary flips {dict(flips)}")
    json.dump(out, open(os.path.join(HERE, "render_check.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
