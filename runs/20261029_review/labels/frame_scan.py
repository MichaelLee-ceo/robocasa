"""Rendering check without a GPU: the rollouts' first observations differ between rollouts of the same decision point
(obs_fp) although their sim states are bit-identical, i.e. the local renders are not deterministic. Are some rendered
observations corrupted (black, blank, or garbage frames)? The rollout videos are the policy's own observation images
(agentview left + wrist, every 4th step). For a random sample of rollouts per run dir, every frame is checked for:
  dark  : mean intensity < 8;   flat : std < 3;
  spike : the frame differs from both its neighbours by more than 3x the neighbours' difference and by > 25 levels.
usage: python frame_scan.py <n per dir>"""
import glob, json, os, random, sys
import numpy as np
import imageio.v3 as iio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, HERE  # noqa: E402

DIRS = ["run_v2/s0", "run_SteamInMicrowave_v2", "run_SteamInMicrowave_fix3", "run_PackIdenticalLunches", "run_PackIdenticalLunches_fix3",
        "run_dense_SteamInMicrowave", "run_dense_PackIdenticalLunches", "run_dense2_SteamInMicrowave", "run_dense2_PackIdenticalLunches",
        "run_dense_KettleBoiling"]


def main():
    n = int(sys.argv[1])
    rng = random.Random(0)
    out = {}
    for d in DIRS:
        vids = sorted(glob.glob(f"{DATA}/{d}/roll/*.mp4"))
        rng.shuffle(vids)
        bad, frames = [], 0
        for v in vids[:n]:
            try:
                fr = iio.imread(v, plugin="pyav")
            except Exception as e:
                bad.append((os.path.basename(v), "unreadable", repr(e)[:80])); continue
            fr = fr.astype(np.float32)
            frames += len(fr)
            m = fr.mean(axis=(1, 2, 3)); s = fr.std(axis=(1, 2, 3))
            for i in range(len(fr)):
                if m[i] < 8:
                    bad.append((os.path.basename(v), i, "dark", round(float(m[i]), 1)))
                elif s[i] < 3:
                    bad.append((os.path.basename(v), i, "flat", round(float(s[i]), 1)))
            if len(fr) >= 3:
                d1 = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2, 3))  # d1[i] = |f[i+1] - f[i]|
                for i in range(1, len(fr) - 1):
                    nb = np.abs(fr[i + 1] - fr[i - 1]).mean()
                    if d1[i - 1] > 25 and d1[i] > 25 and min(d1[i - 1], d1[i]) > 3 * max(nb, 1):
                        bad.append((os.path.basename(v), i, "spike", round(float(d1[i - 1]), 1), round(float(nb), 1)))
        out[d] = dict(videos=min(n, len(vids)), frames=frames, bad=bad)
        print(f"{d:34s} videos {min(n, len(vids)):4d} frames {frames:6d} anomalies {len(bad)}  {bad[:4]}", flush=True)
    json.dump(out, open(os.path.join(HERE, "frame_scan.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
