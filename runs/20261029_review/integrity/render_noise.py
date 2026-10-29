"""Within-point rendering determinism: continuations m0/m1 of one candidate have bit-identical states for steps 0..16,
so the video frames of steps 4, 8, 12, 16 (frames 0-3 of the rollout mp4) are renders of identical states. Decodes them
and counts differing pixels, next to whether the two continuations' first-request obs_fp agree.
usage: python -I render_noise.py [n_points=40]"""
import json, os, sys, random
import numpy as np
import imageio.v3 as iio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, HERE  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
recs = [json.loads(l) for l in open(os.path.join(HERE, "scan.jsonl"))]
recs = [r for r in recs if r.get("status", "") != "x" and r["M"] >= 2 and r["dir"] in
        ("run_SteamInMicrowave_fix3", "run_PackIdenticalLunches_fix3", "run_PackIdenticalLunches", "run_v2/s0", "run_SteamInMicrowave_v2")]
random.Random(0).shuffle(recs)
res = []
for r in recs[:n]:
    for i in range(2):
        a = [o for o in r["rolls"] if o["i"] == i and o["m"] == 0][0]
        b = [o for o in r["rolls"] if o["i"] == i and o["m"] == 1][0]
        if a["steps"] < 16 or b["steps"] < 16:
            continue
        fa = iio.imread(f"{DATA}/{r['dir']}/roll/{r['tag']}_t{r['t']}_c{i}_m0.mp4", index=None)[:4]
        fb = iio.imread(f"{DATA}/{r['dir']}/roll/{r['tag']}_t{r['t']}_c{i}_m1.mp4", index=None)[:4]
        d = [int((np.abs(x.astype(int) - y.astype(int)).max(-1) > 0).sum()) for x, y in zip(fa, fb)]
        res.append(dict(dir=r["dir"], tag=r["tag"], t=r["t"], i=i, obs_fp_equal=a["obs_fp"] == b["obs_fp"], diff_px_frames_0_3=d,
                        frame_shape=list(fa.shape[1:])))
        print(json.dumps(res[-1]), flush=True)
json.dump(res, open(os.path.join(HERE, "render_noise.json"), "w"), indent=0)
eq = [x for x in res if all(v == 0 for v in x["diff_px_frames_0_3"])]
print(f"{len(res)} pairs: identical first 4 frames {len(eq)}; obs_fp equal {sum(x['obs_fp_equal'] for x in res)}; "
      f"both {sum(x['obs_fp_equal'] for x in eq)}")
