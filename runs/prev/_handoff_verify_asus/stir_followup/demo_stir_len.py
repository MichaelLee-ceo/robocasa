"""Item 5: training-demo length of the segment(s) after PickUpSpatula ends (StirVegetables segment). No sim."""
import sys, os, glob, json, collections
import numpy as np, pandas as pd
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
from demo_utils import ds_dir, load_tasks
d = ds_dir("StirVegetables"); tasks = load_tasks(d)
rows = []; seqcount = collections.Counter()
for f in sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet")):
    ep = int(f.split("_")[-1].split(".")[0])
    df = pd.read_parquet(f, columns=["annotation.human.subtask_name", "annotation.human.subtask_stage"])
    names = [tasks[int(x)] for x in df["annotation.human.subtask_name"]]
    stg = [tasks[int(x)] for x in df["annotation.human.subtask_stage"]]
    segs = []
    for i, (n, g) in enumerate(zip(names, stg)):
        if not segs or (n, g) != (segs[-1][0], segs[-1][1]): segs.append([n, g, i, i + 1])
        else: segs[-1][3] = i + 1
    seqcount[tuple(f"{n}:{g}" for n, g, _, _ in segs)] += 1
    ps = [s for s in segs if s[0] == "PickUpSpatula"]
    meta = json.load(open(f"{d}/extras/episode_{ep:06d}/ep_meta.json"))
    if not ps:
        rows.append(dict(ep=ep, layout=meta.get("layout_id"), style=meta.get("style_id"), n=len(df), after=None)); continue
    end = ps[-1][3]
    after = [s for s in segs if s[2] >= end]
    rows.append(dict(ep=ep, layout=meta.get("layout_id"), style=meta.get("style_id"), n=len(df), pick_spatula_end=end,
                     after=len(df) - end, after_segs=[f"{n}:{g}:{e-s}" for n, g, s, e in after]))
def st(v): v = np.array(v); return dict(n=len(v), min=int(v.min()), median=float(np.median(v)), max=int(v.max()), mean=round(float(v.mean()), 1))
ok = [r for r in rows if r["after"] is not None]
summ = dict(dataset=d, n_demos=len(rows), n_without_PickUpSpatula=len(rows) - len(ok), overall=st([r["after"] for r in ok]),
            overall_total_len=st([r["n"] for r in rows]),
            per_layout={str(L): st([r["after"] for r in ok if r["layout"] == L]) for L in sorted({r["layout"] for r in ok})},
            exec_only_overall=st([sum(int(x.rsplit(":", 1)[1]) for x in r["after_segs"] if x.startswith("StirVegetables:execute")) for r in ok]),
            exec_only_per_layout={str(L): st([sum(int(x.rsplit(":", 1)[1]) for x in r["after_segs"] if x.startswith("StirVegetables:execute")) for r in ok if r["layout"] == L]) for L in sorted({r["layout"] for r in ok})},
            after_seg_names=collections.Counter(",".join(x.rsplit(":", 1)[0] for x in r["after_segs"]) for r in ok).most_common(10),
            top_stage_sequences=[[list(k), v] for k, v in seqcount.most_common(3)])
print(json.dumps(summ, indent=1))
json.dump(dict(summary=summ, demos=rows), open(sys.argv[1], "w"), indent=1)
