"""Read-only subtask-annotation audit for the 11 remaining composite-seen tasks (same approach as runs/scripts/subtask_audit.py).
Two sequence views per demo:
  * name_seq : drop NavigateKitchen, merge adjacent segments with the same subtask_name (exactly as subtask_audit.py)
  * unit_seq : same, but a repeated pick->place of the same skill counts as a new step (a 'pick' stage after a 'place'/'execute' stage
               of the same name starts a new unit) -- needed because e.g. two consecutive PickPlaceCounterToSink collapse otherwise."""
import sys, os, glob, json, collections
import pandas as pd
sys.path.insert(0, os.path.expanduser(os.environ["ROBOCASA_ROOT"] + "/runs/scripts"))
from demo_utils import ds_dir, load_tasks
TASKS11 = ["ScrubCuttingBoard", "WashLettuce", "RinseSinkBasin", "LoadDishwasher", "StoreLeftoversInBowl", "SetUpCuttingStation",
           "StirVegetables", "SearingMeat", "PackIdenticalLunches", "DeliverStraw", "GetToastedBread"]
out = {}
for t in TASKS11:
    d = ds_dir(t); tasks = load_tasks(d)
    files = sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet"))
    seqs = collections.Counter(); useqs = collections.Counter(); stage_seqs = collections.Counter()
    sent = collections.defaultdict(collections.Counter); per_demo = {}
    for f in files:
        ep = int(f.split("_")[-1].split(".")[0])
        df = pd.read_parquet(f, columns=["annotation.human.subtask_name", "annotation.human.subtask", "annotation.human.subtask_stage"])
        names = [tasks[int(x)] for x in df["annotation.human.subtask_name"]]
        subs = [tasks[int(x)] for x in df["annotation.human.subtask"]]
        stg = [tasks[int(x)] for x in df["annotation.human.subtask_stage"]]
        segs = []
        for n, s, g in zip(names, subs, stg):
            if not segs or (n, s, g) != segs[-1]:
                segs.append((n, s, g))
        for n, s, g in segs:
            sent[(n, g)][s] += 1
        m = []
        for n, _, _ in segs:
            if n == "NavigateKitchen": continue
            if not m or m[-1] != n: m.append(n)
        u = []; last_stage = None
        for n, _, g in segs:
            if n == "NavigateKitchen": continue
            if not u or u[-1] != n or (g == "pick" and last_stage in ("place", "execute")):
                u.append(n)
            last_stage = g
        seqs[tuple(m)] += 1; useqs[tuple(u)] += 1
        stage_seqs[tuple(f"{n}:{g}" for n, _, g in segs)] += 1
        per_demo[ep] = u
    maj = list(useqs.most_common(1)[0][0])
    out[t] = dict(n=len(files), sequences=[[list(k), v] for k, v in seqs.most_common()],
                  unit_sequences=[[list(k), v] for k, v in useqs.most_common()],
                  majority_unit_sequence=maj, majority_share=round(useqs.most_common(1)[0][1] / len(files), 4),
                  stage_sequences=[[list(k), v] for k, v in stage_seqs.most_common(4)],
                  n_stage_sequences=len(stage_seqs),
                  sentences={f"{n} | {g}": c.most_common(4) for (n, g), c in sent.items()},
                  nonmajority_demos=[ep for ep, u in per_demo.items() if u != maj][:50])
    print(f"== {t}: {len(files)} demos; name-merged seqs: {len(seqs)}; unit seqs: {len(useqs)}; stage seqs: {len(stage_seqs)}")
    for k, v in useqs.most_common(5): print("   U", v, list(k))
    for k, v in stage_seqs.most_common(2): print("   S", v, list(k))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit11.json"), "w"), ensure_ascii=False, indent=1)
