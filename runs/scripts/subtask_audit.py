"""Read-only audit of the target datasets' subtask annotations for the 5 tasks."""
import sys, os, glob, json, collections
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_utils import ds_dir, load_tasks
OURS = dict(KettleBoiling=["PickPlaceCounterToStove", "TurnOnStove"],
            PreSoakPan=["PickPlaceCounterToSink", "PickPlaceCounterToSink", "TurnOnSinkFaucet"],
            StackBowlsCabinet=["PickPlaceCounterToCabinet", "PickPlaceCounterToCabinet"],
            SteamInMicrowave=["PickPlaceSinkToCounter", "PickPlaceCounterToMicrowave", "CloseMicrowave", "TurnOnMicrowave"],
            PrepareCoffee=["CoffeeSetupMug", "StartCoffeeMachine"])
out = {}
for t, ours in OURS.items():
    d = ds_dir(t); tasks = load_tasks(d)
    files = sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet"))
    seqs = collections.Counter(); sent = collections.defaultdict(collections.Counter); stage_seqs = collections.Counter()
    for f in files:
        df = pd.read_parquet(f, columns=["annotation.human.subtask_name", "annotation.human.subtask", "annotation.human.subtask_stage"])
        names = [tasks[int(x)] for x in df["annotation.human.subtask_name"]]
        subs = [tasks[int(x)] for x in df["annotation.human.subtask"]]
        stg = [tasks[int(x)] for x in df["annotation.human.subtask_stage"]]
        merged = []; segs = []
        for i, (n, s, g) in enumerate(zip(names, subs, stg)):
            if not segs or (n, s, g) != segs[-1]:
                segs.append((n, s, g))
        for n, s, g in segs:
            sent[(n, g)][s] += 1
        m = []
        for n, _, _ in segs:
            if not m or m[-1] != n:
                m.append(n)
        seqs[tuple(x for x in m if x != "NavigateKitchen")] += 1
        stage_seqs[tuple(f"{n}:{g}" for n, _, g in segs)] += 1
    match = seqs.get(tuple(ours), 0)
    out[t] = dict(n=len(files), ours=ours, match=match, sequences=[[list(k), v] for k, v in seqs.most_common()],
                  stage_sequences=[[list(k), v] for k, v in stage_seqs.most_common(4)],
                  sentences={f"{n} | {g}": c.most_common(3) for (n, g), c in sent.items()})
    print(f"== {t}: {len(files)} demos, matching our order (NavigateKitchen dropped): {match}")
    for k, v in seqs.most_common(6):
        print("   ", v, list(k))
json.dump(out, open(os.path.expanduser("~/Desktop/robocasa/runs/subtask_audit.json"), "w"), ensure_ascii=False, indent=1)
