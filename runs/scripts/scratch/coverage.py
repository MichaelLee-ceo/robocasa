import sys, os, glob, collections, json
sys.path.insert(0, "/home/michaellee/Desktop/robocasa/runs/scripts")
import pandas as pd
from demo_utils import ds_dir, load_tasks, DS_ROOT
OFF = dict(PickPlaceSinkToCounter=16, TurnOnMicrowave=16, PickPlaceCounterToCabinet=14, PickPlaceCounterToStove=13, CoffeeSetupMug=11,
           TurnOnSinkFaucet=9, PickPlaceCounterToSink=7, PickPlaceCounterToMicrowave=3, CloseMicrowave=0, TurnOnStove=0, StartCoffeeMachine=0)
AS = {"CloseBlenderLid","CloseFridge","CloseToasterOvenDoor","CoffeeSetupMug","NavigateKitchen","OpenCabinet","OpenDrawer","OpenStandMixerHead","PickPlaceCounterToCabinet","PickPlaceCounterToStove","PickPlaceDrawerToCounter","PickPlaceSinkToCounter","PickPlaceToasterToCounter","SlideDishwasherRack","TurnOffStove","TurnOnElectricKettle","TurnOnMicrowave","TurnOnSinkFaucet"}
for t in sorted(os.listdir(DS_ROOT)):
    try: d = ds_dir(t)
    except Exception as e: print(t, "no data"); continue
    tasks = load_tasks(d); files = sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet"))[:60]
    c = collections.Counter()
    for f in files:
        df = pd.read_parquet(f, columns=["annotation.human.subtask_name"])
        names = [tasks[int(x)] for x in df["annotation.human.subtask_name"]]
        m = []
        for n in names:
            if n in ("done", "NavigateKitchen"): continue
            if not m or m[-1] != n: m.append(n)
        c[tuple(m)] += 1
    seq, cnt = c.most_common(1)[0]
    skills = sorted(set(seq))
    print(f"{t:22s} n={len(files)} top({cnt}): {' > '.join(seq)}")
    print("     ", {s: ("AS" if s in AS else "--") + (f" off={OFF[s]}/20" if s in OFF else "") for s in skills})
