import json, glob, os
for p in sorted(glob.glob("survey/*.json")):
    if os.path.getsize(p) == 0:
        print(os.path.basename(p), "(running or failed)"); continue
    d = json.loads(open(p).read().strip().splitlines()[-1])
    px = {c: [s["cams"][c]["semi_pixels"] for s in d["scenes"]] for c in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")}
    tops = set()
    for s in d["scenes"]:
        for c in s["cams"].values():
            for name, n in c["top"]:
                tops.add(name.split("|")[1].split("_group_")[-1][:40] + "|" + name.split("|")[2])
    print(f'{d["task"]:22s} semi geoms {[s["n_semi_geoms"] for s in d["scenes"]]} px L {px["robot0_agentview_left"]} R {px["robot0_agentview_right"]} W {px["robot0_eye_in_hand"]} | {sorted(tops)[:6]}')
