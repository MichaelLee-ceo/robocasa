"""Build the composite-seen stage page: one stage line per task (its atomic subtasks in order), one row per episode
showing at which step each subtask was completed under the composite instruction, where it got stuck, and its video.
usage: python build_stage_page.py <out_dir> <run_dir> [<run_dir> ...]
Writes <out_dir>/index.html and copies the videos to <out_dir>/videos/<Task>/<tag>.mp4."""
import glob, json, os, shutil, sys

STAGES = {  # short names of the tracker steps in runs/scripts/atomic_steps.py, in order
    "StackBowlsCabinet": ["第 1 個碗放進櫃子", "第 2 個碗疊上去"],
    "SteamInMicrowave": ["蔬菜放進碗", "碗放進微波爐", "關微波爐門", "按開始"],
    "KettleBoiling": ["水壺放上爐口", "開火"],
    "PreSoakPan": ["鍋子放進水槽", "海綿放進水槽", "開水龍頭"],
    "PrepareCoffee": ["杯子放到咖啡機下", "按咖啡機按鈕"],
}
NAMES = {"StackBowlsCabinet": "StackBowlsCabinet", "SteamInMicrowave": "SteamInMicrowave", "KettleBoiling": "KettleBoiling",
         "PreSoakPan": "PreSoakPan", "PrepareCoffee": "PrepareCoffee"}
OLD = {  # same checkpoint + scenes on hscc-5090-1 (runs/20261013_accept, runs/20261013_scenes_v2)
    "StackBowlsCabinet": "17/20, 15/20", "SteamInMicrowave": "7/20, 10/20", "KettleBoiling": "11/20",
    "PreSoakPan": "15/20", "PrepareCoffee": "4/20",
}

out, runs = sys.argv[1], sys.argv[2:]
os.makedirs(f"{out}/videos", exist_ok=True)
tasks = {}
for rd in runs:
    note = open(f"{rd}/NOTE.txt").read() if os.path.exists(f"{rd}/NOTE.txt") else ""
    for p in sorted(glob.glob(f"{rd}/results_w*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            if "error" in r or r["task"] not in STAGES:
                continue
            src = f"{rd}/{r['task']}/{r['tag']}.mp4"
            vid = None
            if os.path.exists(src):
                os.makedirs(f"{out}/videos/{r['task']}", exist_ok=True)
                shutil.copy2(src, f"{out}/videos/{r['task']}/{r['tag']}.mp4")
                vid = f"videos/{r['task']}/{r['tag']}.mp4"
            done = {k: t for t, k in r["history"]}  # step index k finished at env step t
            tasks.setdefault(r["task"], dict(task=r["task"], stages=STAGES[r["task"]], instr=r["task_instr"],
                                              horizon=r["horizon"], old=OLD.get(r["task"], ""), run=os.path.basename(rd.rstrip("/")),
                                              eps=[]))
            tasks[r["task"]]["eps"].append(dict(seed=r["seed"], success=r["success"], steps=r["steps"], reached=r["reached"],
                                                done=[done.get(k) for k in range(len(STAGES[r["task"]]))], video=vid))
order = [t for t in STAGES if t in tasks]
for t in order:
    tasks[t]["eps"].sort(key=lambda e: e["seed"])
data = json.dumps([tasks[t] for t in order], ensure_ascii=False)
html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage_page_template.html")).read()
open(f"{out}/index.html", "w").write(html.replace("/*__DATA__*/[]", data))
print(json.dumps({t: f"{sum(e['success'] for e in tasks[t]['eps'])}/{len(tasks[t]['eps'])}" for t in order}))
