"""Data for the report's "Upper bound" tab (index.html#ceiling): the stage-level imagine-then-choose search
(runs/20261026_imagine) and the noise probe it started from (runs/20261026_noise_probe).

Writes report/ceiling.js (window.CEILING) and, for one showcase scene per task, report/ceiling/<tag>.mp4: the committed
search path, i.e. the chosen candidate of every expansion on the path, concatenated (CRF 31, index at the front).
Per scene: the plain policy with policy_seed 0 (runs/20261025_seen480) and 1-3 (runs/20261026_imagine), and the search
with every expansion (stage k, start step t, budget L, kept finishers, each candidate's done / steps / progress).
A failed search is put down to the candidate cap, the horizon, or the official success check failing although every
stage check passed ("criterion"). usage: python report/build_ceiling.py; build_pub.py copies the output."""
import glob, json, os, subprocess, tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.dirname(os.path.dirname(HERE))
IMAG = f"{RUNS}/20261026_imagine"
TASKS = [("KettleBoiling", "full", 10), ("SteamInMicrowave", "multi", 12), ("PackIdenticalLunches", "multi", 26)]
seed_of = lambda tag: int(tag.split("_s")[1].split("_")[0])
calls = lambda steps: -(-steps // 16)


def task_block(task, d, show):
    rs = [json.loads(l) for p in glob.glob(f"{IMAG}/{d}/results_w*.jsonl") for l in open(p)]
    rs = [r for r in rs if "error" not in r and r["tag"].startswith(task + "_")]
    base = defaultdict(dict)
    for p in glob.glob(f"{RUNS}/20261025_seen480/run/results_w*.jsonl"):
        for r in map(json.loads, open(p)):
            if r.get("task") == task:
                base[r["tag"]][0] = r
    for r in rs:
        if r["condition"] == "baseline":
            base[r["tag"]][r["policy_seed"]] = r
    search = {r["tag"]: r for r in rs if r["condition"] == "imagine"}
    job = next(j for j in json.load(open(f"{IMAG}/{d}/jobs.json")) if j["task"] == task and j["condition"] == "imagine")
    n_st = len(job["L"])
    scenes, reasons = [], defaultdict(int)
    for tag in sorted(search, key=seed_of):
        s, b = search[tag], base[tag]
        reason = None
        if not s["success"]:
            reason = "criterion" if s["stage"] >= n_st else ("cap" if s["n_candidates"] + job["N"] > job["max_candidates"] else "horizon")
            reasons[reason] += 1
        scenes.append(dict(
            seed=seed_of(tag), plain=[(int(b[p]["success"]) if p in b else None) for p in range(4)],
            ok=s["success"], steps=s["steps"], cands=s["n_candidates"], calls=s["policy_calls"], bt=s["backtracked"],
            stage=s["stage"], reason=reason, path=s["path"],
            exps=[dict(k=e["k"], t=e["t"], L=e["L"], fin=e["finishers"],
                       c=[[int(c["done"]), c["steps"], c["progress"]] for c in e["cands"]]) for e in s["expansions"]]))
    rolls = [r for b in base.values() for r in b.values()]
    n = len(scenes)
    summ = dict(
        n=n, plain=sum(sum(x for x in sc["plain"] if x is not None) / sum(x is not None for x in sc["plain"]) for sc in scenes) / n,
        best4=sum(any(sc["plain"]) for sc in scenes) / n, search=sum(sc["ok"] for sc in scenes) / n,
        greedy=sum(sc["ok"] and not sc["bt"] for sc in scenes) / n,
        calls_plain=sum(calls(r["steps"]) for r in rolls) / len(rolls),
        calls_best4=sum(calls(r["steps"]) for r in rolls) / n, calls_search=sum(sc["calls"] for sc in scenes) / n)
    out = dict(task=task, n_stages=n_st, N=job["N"], keep=job.get("keep", 3), cap=job["max_candidates"], L=job["L"],
               horizon=job["horizon"], summary=summ, reasons=dict(reasons), scenes=scenes)
    # showcase: the committed path of one scene the plain policy never solved
    tag = f"{task}_s{show}_single"
    s = search[tag]
    segs = [dict(k=s["expansions"][ei]["k"], steps=s["expansions"][ei]["cands"][i]["steps"]) for ei, i in s["path"]]
    os.makedirs(f"{HERE}/ceiling", exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for ei, i in s["path"]:
            f.write(f"file '{IMAG}/{d}/videos/{tag}/e{ei}_c{i}.mp4'\n")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", f.name, "-c:v", "libx264", "-crf", "31",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", f"{HERE}/ceiling/{tag}.mp4"], check=True)
    os.unlink(f.name)
    out["showcase"] = dict(seed=show, video=f"ceiling/{tag}.mp4", segs=segs)
    return out


probe = []
for p in json.load(open(f"{RUNS}/20261026_noise_probe/summary.json")):
    sd, t0 = p["point"].split(" t0=")
    probe.append(dict(seed=int(sd[1:]), t0=int(t0), role=p["role"], n=p["n"], success=p["success"], lifted=p["lifted"],
                      spread=p["pos_disp_spread"], grip=p["grip_disagree"]))
C = dict(tasks=[task_block(*t) for t in TASKS], probe=probe)
with open(f"{HERE}/ceiling.js", "w") as f:
    f.write("window.CEILING = ")
    json.dump(C, f, separators=(",", ":"))
    f.write(";\n")
for t in C["tasks"]:
    s = t["summary"]
    print(f"{t['task']}: plain {100 * s['plain']:.1f} best4 {100 * s['best4']:.1f} search {100 * s['search']:.1f} greedy {100 * s['greedy']:.1f} "
          f"calls/plain rollout {s['calls_plain']:.0f} best4/scene {s['calls_best4']:.0f} search/scene {s['calls_search']:.0f} fails {t['reasons']}")
print(f"ceiling.js {os.path.getsize(f'{HERE}/ceiling.js') // 1024} KB;", {p: os.path.getsize(f'{HERE}/ceiling/{p}') // 1024 for p in os.listdir(f'{HERE}/ceiling')})
