"""Fill only the DATA object of the published page (layout/text untouched) from the frozen-scene runs."""
import json, glob, os, re, shutil, sys, datetime
OUT = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/runs/20261013_scenes_v2")
SRC = sys.argv[1]  # live index.html read from the artifact
DST = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/runs/report_v2")
status = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
os.makedirs(f"{DST}/videos", exist_ok=True)

R = {}
for p in glob.glob(f"{OUT}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r:
            R[r["tag"]] = r

html = open(SRC).read()
m = re.search(r"const DATA = (\{.*?\n\});\n", html, re.S)
D = json.loads(m.group(1))


def frac(rows, f):
    return [sum(bool(f(r)) for r in rows), len(rows)] if rows else None


def poster(src, dst):
    try:
        import av, imageio.v3 as iio
        c = av.open(src); fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]; c.close()
        iio.imwrite(dst, fr[len(fr) // 2], extension=".jpg", quality=70); return True
    except Exception:
        return False


tot = {}
files = []
for T in D["tasks"]:
    t, n = T["id"], len(T["steps"])
    run = {"single": [R.get(f"{t}_s{s}_single") for s in range(20)]}
    for ap in ("aseen", "pre"):
        run[ap] = [R.get(f"{t}_s{s}_{ap}_chain") for s in range(20)]
    T["videos"] = []
    for ap, rows in run.items():
        rows = [r for r in rows if r]
        res = T["result"][ap]
        res["task"] = frac(rows, lambda r: r["success"])
        res["chain"] = [frac(rows, lambda r, k=k: r["reached"] > k or r["success"]) for k in range(n)]
        if res["task"]:
            tot.setdefault(ap, [0, 0]); tot[ap][0] += res["task"][0]; tot[ap][1] += res["task"][1]
        if ap != "single":
            res["solo"] = [frac([R[x] for x in R if x.startswith(f"{t}_s") and x.endswith(f"_{ap}_solo_k{k}")], lambda r: r["step_success"])
                           for k in range(n)]
            res["solo_demo"] = [frac([R[x] for x in R if x.startswith(f"{t}_d") and x.endswith(f"_{ap}_solodemo_k{k}")], lambda r: r["step_success"])
                                for k in range(n)]
        if ap == "single":  # one success and one failure, lowest scene id
            picks = [sorted([r for r in rows if bool(r["success"]) == w], key=lambda r: r["seed"])[:1] for w in (True, False)]
        else:  # same scene as the single-horizon success video, plus this approach's first success if that one failed
            ref = sorted([r["seed"] for r in run["single"] if r and r["success"]])[:1]
            same = [r for r in rows if r["seed"] in ref]
            picks = [same] + ([sorted([r for r in rows if r["success"]], key=lambda r: r["seed"])[:1]]
                              if not any(r["success"] for r in same) else [])
        for c in picks:
            if not c:
                continue
            r = c[0]; src = f"{OUT}/{t}/{r['tag']}.mp4"
            if not os.path.exists(src):
                continue
            shutil.copy(src, f"{DST}/videos/{r['tag']}.mp4"); files.append(f"videos/{r['tag']}.mp4")
            v = dict(approach=ap, scene=r["seed"], success=bool(r["success"]), stuck=None if r["success"] else r["reached"] + 1,
                     file=f"videos/{r['tag']}.mp4")
            if poster(src, f"{DST}/videos/{r['tag']}.jpg"):
                v["poster"] = f"videos/{r['tag']}.jpg"; files.append(v["poster"])
            T["videos"].append(v)
for ap in ("single", "aseen", "pre"):
    D["totals"][ap] = tot.get(ap)
full = [R[x] for x in R if x.endswith("_aseen_full")]
D["control"]["aseen"] = frac(full, lambda r: r["success"])
if status:
    D["status"] = status
D["updated"] = datetime.date.today().isoformat()
new = html[:m.start(1)] + json.dumps(D, ensure_ascii=False, indent=1) + html[m.end(1):]
open(f"{DST}/index.html", "w").write(new)
json.dump(sorted(set(files)), open(f"{DST}/files.json", "w"))
print(json.dumps(dict(totals=D["totals"], control=D["control"], status=D["status"],
                      tasks={T["id"]: T["result"] for T in D["tasks"]}), ensure_ascii=False))
