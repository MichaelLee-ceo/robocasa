"""Build the HTML report (runs/report/index.html + videos/) from the experiment summaries."""
import json, os, sys, shutil, glob, html, collections
import numpy as np

RUNS = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/runs")
A = f"{RUNS}/20261012_atomic_decomp_seen"
RS = f"{RUNS}/20261012_groot_repro_seen_h1x"
RU = f"{RUNS}/20261012_groot_repro_h1x"
OUT = f"{RUNS}/report"
os.makedirs(f"{OUT}/videos", exist_ok=True)

S = json.load(open(f"{A}/summary.json"))


def repro_table(d, paper_file):
    P = json.load(open(paper_file))
    rows = collections.defaultdict(list); by_tag = {}
    for p in glob.glob(f"{d}/results_w*.jsonl") + glob.glob(f"{d}/retry/results_w*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            if "error" not in r:
                by_tag[r["tag"]] = r  # one record per episode (a retried tag can appear twice)
    for r in by_tag.values():
        rows[r["task"]].append(r["success"])
    out = []
    for t, v in rows.items():
        out.append(dict(task=t, k=sum(v), n=len(v), paper=P.get(t)))
    out.sort(key=lambda x: -(x["paper"] or 0))
    K = sum(o["k"] for o in out); N = sum(o["n"] for o in out)
    return dict(rows=out, k=K, n=N, paper_avg=P.get("average"))


seen = repro_table(RS, f"{RS}/paper_table11_composite_seen.json")
unseen = repro_table(RU, f"{RU}/paper_table11_composite_unseen.json")


def poster(src, dst):
    try:
        import av
        c = av.open(src); fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        if not fr:
            return False
        import imageio.v3 as iio
        iio.imwrite(dst, fr[len(fr) // 2], extension=".jpg", quality=70)
        return True
    except Exception:
        return False


def pick_videos(task, T):
    """a few representative episodes per task: baseline success+fail, chain, and one reset per step"""
    eps = T["episodes"]; chosen = []

    def first(f):
        for e in eps:
            if f(e) and e not in chosen:
                return e

    for f, label in [(lambda e: e["condition"] == "baseline" and e["success"], "完整指令・成功"),
                     (lambda e: e["condition"] == "baseline" and not e["success"], "完整指令・失敗"),
                     (lambda e: e["condition"] == "chain_atomic" and e["success"], "atomic 串接・成功"),
                     (lambda e: e["condition"] == "chain_atomic" and not e["success"], "atomic 串接・失敗")]:
        e = first(f)
        if e:
            chosen.append(dict(e, label=label))
    for k in range(len(T["steps"])):
        e = first(lambda e: e["condition"] == "reset_atomic" and e["step"] == k and not e["step_success"]) or \
            first(lambda e: e["condition"] == "reset_atomic" and e["step"] == k)
        if e:
            chosen.append(dict(e, label=f"第 {k+1} 步單獨・{'成功' if e['step_success'] else '失敗'}"))
    vids = []
    for e in chosen:
        src = f"{A}/{e['video']}"
        if not os.path.exists(src):
            continue
        name = os.path.basename(src)
        shutil.copy(src, f"{OUT}/videos/{name}")
        ok = poster(src, f"{OUT}/videos/{name[:-4]}.jpg")
        vids.append(dict(file=f"videos/{name}", poster=f"videos/{name[:-4]}.jpg" if ok else None, label=e["label"],
                         demo=e["demo"], steps=e["steps"], budget=e["budget"], success=e["success"] if e["condition"] in ("baseline", "chain_atomic") else e["step_success"],
                         instr=e["instr_log"]))
    return vids


data = dict(tasks={}, seen=seen, unseen=unseen, n_rollouts=S["n_rollouts"])
for t, T in S["tasks"].items():
    T = dict(T); T["videos"] = pick_videos(t, T); T.pop("episodes")
    data["tasks"][t] = T

# seeded (fresh scenes) comparison + verification
SD = f"{RUNS}/20261012_atomic_seeded"
SS = json.load(open(f"{SD}/summary.json"))
seeded_rows = {}
for p in glob.glob(f"{SD}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r:
            seeded_rows[r["tag"]] = r
for t, T in SS["tasks"].items():
    vids = []
    for cond, succ, label in (("baseline_seed", True, "新場景・整句指令・成功"), ("baseline_seed", False, "新場景・整句指令・失敗"),
                              ("chain_seed", True, "新場景・atomic 串接・成功"), ("chain_seed", False, "新場景・atomic 串接・失敗")):
        c = sorted([r for r in seeded_rows.values() if r["task"] == t and r["condition"] == cond and r["success"] == succ], key=lambda r: r["seed"])
        if not c:
            continue
        r = c[0]; src = f"{SD}/{t}/{r['tag']}.mp4"
        if not os.path.exists(src):
            continue
        name = "seed_" + os.path.basename(src); shutil.copy(src, f"{OUT}/videos/{name}")
        ok = poster(src, f"{OUT}/videos/{name[:-4]}.jpg")
        vids.append(dict(file=f"videos/{name}", poster=f"videos/{name[:-4]}.jpg" if ok else None, label=label, seed=r["seed"],
                         steps=r["steps"], budget=r["horizon"], success=r["success"], instr=r["instr_log"],
                         reached=r["reached"], n_steps=r["n_steps"]))
    T["videos"] = vids
data["seeded"] = SS
# experiments A / B (StackBowlsCabinet, SteamInMicrowave)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomic_steps import TASKS as _TASKS
ABf = f"{SD}/summary_AB.json"
if os.path.exists(ABf):
    AB = json.load(open(ABf))
    for t, T in AB.items():
        T["aug_instr"] = [st["instr_aug"](None, None) if "instr_aug" in st else None for st in _TASKS[t]]
        for cond, succ, label in (("chain_then_task_seed", False, "新場景・A（第 1 步 atomic → 整句）・失敗"),
                                  ("chain_aug_seed", True, "新場景・B（atomic 補資訊）・成功"),
                                  ("chain_aug_seed", False, "新場景・B（atomic 補資訊）・失敗")):
            c = sorted([r for r in seeded_rows.values() if r["task"] == t and r["condition"] == cond and r["success"] == succ], key=lambda r: r["seed"])
            if not c:
                continue
            r = c[0]; src = f"{SD}/{t}/{r['tag']}.mp4"
            if not os.path.exists(src):
                continue
            name = "seed_" + os.path.basename(src); shutil.copy(src, f"{OUT}/videos/{name}")
            ok = poster(src, f"{OUT}/videos/{name[:-4]}.jpg")
            SS["tasks"][t]["videos"].append(dict(file=f"videos/{name}", poster=f"videos/{name[:-4]}.jpg" if ok else None, label=label,
                                                 seed=r["seed"], steps=r["steps"], budget=r["horizon"], success=r["success"],
                                                 instr=r["instr_log"], reached=r["reached"], n_steps=r["n_steps"]))
    data["AB"] = AB
_grp = {"baseline_seed": "整句", "baseline_seed_save": "整句", "chain_seed": "串接", "chain_then_task_seed": "A",
        "chain_aug_seed": "B", "reset_seed_atomic": "D", "reset_seed_task": "D"}
_by = collections.Counter(_grp.get(r["condition"], r["condition"]) for r in seeded_rows.values())
data["seeded_count"] = dict(total=sum(_by.values()), by={k: _by[k] for k in ("整句", "串接", "A", "B", "D") if _by[k]})
RT = f"{RUNS}/20261012_atomic_repeat_test"
rep = collections.defaultdict(lambda: [0, 0])
for p in glob.glob(f"{RT}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        ok = (r["reached"] >= 1) if r["condition"] == "chain_atomic" else r["step_success"]
        rep[r["condition"]][0] += int(bool(ok)); rep[r["condition"]][1] += 1
data["repeat_test"] = dict(rep)

json.dump(data, open(f"{OUT}/data.json", "w"), ensure_ascii=False, indent=1)
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html")).read()
F = json.load(open(f"{A}/findings.json")) if os.path.exists(f"{A}/findings.json") else None
open(f"{OUT}/index.html", "w").write(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)).replace("/*__FINDINGS__*/null", json.dumps(F, ensure_ascii=False)))
print("videos:", len(os.listdir(f"{OUT}/videos")), "size MB:", round(sum(os.path.getsize(f"{OUT}/videos/{f}") for f in os.listdir(f"{OUT}/videos")) / 1e6, 1))
