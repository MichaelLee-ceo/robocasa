"""Build the browsable diagnosis report (report/index.html reads report/data.js).

Collects, per task and episode: the rollout's online stage log (the sequential stage tracker, logged at chunk
boundaries), the replay packet (stage conditions, scene), the video agents' diagnosis (diagnosis/<Task>/<tag>.json) and
the failure-mode mapping (diagnosis/_aggregate.json, from diagnosis/aggregate.py). Each stage gets a clip = the video
between the previous stage's logged completion and its own (the first unfinished stage runs to the end of the episode),
and a poster frame (the completion moment, or the start of the root-cause window for the failed stage).
Units: 1 step = 0.05 s; video frame i = state after step 4*(i+1), played at 10 fps.
usage (groot env, for PyAV and markdown): python report/build.py"""
import glob, json, os, re
from multiprocessing import Pool

import av
import markdown
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # runs/20261025_seen480
MODE_LABEL = {
    "fixture_actuation": "Fixture actuation", "grasp_acquisition": "Grasp acquisition", "placement": "Placement",
    "grasp_retention": "Grasp retention", "task_level": "Task level", "stage_not_terminated": "Stage not ended",
    "process_skill": "Process skill", "disturbance": "Disturbance", "criterion_or_sim": "Criterion / sim",
}


def seed_of(tag):
    return int(re.search(r"_s(\d+)_", tag).group(1))


def episode(task, tag, agg):
    p = json.load(open(f"{ROOT}/packets/{task}/{tag}.json"))
    d = json.load(open(f"{ROOT}/diagnosis/{task}/{tag}.json"))
    T = p["steps"]
    logged = {k: t for t, k in p["online_stage_log"]}
    stages, prev, failed_seen = [], 0, False
    for s in p["stages"]:
        k = s["stage"]
        t = logged.get(k)
        last_undo = s["undone_events"][-1]["reason"] if s["undone_events"] else ""
        st = dict(k=k, skill=s["skill"], logged=t, replay_first=s["first_complete_step"],
                  complete_at_end=s["complete_at_end"],
                  # undone for real = the condition is off at the end and not only because the gripper came back near
                  undone_real=not s["complete_at_end"] and not last_undo.startswith("objects still at target"))
        if t is not None:
            st.update(status="done", clip=[prev, t], poster=t); prev = max(prev, t)
        elif not failed_seen:
            failed_seen = True
            rc = d.get("root_cause_steps") or []
            pstep = rc[0] if rc and prev <= rc[0] <= T else T
            st.update(status="failed" if not p["success"] else "done", clip=[prev, T], poster=pstep)
        else:
            st.update(status="not_reached")
        stages.append(st)
    a = agg.get(tag, {})
    ev = (d.get("evidence") or {}).get("sheets") or []
    return dict(
        tag=tag, seed=seed_of(tag), success=bool(p["success"]), steps=T, horizon=p["horizon"],
        success_step=p["success_step"], layout=p["scene"].get("layout"), style=p["scene"].get("style"),
        objects={k: v.split("/")[-2] if "/" in v else v for k, v in (p["scene"].get("objects") or {}).items()
                 if not k.startswith("distr")},
        stages=stages, replay_matches_online=p["replay_matches_online_log"],
        diag=dict(failed_stage=d.get("failed_stage"), root_cause_stage=d.get("root_cause_stage"),
                  root_cause_steps=d.get("root_cause_steps"), category_raw=d.get("category"),
                  category=a.get("category"), mode=a.get("mode"), secondary=d.get("secondary") or [],
                  what_happened=d.get("what_happened") or "", causal_chain=d.get("causal_chain") or "",
                  self_recovery=d.get("self_recovery") or "", criterion=d.get("official_criterion_notes") or "",
                  confidence=d.get("confidence"), uncertain=d.get("uncertain_about") or "",
                  sheets=[f"../diagnosis/{task}/{s}" for s in ev if os.path.exists(f"{ROOT}/diagnosis/{task}/{s}")]),
        video=f"../run/{task}/{tag}.mp4")


def posters(job):
    """job = (mp4, [(step, out_jpg), ...]); one decode per video."""
    mp4, want = job
    want = [(s, o) for s, o in want if not os.path.exists(o)]
    if not want:
        return 0
    frames = [f.to_ndarray(format="rgb24") for f in av.open(mp4).decode(video=0)]
    for step, out in want:
        i = min(max(step // 4 - 1, 0), len(frames) - 1)
        Image.fromarray(frames[i]).save(out, quality=72, optimize=True)
    return len(want)


def main():
    agg = {r["tag"]: r for r in json.load(open(f"{ROOT}/diagnosis/_aggregate.json"))}
    tasks, jobs = [], []
    for tj in sorted(glob.glob(f"{ROOT}/packets/*/_task.json")):
        t = json.load(open(tj))
        task = t["task"]
        tags = sorted((os.path.basename(p)[:-5] for p in glob.glob(f"{ROOT}/packets/{task}/{task}_s*_single.json")),
                      key=seed_of)
        eps = [episode(task, tag, agg) for tag in tags]
        os.makedirs(f"{HERE}/posters/{task}", exist_ok=True)
        for e in eps:
            want = []
            for s in e["stages"]:
                if "poster" in s:
                    s["poster_img"] = f"posters/{task}/{e['tag']}_{s['k']}.jpg"
                    want.append((s["poster"], f"{HERE}/{s['poster_img']}"))
            jobs.append((f"{ROOT}/run/{task}/{e['tag']}.mp4", want))
        summary = open(f"{ROOT}/diagnosis/{task}/_summary.md").read()
        tasks.append(dict(
            task=task, instruction=t["instruction"], horizon=t["horizon"], successes=t["successes"],
            stages=[re.sub(r"^\d+:\s*", "", s) for s in t["stages"]],
            stage_definitions=t["stage_definitions"], typical=t["typical_stage_steps_in_successes"],
            summary_html=markdown.markdown(summary, extensions=["tables", "fenced_code", "sane_lists"]),
            episodes=eps))
    with Pool(32) as pool:
        made = sum(pool.map(posters, jobs, chunksize=4))
    data = dict(tasks=tasks, mode_label=MODE_LABEL, generated_from="runs/20261025_seen480")
    with open(f"{HERE}/data.js", "w") as f:
        f.write("window.REPORT = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    n = sum(len(t["episodes"]) for t in tasks)
    print(f"{len(tasks)} tasks, {n} episodes, {made} posters written, data.js {os.path.getsize(f'{HERE}/data.js') // 1024} KB")


if __name__ == "__main__":
    main()
