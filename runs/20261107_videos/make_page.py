"""Writes index.html: GR00T's original runs next to the coordinator's rewind-and-retry runs (videos from
coordinator/scripts/video.py, outcomes from their .json files). usage: python3 make_page.py"""
import html, json, os

HERE = os.path.dirname(os.path.abspath(__file__))

# (point id, policy seed): the section follows the live run (a VLM unstick -> "unstick", else success -> "fixed")
EXAMPLES = [("KettleBoiling_s17", 2), ("KettleBoiling_s9", 0), ("SteamInMicrowave_s1", 2), ("SteamInMicrowave_s12", 1),
            ("PackIdenticalLunches_s19", 0), ("PackIdenticalLunches_s19", 2),
            ("SteamInMicrowave_s28", 1), ("KettleBoiling_s26", 1), ("PackIdenticalLunches_s20", 2),
            ("PackIdenticalLunches_s20", 0), ("PackIdenticalLunches_s20", 3), ("PackIdenticalLunches_s21", 1),
            ("PackIdenticalLunches_s13", 3), ("PackIdenticalLunches_s25", 0)]
TASKS = {"KettleBoiling": "Kettle Boiling", "SteamInMicrowave": "Steam In Microwave",
         "PackIdenticalLunches": "Pack Identical Lunches"}


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def card(pid, seed):
    task, scene = pid.rsplit("_s", 1)
    plain, ours = load(f"{pid}_groot.mp4.json"), load(f"{pid}_p{seed}_ours.mp4.json")
    if plain is None or ours is None:
        return None, None, None
    plan, rts = ours["plan"], ours["retraces"]
    unstick = any(r["unstick"] for r in rts)
    section = "unstick" if unstick else ("fixed" if ours["success"] else None)
    k = rts[0]["stage"] if rts else 1
    t = ours["gr00t_time_s"] / ours["official_horizon_s"]
    if ours["success"]:
        verdict, pill, cls = f"task done, GR00T time {t:.2f}x the official limit", "success", "ok"
    elif ours["fixed"]:
        verdict, pill, cls = "the stuck step got done, the whole task did not", "step fixed", "mid"
    else:
        verdict, pill, cls = "not recovered", "failed", "bad"
    plan_li = "".join(f"<li>{html.escape(x)}</li>" for x in plan)
    lines = []
    for r in rts:
        say = " &rarr; ".join(f"VLM: &ldquo;{html.escape(x)}&rdquo;" for x in r["unstick"])
        lines.append(f"<li>step {r['stage']}: " + (f"rewind blocked &rarr; {say} &rarr; " if say else "")
                     + ("back at the step's start, retry" if r["arrived"] else "not back at the start, retry from there")
                     + "</li>")
    body = f"""
<article class="card">
  <header>
    <h3>{TASKS[task]} <span class="muted">scene {scene} · policy seed {seed}</span></h3>
    <span class="pill {cls}">{pill}</span>
  </header>
  <p class="why">GR00T alone got stuck at step {k} of {len(plan)}: <i>{html.escape(plan[k - 1])}</i></p>
  <details><summary>All steps (split from the task sentence)</summary><ol>{plan_li}</ol></details>
  <div class="pair">
    <figure>
      <figcaption><b>GR00T alone</b>: its original run, from 8 chunks before it got stuck to the time limit ({'success' if plain['success'] else 'failed'})</figcaption>
      <video controls muted playsinline preload="metadata" src="videos/{pid}_groot.mp4"></video>
    </figure>
    <figure>
      <figcaption><b>Ours</b>: the same run up to the stuck point, then rewind and retry ({verdict})</figcaption>
      <video controls muted playsinline preload="metadata" src="videos/{pid}_p{seed}_ours.mp4"></video>
    </figure>
  </div>
  <p class="facts">Rewinds ({len(rts)}):</p>
  <ul class="says">{"".join(lines)}</ul>
</article>"""
    return section, body, [f"{pid}_groot.mp4", f"{pid}_p{seed}_ours.mp4"]


def main():
    cards = {"fixed": [], "unstick": []}
    used = []
    scenes = set()
    for pid, seed in EXAMPLES:
        sec, c, files = card(pid, seed)
        if sec is None or (sec == "unstick" and pid in scenes) or len(cards.get(sec, [])) >= 6:
            continue
        scenes.add(pid)
        cards[sec].append(c)
        used += [f for f in files if f not in used]
    page = open(os.path.join(HERE, "page_template.html")).read()
    page = page.replace("<!--FIXED-->", "\n".join(cards["fixed"])).replace("<!--UNSTICK-->", "\n".join(cards["unstick"]) or
        '<p class="note"><i>These videos are still being recorded; this section will be filled in when they are done.</i></p>')
    open(os.path.join(HERE, "index.html"), "w").write(page)
    json.dump(used, open(os.path.join(HERE, "page_files.json"), "w"), indent=1)
    print(len(cards["fixed"]), "fixed cards,", len(cards["unstick"]), "unstick cards,", len(used), "videos")


if __name__ == "__main__":
    main()
