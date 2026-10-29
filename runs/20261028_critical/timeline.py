"""Episode timelines: stages, physical events (events.py) and the outcome at each collected decision point.
One PNG per task, timeline_<task>.png, a row per seed-0 main-run episode.
Decision point classes, from its 8 candidates x 2 continuations:
  all done / all failed   every one of the 16 rollouts finished the stage / none did
  mixed                   some did, but no candidate went 2/2 while another went 0/2
  split                   at least one candidate finished in both continuations and another in neither: the chunk
                          evidently changes the outcome there
Points: the coarse rows in use (data.py). Writes the PNGs to $CRIT_OUT. usage: python -I timeline.py"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import HERE, OUT, TASKS  # noqa: E402
from effect_by_event import points  # noqa: E402

SURFACE, INK, INK2, ACCENT = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
BAND = ["#dcdbd7", "#c8c7c2"]
EVENTS = {"hold": ("^", "grasp starts"), "release": ("v", "release"), "fx_move": ("D", "fixture starts moving")}


def cls(y):
    per = y.sum(1)
    if y.sum() == y.size:
        return "all done"
    if y.sum() == 0:
        return "all failed"
    return "split" if (per == 2).any() and (per == 0).any() else "mixed"


STYLE = {"all done": dict(facecolor="#b9b8b3", edgecolor="#b9b8b3"), "all failed": dict(facecolor=INK2, edgecolor=INK2),
         "mixed": dict(facecolor=SURFACE, edgecolor=ACCENT), "split": dict(facecolor=ACCENT, edgecolor=ACCENT)}


def plot(task, eps, pts):
    T = max(e["T"] for e in eps)
    fig, ax = plt.subplots(figsize=(14, 0.3 * len(eps) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    counts = {c: 0 for c in STYLE}
    for i, e in enumerate(eps):
        y = len(eps) - 1 - i
        ends = [t for t, _ in e["history"]]
        a = 0
        for k, b in enumerate(ends):
            ax.add_patch(Rectangle((a, y - 0.16), b - a, 0.32, color=BAND[k % 2], lw=0))
            if b - a > 60:
                ax.text((a + b) / 2, y, f"S{k}", ha="center", va="center", fontsize=6.5, color=INK2)
            a = b
        if a < e["T"] - 1:
            ax.add_patch(Rectangle((a, y - 0.16), e["T"] - 1 - a, 0.32, facecolor=SURFACE, edgecolor=BAND[1], hatch="////", lw=0.6))
            ax.text(a + 8, y, f"S{len(ends)} not done", ha="left", va="center", fontsize=6.5, color=INK2)
        for ev in e["events"]:
            if ev["type"] in EVENTS:
                ax.plot(ev["t"], y + 0.3, EVENTS[ev["type"]][0], ms=3.2, color=INK2, mew=0)
        for p in pts.get(e["tag"], []):
            c = cls(p["y"]); counts[c] += 1
            ax.scatter(p["t"], y, s=26, zorder=3, linewidths=1.3, **STYLE[c])
        ax.text(-12, y, f"s{e['tag'].split('_s')[1].split('_')[0]} {'✓' if e['success'] else '✗'}", ha="right", va="center",
                fontsize=7.5, color=INK if e["success"] else INK2)
    ax.set_xlim(-70, T + 10); ax.set_ylim(-0.8, len(eps) - 0.2)
    ax.set_yticks([]); ax.set_xlabel("step (20 per second; a chunk is 16 steps)", color=INK2, fontsize=8.5)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#a9a8a3")
    n = sum(counts.values())
    ax.set_title(f"{task}: the candidate chunk visibly changes the stage outcome at {counts['split']} of {n} decision points "
                 f"(filled blue)", loc="left", fontsize=11, color=INK, pad=24)
    handles = [plt.scatter([], [], s=26, linewidths=1.3, label=f"{c} ({counts[c]})", **STYLE[c]) for c in ["split", "mixed", "all done", "all failed"]]
    handles += [plt.Line2D([], [], marker=m, ls="", ms=4, color=INK2, label=lab) for m, lab in EVENTS.values()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=7, frameon=False, fontsize=7.5,
              labelcolor=INK2, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    fig.tight_layout()
    out = os.path.join(OUT, f"timeline_{task}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE); plt.close(fig)
    return out, counts


def main():
    ev = {r["tag"]: r for r in map(json.loads, open(os.path.join(HERE, "events.jsonl")))}
    by_tag = {}
    for p in points():
        by_tag.setdefault(p["tag"], []).append(p)
    for task in TASKS:
        eps = sorted([e for e in ev.values() if e["task"] == task], key=lambda e: int(e["tag"].split("_s")[1].split("_")[0]))
        print(*plot(task, eps, by_tag))


if __name__ == "__main__":
    main()
