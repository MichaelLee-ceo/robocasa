"""Videos of runs, to show them: one frame per chunk (the three camera views GR00T sees) with a label bar.

  plain <point id> <out.mp4>
      the original GR00T run of a phase 0 failure point (its record, replayed exactly), from a few chunks before the
      point to its end; the chunk where the coordinator steps in is marked
  run <point id> <policy seed> <out.mp4> [--vlm SPEC] [--horizon-scale X] [--stuck-windows N]
      a coordinator run (config oracle_retrace) from that point, every chunk labelled: GR00T / rewind / unstick (with
      the sentence GR00T was given), the sub-goal and the attempt, GR00T's time; writes <out>.json with the outcome
A live run is not exactly repeatable (MSAA rendering noise), so its outcome can differ from a batch run of the same
point and seed.
usage: python -I coordinator/scripts/video.py plain|run ..."""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from coordinator import configs, paths  # noqa: E402
from coordinator.points import load_point, run_from_point  # noqa: E402
from coordinator.schema import CONTROL_HZ  # noqa: E402

paths.use_envkit()
import envkit, stages  # noqa: E402

CAMS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")
COLORS = {"GR00T": (70, 130, 220), "rewind": (230, 140, 30), "unstick": (190, 60, 190), "recorded": (110, 110, 110),
          "success": (40, 160, 80)}
BEFORE = 8          # chunks of the record shown before the failure point
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 15)
except OSError:
    FONT = ImageFont.load_default()


def frame(obs, kind, text):
    im = Image.fromarray(np.concatenate([np.asarray(obs[c][0]) for c in CAMS], axis=1).astype(np.uint8))
    out = Image.new("RGB", (im.width, im.height + 30), (20, 20, 24))
    out.paste(im, (0, 30))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, 118, 29], fill=COLORS.get(kind, (90, 90, 90)))
    d.text((8, 6), kind.upper(), fill=(255, 255, 255), font=FONT)
    d.text((128, 6), text, fill=(235, 235, 235), font=FONT)
    return np.asarray(out)


def write(frames, path, fps=8):
    import imageio.v2 as imageio
    imageio.mimwrite(path, frames, fps=fps, codec="libx264")


def plain(pid, out):
    point, rec = load_point(pid)
    ep = envkit.Episode(point["task"], point["seed"])
    names = stages.instructions(ep)
    frames, first = [], max(0, point["chunk"] + 1 - BEFORE)
    for i, c in enumerate(rec["chunks"]):
        ep.step(c["actions"])
        if i + 1 < first:
            continue
        tr = rec["tracks"][i + 1]
        k = min(tr["stage"], len(names) - 1)
        mark = "   << stuck: our loop steps in here" if i == point["chunk"] else ""
        frames.append(frame(ep.obs, "success" if ep.success else "GR00T",
                            f"{ep.t / CONTROL_HZ:4.0f}/{ep.horizon / CONTROL_HZ:.0f} s  step {k + 1}/{len(names)}: "
                            f"{names[k][:44]}{mark}"))
    write(frames, out)
    json.dump(dict(id=pid, success=bool(ep.success), chunks=len(rec["chunks"]), shown_from=first), open(out + ".json", "w"))


def run(pid, seed, out, a):
    point, rec = load_point(pid)
    ep = envkit.Episode(point["task"], point["seed"], horizon=10 ** 6)
    co = configs.build("oracle_retrace", ep, int(os.environ.get("POLICY_PORT", 5581)), f"loop/{pid}/p{seed}",
                       vlm=a.vlm, horizon_scale=a.horizon_scale, stuck_windows=a.stuck_windows)
    frames, n_replay = [], point["chunk"] + 1
    step = ep.step

    def label():
        m = co.memory
        k = min(m.current, len(m.plan) - 1)
        att = len(m.stages[k].attempts)
        if len(ep.chunks) <= n_replay:
            return "recorded", f"GR00T's own run, replayed   step {k + 1}/{len(m.plan)}: {m.plan[k].text[:48]}"
        if co.trace.override is not None:
            return "rewind", f"back to where step {k + 1} began (not timed): {m.plan[k].text[:44]}"
        if m.stream.startswith("u"):
            return "unstick", f"rewind blocked, VLM tells GR00T: \"{(m.gr00t_instruction or 'the task sentence')[:56]}\""
        t = co.ptime() / CONTROL_HZ
        return ("success" if co.success_t is not None else "GR00T"), \
            f"{t:4.0f} s  step {k + 1}/{len(m.plan)} try {att}: {m.plan[k].text[:48]}"

    def wrapped(actions):
        r = step(actions)
        if co.memory is not None and len(ep.chunks) > n_replay - BEFORE:
            kind, text = label()
            frames.append(frame(ep.obs, kind, text))
        return r
    ep.step = wrapped
    res, _ = run_from_point(co, point, rec)
    if res["success"]:
        frames += [frame(ep.obs, "success", "task done (RoboCasa's own success check)")] * 12
    write(frames, out)
    rts = [iv for iv in res["memory"]["interventions"] if iv["kind"] == "retrace"]
    json.dump(dict(id=pid, seed=seed, success=res["success"], fixed=res["fixed"], ended=res.get("ended"),
                   gr00t_time_s=res["gr00t_time"] / CONTROL_HZ, official_horizon_s=res["official_horizon"] / CONTROL_HZ,
                   retraces=[dict(stage=iv["stage"] + 1, arrived=iv["result"]["arrived"],
                                  unstick=[u["instruction"] for u in iv["result"].get("unstick", [])]) for iv in rts],
                   plan=[g.text for g in co.memory.plan], frames=len(frames)), open(out + ".json", "w"), indent=1)
    ep.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plain", "run"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--vlm", default=None)
    ap.add_argument("--horizon-scale", type=float, default=5)
    ap.add_argument("--stuck-windows", type=float, default=4)
    a = ap.parse_args()
    if a.mode == "plain":
        plain(a.args[0], a.args[1])
    else:
        run(a.args[0], int(a.args[1]), a.args[2], a)
