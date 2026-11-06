"""Check retrace.py at phase 0 failure points: replay to the failure point with a Recorder attached, retrace to the
failed stage's start, and report how close the robot gets to its stage-start pose, how many chunks it takes, where
every task object ends (vs. the stage start and the failure point), and the stages before and after.
Saves loop/retrace_check/<id>.pkl.gz (for trial_video.py, from the failure point on).
Frames of the retrace itself (it cannot be replayed from the actions: the arm runs on joint torques) are written live:
loop/video/retrace/<id>_live.mp4 and _sheet.jpg (every 2nd chunk), with the gripper command and the entry reached.
usage: python -I check_retrace.py <point id> ..."""
import gzip, json, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"), HERE]
import numpy as np  # noqa: E402
import envkit, stages  # noqa: E402
import retrace  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image  # noqa: E402
from trial_video import frame  # noqa: E402

for pid in sys.argv[1:]:
    point = next(p for p in json.load(open(os.path.join(HERE, "phase0", "points.json"))) if p["id"] == pid)
    rec = pickle.load(gzip.open(os.path.join(HERE, "phase0", "records", f"{point['task']}_s{point['seed']}.pkl.gz")))
    ep = envkit.Episode(point["task"], point["seed"], horizon=10 ** 6)  # the retrace's time is not counted
    e = ep.kenv
    R = retrace.Recorder(ep)
    objs = lambda: {o: e.sim.data.body_xpos[e.obj_body_id[o]].copy() for o in e.objects}  # noqa: E731
    at_start = start_idx = None
    for i, c in enumerate(rec["chunks"][:point["chunk"] + 1]):
        if i == point["stage_start"]:
            at_start, start_idx = objs(), len(R.path) - 1
        ep.step(c["actions"])
    if start_idx is None:
        at_start, start_idx = objs(), len(R.path) - 1
    at_fail, st_fail = objs(), stages.track(ep)["stage"]
    n0 = len(ep.chunks)
    frames = [frame(ep.obs, f"{pid} failure point")]
    out = retrace.retrace(ep, R, start_idx, on_chunk=lambda ep, st: frames.append(
        frame(ep.obs, f"{pid} retrace chunk {len(frames)} entry {st['k']} grip {st['grip']:.0f}")))
    vd = os.path.join(HERE, "loop", "video", "retrace")
    os.makedirs(vd, exist_ok=True)
    imageio.mimwrite(os.path.join(vd, f"{pid}_live.mp4"), [np.asarray(f) for f in frames], fps=4, codec="libx264")
    sel = frames[::2] + ([frames[-1]] if len(frames) % 2 == 0 else [])
    sheet = Image.new("RGB", (sel[0].size[0], sel[0].size[1] * len(sel)))
    for j, f in enumerate(sel):
        sheet.paste(f, (0, sel[0].size[1] * j))
    sheet.save(os.path.join(vd, f"{pid}_sheet.jpg"), quality=85)
    after = objs()
    moved = {o: dict(from_start_cm=round(100 * float(np.linalg.norm(after[o] - at_start[o])), 1),
                     failure_from_start_cm=round(100 * float(np.linalg.norm(at_fail[o] - at_start[o])), 1))
             for o in e.objects if np.linalg.norm(at_fail[o] - at_start[o]) > 0.01 or np.linalg.norm(after[o] - at_fail[o]) > 0.01}
    print(json.dumps(dict(id=pid, stage=point["stage"], stage_before=st_fail, stage_after=stages.track(ep)["stage"],
                          **out, objects_moved=moved), default=str), flush=True)
    d = os.path.join(HERE, "loop", "retrace_check")
    os.makedirs(d, exist_ok=True)
    with gzip.open(os.path.join(d, f"{pid}.pkl.gz"), "wb") as f:
        pickle.dump(dict(out=dict(task=point["task"], seed=point["seed"]), start_chunk=n0, chunks=ep.chunks), f)
    ep.close()
