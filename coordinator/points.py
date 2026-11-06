"""Runs from the phase 0 failure points (runs/20261105_recovery/phase0/points.json), exactly as loop.py ran them:
replay the recorded chunks up to the point (each stage's entry recorded as loop.py's remember() did, up to the failed
stage), the failed stage stuck at once (detection assumed solved), and loop.py's rule for "fixed": the first time the
run moves on from the failed stage, else, at the end, the failed stage done by the stage check or the task succeeded
(fixed_after = GR00T chunks since the failure point, the end clipped to the horizon)."""
import gzip, json, os, pickle

from .paths import POINTS, RECORDS, use_envkit
from .schema import STEPS_PER_CHUNK, Trigger

use_envkit()
import stages  # noqa: E402


def load_point(pid):
    point = next(p for p in json.load(open(POINTS)) if p["id"] == pid)
    rec = pickle.load(gzip.open(os.path.join(RECORDS, f"{point['task']}_s{point['seed']}.pkl.gz")))
    return point, rec


def run_from_point(co, point, rec):
    """replay to the point, run the coordinator from it; returns (result dict, GR00T time at the failure point)"""
    co.start()
    co.replay(rec["chunks"][:point["chunk"] + 1], stage_after=[t["stage"] for t in rec["tracks"]], cap=point["stage"])
    t_fail = co.ptime()
    out = co.run(Trigger("failure_point", point["stage"], len(co.ep.chunks), dict(kind=point["kind"])))
    out.update(fixed_rule(co, point["stage"], t_fail, out["success"]))
    out.update(id=point["id"], task=point["task"], seed=point["seed"], stage=point["stage"], kind=point["kind"],
               start_t=t_fail, start_chunk=point["chunk"] + 1, chunks_left=point["chunks_left"])
    return out, t_fail


def fixed_rule(co, stage, t_fail, success):
    """loop.py: fixed_at = round((ptime - t_fail) / 16) when the run moves on from the failed stage; at the end, if
    not yet fixed and the failed stage is done (stage check) or the task succeeded, round((min(ptime, H) - t_fail) / 16)"""
    ev = next((e for e in co.memory.moved_on if e["frm"] == stage and e["t"] >= t_fail), None)
    if ev is not None:
        at = round((ev["t"] - t_fail) / STEPS_PER_CHUNK)
    elif stages.track(co.ep)["done"][stage] or success:
        at = round((min(co.ptime(), co.H) - t_fail) / STEPS_PER_CHUNK)
    else:
        at = None
    return dict(fixed=at is not None, fixed_after=at)
