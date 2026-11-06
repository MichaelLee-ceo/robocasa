"""When the retrace cannot follow the recorded path (PackIdenticalLunches s4: the arm wedged against the fridge's inner
panel), which fallback gets the arm out so that the retrace can go on? (2026-11-07, the user's idea: the deterministic
retrace by default; when it fails, anything that might work.)

Replay to the failure point; retrace (motion/retrace.py, final: blocked when the arm lags its entry by more than
0.25 rad at 2 chunk ends in a row); then one fallback; then resume the retrace along the recorded path from the last
entry before the blocked one at which the hand was 0.25 m or more from where it is stuck (retrace.resume_entry). The
coordinator does the same with the VLM choosing the sentence (loop.py _unstick, recovery/unstick.py).

fallbacks  none                      nothing (resume at once)
           base_back                 gripper open, the arm's OSC holds the hand in the base frame, the base backs up
                                     for one chunk (command -0.5: about 0.24 m)
           gr00t:<instr>:<seed>      GR00T for 8 chunks with that instruction ("task" = the task sentence)
usage: python -I coordinator/tests/check_retrace_fallback.py <point id> <fallback> [GR00T port]
       one JSON line; frames in runs/20261106_coordinator/retrace_fallback/<id>_<fallback>.mp4"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from coordinator import paths  # noqa: E402
from coordinator.motion import retrace as rt  # noqa: E402
from coordinator.points import load_point  # noqa: E402
from coordinator.tests.check_retrace_state import robot_state, compare  # noqa: E402

paths.use_envkit()
import envkit, policy  # noqa: E402
sys.path.insert(0, paths.RECOVERY)
from trial_video import frame  # noqa: E402

OUT = os.path.join(paths.OUT, "retrace_fallback")


def main(pid, fb, port):
    point, rec = load_point(pid)
    ep = envkit.Episode(point["task"], point["seed"], horizon=10 ** 6)
    d = ep.kenv.sim.data
    trace = rt.RobotTrace(ep)
    frames = []
    starts, reached, S0 = {}, -1, None
    for i, c in enumerate(rec["chunks"][:point["chunk"] + 1]):
        while reached < min(rec["tracks"][i]["stage"], point["stage"]):
            reached += 1
            starts[reached] = len(trace.path) - 1
            if reached == point["stage"]:
                S0 = robot_state(ep, trace)
        ep.step(c["actions"])
    start, n_before = starts[point["stage"]], len(trace.path)
    snap = lambda label: frames.append(frame(ep.obs, f"{pid} {fb}: {label}"))  # noqa: E731
    r1 = rt.retrace(ep, trace, start, 400, on_chunk=lambda e, st: snap(f"retrace k {st['k']}"))
    out = dict(id=pid, fallback=fb, first=r1)
    if not r1["arrived"]:
        p_stall = d.site_xpos[trace.hand].copy()
        if fb == "base_back":
            n = rt.STEPS_PER_CHUNK
            ep.step({"action.end_effector_position": np.zeros((n, 3)), "action.end_effector_rotation": np.zeros((n, 3)),
                     "action.gripper_close": np.zeros((n, 1)), "action.base_motion": np.tile([-0.5, 0, 0, 0], (n, 1)),
                     "action.control_mode": np.ones((n, 1))})
            snap("base back")
        elif fb.startswith("gr00t:"):
            _, instr, seed = fb.split(":")
            client = policy.Client(port)
            for _ in range(8):
                ep.step(policy.act(client, ep.obs, policy.chunk_seed(f"fallback/{pid}/{instr}", int(seed), len(ep.chunks)),
                                   instruction=None if instr == "task" else instr))
                snap(f"GR00T '{instr}'")
        out["hand_moved_cm"] = round(100 * float(np.linalg.norm(d.site_xpos[trace.hand] - p_stall)), 1)
        resume = rt.resume_entry(trace.path[:n_before], start, r1["entry"], p_stall)
        out["resume_entry"], out["blocked_entry"] = int(resume), int(r1["entry"])
        r2 = rt.retrace(ep, trace, start, 400, on_chunk=lambda e, st: snap(f"resume k {st['k']}"), end=resume)
        out["second"] = r2
    out["arrived"] = bool((out.get("second") or r1)["arrived"])
    out["restart"] = {k: v for k, v in compare(S0, robot_state(ep, trace)).items()
                      if k in ("arm_rad", "hand_mm", "base_cm", "fingers_mm", "objects_cm")}
    os.makedirs(OUT, exist_ok=True)
    import imageio.v2 as imageio
    imageio.mimwrite(os.path.join(OUT, f"{pid}_{fb.replace(':', '_').replace(' ', '_')[:60]}.mp4"),
                     [np.asarray(f) for f in frames], fps=4, codec="libx264")
    ep.close()
    return out


if __name__ == "__main__":
    pid, fb = sys.argv[1], sys.argv[2]
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 5581
    try:
        print(json.dumps(main(pid, fb, port), default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), flush=True)
    except Exception as x:
        import traceback
        print(json.dumps(dict(id=pid, fallback=fb, error=f"{type(x).__name__}: {x}", trace=traceback.format_exc()[-1500:])),
              flush=True)
