"""Observation fidelity of restored rollouts.
A. Rendering determinism: the forced observation at one fixed state, rendered repeatedly (with collect.snapshot()'s
   480x480 renders in between, as collect.decision_point does).
B. Observation-sampling phase: robosuite observables sample when their own clock (_time_since_last_sample, never reset by
   rollout()) wraps; every rollout() start adds one forced update (+1 physics substep, 2 ms). So rollout r of a decision
   point samples its policy observations 1 substep earlier than rollout r-1. Here: the same saved rollout (same actions,
   same restored start) is run R times in sequence, as collect.py runs its N*M rollouts; the policy request at each
   later chunk boundary is compared with rollout 0's and with the true observation of the state at that boundary.
usage: python -I obs_phase.py <dir> <task> <tag> <t> <c> <R> out.json"""
import copy, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
from collect import snapshot  # noqa: E402


def ob_phase(kenv):
    o = kenv._observables
    return {n: (round(float(o[n]._time_since_last_sample), 6), bool(o[n]._sampled))
            for n in ("robot0_eye_in_hand_image", "robot0_base_to_eef_pos") if n in o}


def main():
    d, task, tag, t, c, R, outp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
    row = H.find_row(d, tag, t); job = H.find_job(tag, t)
    base, kenv = H.make_env(task)
    P = H.Point(base, kenv, task, tag, t, row["k"], job["scene"], version=H.VERSION[row["hidden"]], row=row)
    res = dict(case=dict(dir=d, task=task, tag=tag, t=t, c=c), after_point=ob_phase(kenv))
    # A. rendering determinism at the decision point's state
    H.set_state(kenv, P.ep[t]); H.hidden_state.restore(kenv, P.H0)
    imgs = []
    for i in range(6):
        if i == 3:
            snapshot(kenv)  # as collect.decision_point does once, before the rollouts
        o = base.get_observation(kenv._get_observations(force_update=True))
        imgs.append({k: np.array(v) for k, v in o.items() if k.startswith(("video.", "state."))})
    res["render_repeat_maxdiff"] = [{k: float(np.abs(imgs[i][k].astype(float) - imgs[0][k].astype(float)).max()) for k in imgs[0]}
                                    for i in range(1, len(imgs))]
    res["after_renders"] = ob_phase(kenv)
    # B. R rollouts in sequence from the same restored start, as collect.py does
    saved = np.load(H.roll_path(d, tag, t, c, 0))
    ss = saved["states"]
    n = min(len(ss) - 1, 64)
    runs = []
    for r in range(R):
        ph0 = ob_phase(kenv)
        st, cl = P.run(saved, P.ep[t], n, P.H0)
        runs.append(dict(phase_start=ph0, reqs=cl.reqs, states_match=float(np.abs(st - ss[:len(st)]).max())))
    # true observation at each boundary: forced update right after set_state of the saved boundary state
    truth = {}
    for b in range(16, n + 1, 16):
        H.set_state(kenv, ss[b])
        o = base.get_observation(kenv._get_observations(force_update=True))
        truth[b] = {k: np.array(v) for k, v in o.items() if k.startswith("state.")}
    res["runs"] = []
    for r, ru in enumerate(runs):
        rr = dict(r=r, phase_start=ru["phase_start"], states_vs_saved=ru["states_match"], boundary={})
        for ci, rq in enumerate(ru["reqs"]):
            b = 16 * ci
            if b == 0 or b not in truth:
                continue
            rr["boundary"][b] = {k: dict(vs_r0=float(np.abs(np.asarray(rq[k]) - np.asarray(runs[0]["reqs"][ci][k])).max()),
                                         vs_true=float(np.abs(np.asarray(rq[k])[0, 0] - truth[b][k][0]).max()))
                                 for k in rq if k.startswith("state.")}
        res["runs"].append(rr)
        print(r, ru["phase_start"], {b: {k.split('.')[-1]: round(v["vs_true"] * 1000, 2) for k, v in x.items()} for b, x in rr["boundary"].items()}, flush=True)
    json.dump(res, open(outp, "w"), default=str, indent=1)


if __name__ == "__main__":
    main()
