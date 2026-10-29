"""The observables' sampling clock at every chunk boundary of R rollouts run in sequence from one restored decision point
(as collect.py runs its N*M rollouts), with the current hidden_state (capture/restore include the clock). Before the fix
rollout r was r substeps behind rollout 0 (restore review F4); now every rollout should read the same phase, equal to
the main run's (the clock right after load_scene + one forced update, constant over steps).
usage: python -I obs_clock_check.py <dir> <task> <tag> <t> <c> <R> out.json"""
import json, os, sys
import numpy as np

RV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "restore")
sys.path.insert(0, RV)
import harness as H  # noqa: E402


def phase(kenv):
    o = kenv._observables
    return {n: (round(float(o[n]._time_since_last_sample), 6), bool(o[n]._sampled)) for n in ("robot0_eye_in_hand_image", "robot0_base_to_eef_pos")}


def main():
    d, task, tag, t, c, R, outp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
    row = H.find_row(d, tag, t); job = H.find_job(tag, t)
    base, kenv = H.make_env(task)
    P = H.Point(base, kenv, task, tag, t, row["k"], job["scene"], version="v5", row=row)
    saved = np.load(H.roll_path(d, tag, t, c, 0))
    runs = []
    for r in range(R):
        ph = {}
        st, cl = P.run(saved, P.ep[t], 64, P.H0, hook=lambda i, req, ph=ph: ph.__setitem__(16 * i, phase(kenv)))
        runs.append(dict(r=r, phase=ph, state_vs_r0=None if not runs else float(np.abs(st - runs[0]["st"]).max()), st=st,
                         req=[{k: np.asarray(v) for k, v in q.items()} for q in cl.reqs]))
        print(r, ph, flush=True)
    out = dict(case=dict(dir=d, task=task, tag=tag, t=t, c=c), after_point=phase(kenv), runs=[])
    for ru in runs:
        reqd = [max(float(np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)).max()) for k in a) for a, b in zip(ru["req"], runs[0]["req"])]
        out["runs"].append(dict(r=ru["r"], phase=ru["phase"], state_vs_r0=ru["state_vs_r0"], proprio_req_vs_r0=reqd))
    same = all(ru["phase"] == runs[0]["phase"] for ru in runs)
    out["all_phases_equal"] = same
    print("all phases equal:", same, "| proprio request diff vs r0 per chunk:", [o["proprio_req_vs_r0"] for o in out["runs"]])
    json.dump(out, open(outp, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
