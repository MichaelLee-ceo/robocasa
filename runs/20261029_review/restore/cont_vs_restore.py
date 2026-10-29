"""Tasks 2 + 3: continuous run vs set_state + settle at chunk boundaries, with the same saved actions afterwards, plus the
attribute audit at the same sim state.

Per case (a saved rollout and a list of boundaries b = 16 * chunk):
  cont : the saved rollout replayed exactly from its decision point (checked against the saved states); at each b the
         hook records the true hidden state (hidden_state.capture), every attribute (harness.snap_attrs), the policy
         request and the python fixture state.
  settle (current code, v4) and, optionally, other settle versions: set_state(states[b]) + settle(states[b-1]) + capture,
         then rollout() restores it and runs the saved chunks from chunk b/16 on for `horizon` steps.
  exact: the same, restoring the captured true hidden state (validates the harness: must be 0).
  t=0 cases (b = 0 of a t=0 rollout): 'cont' is the main run's own start, i.e. the hidden state right after load_scene.
Reports max |diff| per state group after 1, 4, 8, 16, 32 steps; held objects' paths; the attribute diff; the request diff.
usage: python -I cont_vs_restore.py cases.json out.jsonl"""
import copy, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
from imagine import py_state, set_py_state  # noqa: E402

AT = (1, 4, 8, 16, 32)


def obj_paths(kenv, states, names):
    """world xyz of the named objects' root bodies along states (set_state each)."""
    out = {n: [] for n in names}
    for s in states:
        H.set_state(kenv, s)
        for n in names:
            out[n].append(np.array(kenv.sim.data.body_xpos[kenv.sim.model.body_name2id(n)]))
    return {n: np.array(v) for n, v in out.items()}


def req_diff(a, b):
    out = {}
    for k in a:
        if k.startswith(("state.", "video.")) and k in b:
            x, y = np.asarray(a[k]).astype(float), np.asarray(b[k]).astype(float)
            if x.shape == y.shape:
                out[k] = round(float(np.abs(x - y).max()), 6)
    return out


def phase(kenv):
    ob = kenv._observables.get("robot0_eye_in_hand_image") or next(iter(kenv._observables.values()))
    return round(float(ob._time_since_last_sample), 6), bool(ob._sampled)


def main():
    cases = json.load(open(sys.argv[1]))
    out = open(sys.argv[2], "a")
    horizon = int(os.environ.get("HORIZON", "32"))
    base, kenv = H.make_env(cases[0]["task"])
    G, OBJ = None, None
    for cs in cases:
        t0 = time.time()
        row = H.find_row(cs["dir"], cs["tag"], cs["t"])
        job = H.find_job(cs["tag"], cs["t"])
        ver = H.VERSION[row["hidden"]]
        saved = np.load(H.roll_path(cs["dir"], cs["tag"], cs["t"], cs["c"], cs["m"]))
        ss = saved["states"]
        P = H.Point(base, kenv, cs["task"], cs["tag"], cs["t"], row["k"], job["scene"], version=ver, row=row)
        G, OBJ = H.groups(kenv)
        bset = sorted(set(cs["boundaries"]))
        rec = {}

        ramp_cont = {}

        def hook(i, req):
            b = 16 * i
            ramp_cont[b] = float(kenv.robots[0].gripper["right"].current_action[1])  # true ramp c at every boundary
            if b in bset:
                rec[b] = dict(H=H.hidden_state.capture(kenv), A=H.snap_attrs(kenv), req=copy.deepcopy(req),
                              ps=py_state(kenv), touch=H.contacts_with_fingers(kenv), phase=phase(kenv))
        n_cont = min(len(ss) - 1, max(bset) + horizon)
        cont, _ = P.run(saved, P.ep[cs["t"]], n_cont, P.H0, hook=hook)
        exact = float(np.abs(cont - ss[:len(cont)]).max())
        res = dict(case=cs, version=ver, cont_vs_saved=exact, S0=P.S0, k_replay=P.k_replay, k=row["k"], boundaries=[])
        # offline ramp (start ramp + 0.2 per step toward the command) vs the env's current_action read at each boundary
        A = saved["action.gripper_close"][:, :, 0].reshape(-1)
        cc = [float(P.H0["robots"][0]["grippers"]["right"]["current_action"][1])]
        for a in np.where(A < 0.5, -1.0, 1.0)[:len(cont)]:
            cc.append(float(np.clip(cc[-1] + 0.2 * a, -1, 1)))
        res["ramp_check"] = dict(n=len(ramp_cont), max_abs_diff=max(abs(cc[b] - v) for b, v in ramp_cont.items()),
                                 n_mid=sum(abs(v) < 0.999 for v in ramp_cont.values()), ramps={b: round(v, 3) for b, v in ramp_cont.items()})
        for b in bset:
            if b == 0 and cs["t"] == 0:  # the main run's own start: true hidden state = right after load_scene
                H.set_state(kenv, ss[0]); truth = dict(H=P.after_load, ps=P.ps)
                ref = None
            elif b not in rec:
                continue
            else:
                truth = rec[b]; ref = cont
            st0, prev = ss[b], (ss[b - 1] if b else None)
            br = dict(b=b, chunk=b // 16, touch=truth.get("touch"), phase_cont=truth.get("phase"), runs={})
            variants = [("exact", None)] + [("settle_" + v, v) for v in cs.get("versions", ["v4"])]
            if b == 0 and cs["t"] == 0:
                variants = [("exact", None), ("settle_" + ver, ver)]
            trajs = {}
            for name, v in variants:
                if v is None:
                    Hh = truth["H"]
                else:
                    set_py_state(kenv, truth["ps"])
                    H.set_state(kenv, st0)
                    H.settle(kenv, prev, v)
                    Hh = H.hidden_state.capture(kenv)
                snapA = {}

                def hk(i, req, snapA=snapA):
                    if i == 0:
                        snapA.update(A=H.snap_attrs(kenv), req=copy.deepcopy(req), phase=phase(kenv))
                P.ps = truth["ps"]
                st, _ = P.run(saved, st0, horizon, Hh, chunk0=b // 16, hook=hk)
                trajs[name] = st
                rr = dict(n=len(st) - 1)
                if b == 0 and cs["t"] == 0:
                    rr["vs_saved"] = {f: H.group_diff(st[f], ss[f], G) for f in AT if f < len(st) and f < len(ss)}
                else:
                    rr["vs_cont"] = {f: H.group_diff(st[f], ss[b + f], G) for f in AT if f < len(st) and b + f < len(ss)}
                if name != "exact" and "A" in truth:
                    rr["attr_diff"] = H.diff_attrs(truth["A"], snapA["A"])
                    rr["req_diff"] = req_diff(truth["req"], snapA["req"])
                    rr["phase_restored"] = snapA["phase"]
                if "H" in Hh and False:
                    pass
                rr["ramp"] = [float(x) for x in Hh["robots"][0]["grippers"]["right"]["current_action"]]
                br["runs"][name] = rr
            # held objects: paths under each variant vs the saved (continuous) one
            held = [o for o in (truth.get("touch") or []) if o.endswith("_main")]
            if b == 0 and cs["t"] == 0:
                held = []
            if held:
                nn = min(len(x) for x in trajs.values())
                nn = min(nn, len(ss) - b)
                P_saved = obj_paths(kenv, ss[b:b + nn], held)
                br["held"] = {}
                for name, st in trajs.items():
                    Pp = obj_paths(kenv, st[:nn], held)
                    br["held"][name] = {o: dict(max_xyz_diff_m=round(float(np.abs(Pp[o] - P_saved[o]).max()), 4),
                                               dz_end_m=round(float(Pp[o][-1, 2] - P_saved[o][-1, 2]), 4),
                                               z_cont=[round(float(x), 3) for x in P_saved[o][::8, 2]],
                                               z_rest=[round(float(x), 3) for x in Pp[o][::8, 2]]) for o in held}
            res["boundaries"].append(br)
            summ = {k: {f: {g: round(v, 5) for g, v in d.items() if v > 1e-9} for f, d in
                        (r.get("vs_cont") or r.get("vs_saved") or {}).items() if f in (1, 16)} for k, r in br["runs"].items()}
            print(json.dumps(dict(case=f"{cs['tag']} t{cs['t']} c{cs['c']} b{b}", label=cs.get("label"), summ=summ)), flush=True)
        res["wall_s"] = round(time.time() - t0, 1)
        out.write(json.dumps(res, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n"); out.flush()


if __name__ == "__main__":
    main()
