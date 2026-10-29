"""State error caused by each gripper-ramp rule (coordinator request 2026-10-29).
Per case (a saved rollout and boundaries b): replay it exactly from its decision point (hook at every chunk boundary:
the env's true ramp c = env.robots[0].gripper["right"].current_action[1], compared with the offline formula); at each
selected b restore the state with set_state + settle (current code, v4), then overwrite only the gripper ramp with the
value of each rule (and with the true c, which isolates the rest of settle), run the same saved actions for 16 steps
and compare with the continuous run. 'exact' restores the captured true hidden state (must give 0).
Rules: see grip_eval.py (a_v3, b_v4, c_v5, d_val, f_hyb); contacts for d_val / f_hyb are read in the env at b.
usage: python -I ramp_restore.py cases.json out.jsonl"""
import copy, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
from imagine import py_state, set_py_state  # noqa: E402
from grip_rules3 import FIXTURE_WORDS  # noqa: E402

OPEN_V4, OPEN_V3 = 0.0785, abs(0.020833 * 2) * 1.9 - 0.004
AT = (1, 2, 4, 8, 16)


def finger_contacts(kenv):
    """root bodies touched by the left / right finger bodies."""
    m, d = kenv.sim.model, kenv.sim.data
    L, Rr = set(), set()
    for i in range(d.ncon):
        c = d.contact[i]
        for ga, gb in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            ba, bb = m.geom_bodyid[ga], m.geom_bodyid[gb]
            na = m.body_id2name(ba) or ""
            if not na.startswith("gripper0_"):
                continue
            other = m.body_id2name(m.body_rootid[bb]) or ""
            if other.startswith(("robot0_", "gripper0_", "mobilebase0_")):
                continue
            if "leftfinger" in na or "joint1_tip" in na:
                L.add(other)
            elif "rightfinger" in na or "joint2_tip" in na:
                Rr.add(other)
    return L, Rr


def rule_values(g, b, held, obj):
    g0, g1, g4, g8 = g[b], g[b - 1], g[max(b - 4, 0)], g[max(b - 8, 0)]
    v1, d4, d8 = g0 - g1, g0 - g4, g0 - g8
    out = {}
    out["a_v3"] = -1.0 if (g0 >= OPEN_V3 or v1 > 0.0005) else 1.0
    out["b_v4"] = -1.0 if (g0 >= OPEN_V4 or v1 > 0.0005) else 1.0
    out["c_v5"] = 1.0 if d8 < -0.0003 else (-1.0 if d8 > 0.0003 else (1.0 if g0 < OPEN_V4 else -1.0))
    val = float(np.clip(np.round((1 - g0 / 0.04) / 0.2) * 0.2, -1, 1))
    out["d_val"] = -1.0 if d4 > 0.0002 else (1.0 if d4 < -0.0002 else (1.0 if obj else val))
    out["f_hyb"] = out["b_v4"] if held else out["a_v3"]
    return out


def main():
    cases = json.load(open(sys.argv[1]))
    out = open(sys.argv[2], "a")
    base, kenv = H.make_env(cases[0]["task"])
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
        ramp_env, rec = {}, {}

        def hook(i, req):
            b = 16 * i
            ramp_env[b] = float(kenv.robots[0].gripper["right"].current_action[1])
            if b in bset:
                L, Rr = finger_contacts(kenv)
                rec[b] = dict(H=H.hidden_state.capture(kenv), ps=py_state(kenv), L=sorted(L), R=sorted(Rr))
        n_cont = min(len(ss) - 1, max(bset) + 16)
        cont, _ = P.run(saved, P.ep[cs["t"]], n_cont, P.H0, hook=hook)
        A = saved["action.gripper_close"][:, :, 0].reshape(-1)
        cc = [float(P.H0["robots"][0]["grippers"]["right"]["current_action"][1])]
        for a in np.where(A < 0.5, -1.0, 1.0)[:len(cont)]:
            cc.append(float(np.clip(cc[-1] + 0.2 * a, -1, 1)))
        res = dict(case=cs, version=ver, cont_vs_saved=float(np.abs(cont - ss[:len(cont)]).max()),
                   ramp_check=dict(n=len(ramp_env), max_abs_diff=max(abs(cc[b] - v) for b, v in ramp_env.items()),
                                   n_mid=sum(abs(v) < 0.999 for v in ramp_env.values())), boundaries=[])
        g = np.abs(ss[:, 12] - ss[:, 13])
        for b in bset:
            if b not in rec:
                continue
            r = rec[b]
            free_objs = lambda s: {o for o in s if o.endswith("_main") and not any(w in o for w in FIXTURE_WORDS)}
            held = bool(free_objs(r["L"]) & free_objs(r["R"]))
            obj = bool(free_objs(r["L"]) | free_objs(r["R"]))
            c_true = ramp_env[b]
            rv = rule_values(g, b, held, obj)
            values = sorted({round(c_true, 3)} | {round(v, 3) for v in rv.values()})
            br = dict(b=b, gap_mm=round(float(g[b]) * 1000, 2), c_true=round(c_true, 3), held=held, obj=obj,
                      contacts=dict(L=r["L"], R=r["R"]), rules=rv, runs={})
            objs = sorted({o for o in r["L"] + r["R"] if o.endswith("_main")})
            for name, cval in [("exact", None)] + [(f"ramp={v:+.1f}", v) for v in values]:
                if cval is None:
                    Hh = r["H"]
                else:
                    set_py_state(kenv, r["ps"]); H.set_state(kenv, ss[b]); H.settle(kenv, ss[b - 1], "v4")
                    kenv.robots[0].gripper["right"].current_action = np.array([-cval, cval])
                    Hh = H.hidden_state.capture(kenv)
                P.ps = r["ps"]
                st, _ = P.run(saved, ss[b], 16, Hh, chunk0=b // 16)
                n = min(len(st), len(ss) - b)
                d = {f: H.group_diff(st[f], ss[b + f], G) for f in AT if f < n}
                gr = np.abs(st[:n, 12] - st[:n, 13]); gc = g[b:b + n]
                rr = dict(diff=d, finger_gap_dev_mm=round(float(np.abs(gr - gc).max()) * 1000, 2),
                          gap_rest_mm=[round(x * 1000, 1) for x in gr[::2]], gap_cont_mm=[round(x * 1000, 1) for x in gc[::2]])
                if objs:
                    dev = {}
                    for o in objs:
                        if o in kenv.sim.model.body_names:
                            bid = kenv.sim.model.body_name2id(o)
                            pr, pc = [], []
                            for f in range(n):
                                H.set_state(kenv, st[f]); pr.append(np.array(kenv.sim.data.body_xpos[bid]))
                                H.set_state(kenv, ss[b + f]); pc.append(np.array(kenv.sim.data.body_xpos[bid]))
                            dev[o] = round(float(np.abs(np.array(pr) - np.array(pc)).max()), 4)
                    rr["obj_dev_m"] = dev
                br["runs"][name] = rr
            res["boundaries"].append(br)
            print(json.dumps(dict(case=f"{cs['tag']} t{cs['t']} c{cs['c']} b{b}", c_true=br["c_true"], gap=br["gap_mm"], held=held,
                                  obj=obj, rules=rv, gapdev={k: v["finger_gap_dev_mm"] for k, v in br["runs"].items()},
                                  objdev={k: v.get("obj_dev_m") for k, v in br["runs"].items()})), flush=True)
        res["wall_s"] = round(time.time() - t0, 1)
        out.write(json.dumps(res, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n"); out.flush()


if __name__ == "__main__":
    main()
