"""Quality check of collected settled-v5 rows (run by watchdog.sh after every poll; incremental). For every new row in
data/<task>/<dataset>/ that has the per-boundary logs (settled-v5), without simulation:
  restore  the restored ramp is on the 0.2 grid; at t = 0 it is exactly 0 (no settle).
  fingers  the main run's finger gap vs the 8 candidates' (m0) at steps 1, 2, 4 and 8. > 3 mm outside their range within 4
           steps is flagged and classed:
           - policy: the ramp the main run used in its first step is within one command step of the restored one;
           - contact: the fingers touch something at t, where the ramp cannot be inferred from the gap;
           - restore?: the restore itself may be off.
  bounds   per rollout: one boundary log per chunk run; done by the stage check <=> stage k's bit at the last boundary;
           a success has success steps logged.
Writes logs/qc.jsonl (one record per row), logs/flags.jsonl (one line per flag), logs/qc_summary.txt. Investigated flags
are explained in logs/flags_reviewed.md. usage: python -I qc.py"""
import glob, json, os
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
Q = f"{D}/logs"
MAIN = f"{D}/../20261025_seen480/run"
F1, F2 = 12, 13  # finger joints in the flattened state (1 + qpos index; the robot comes first in every task)


def rows(d):
    out = []
    for p in sorted(glob.glob(f"{d}/results*.jsonl")):
        for l in open(p):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def find(d, tag, t):
    hit = None
    for r in rows(d):
        if r.get("tag") == tag and r.get("t") == t and "error" not in r:
            hit = r
    return hit


def labels(r):
    return np.array([[ro["done"] for ro in c["rollouts"]] for c in r["cands"]], float)


def check(d, r):
    task, tag, t, k = r["task"], r["tag"], r["t"], r["k"]
    rec = dict(dir=os.path.relpath(d, f"{D}/data"), tag=tag, t=t, k=k, flags=[])
    ri = r.get("restore", {})
    c0 = ri.get("ramp")
    rec["ramp"] = c0
    if c0 is None or abs(round(c0 / 0.2) * 0.2 - c0) > 1e-6 or abs(c0) > 1:
        rec["flags"].append(f"restore: ramp {c0} off the 0.2 grid")
    if t == 0 and c0 != 0:
        rec["flags"].append(f"restore: t=0 ramp {c0}, must be 0")
    ep = np.load(f"{MAIN}/{task}/{tag}_steps.npz")["states"]
    gm = np.abs(ep[t:t + 9, F1] - ep[t:t + 9, F2])
    G = []
    for i in range(r["N"]):
        s = np.load(f"{d}/roll/{tag}_t{t}_c{i}_m0.npz")["states"][:9]
        G.append(np.abs(s[:, F1] - s[:, F2]))
    n = min(min(len(g) for g in G), len(gm))
    G = np.array([g[:n] for g in G])
    out = {s: round(1000 * float(max(gm[s] - G[:, s].max(), G[:, s].min() - gm[s], 0)), 2) for s in (1, 2, 4, 8) if s < n}
    rec["main_outside_mm"] = out
    if max([v for s, v in out.items() if s <= 4] + [0]) > 3:
        # The ramp the main run used during its first step, from its own motion (the servo relation of
        # hidden_state.gripper_ramp). Within one command step (+-0.2) of the restored ramp, the restore agrees with the
        # main run, and the gap comes from its commands differing from all 8 candidates' (checked at
        # PackIdenticalLunches s0 t352: one close command, then open). Otherwise the restore itself may be off.
        c_main = 1 - (gm[1] + 2 * (gm[1] - gm[0])) / 0.04
        rec["c_main_step0"] = round(float(c_main), 2)
        # With the fingers touching something at t the servo relation does not hold (contact sets the gap), so the
        # inferred ramp says nothing there: "contact" (SteamInMicrowave s21 t928,
        # fingers against the microwave door: inferred -0.41 vs restored -1, V 1.0 both ways).
        if ri.get("fixture") or ri.get("obj") or ri.get("held"):
            kind = "contact"
        else:
            kind = "policy" if c0 is not None and abs(c_main - c0) <= 0.3 else "restore?"
        rec["flags"].append(f"fingers({kind}): main run {out} mm outside the candidates' range; restored ramp {c0}, "
                            f"main's first step implies {c_main:+.2f}")
    y = labels(r)
    rec["V"] = round(float(y.mean()), 3)
    bad = []
    for c in r["cands"]:
        for ro in c["rollouts"]:
            b = ro.get("bounds")
            if b is None:
                bad.append("no bounds"); continue
            if len(b) != -(-ro["steps"] // 16):
                bad.append(f"c{c['cand']}m{ro['m']}: {len(b)} bounds for {ro['steps']} steps")
            stage_done = ro["reached"] > k
            last_bit = bool(b and b[-1][1] >> k & 1)
            if stage_done and not last_bit:
                bad.append(f"c{c['cand']}m{ro['m']}: done by stage check but bit {k} not set at the last boundary")
            if not stage_done and any(x[1] >> k & 1 for x in b):
                bad.append(f"c{c['cand']}m{ro['m']}: not done but bit {k} set at a boundary")
            if ro["success"] and not any(x[2] > 0 for x in b):
                bad.append(f"c{c['cand']}m{ro['m']}: success without success steps logged")
    if bad:
        rec["flags"].append("bounds: " + "; ".join(bad[:4]) + (f" (+{len(bad) - 4})" if len(bad) > 4 else ""))
    return rec


def main():
    done_path = f"{Q}/qc.jsonl"
    seen = set()
    if os.path.exists(done_path):
        for l in open(done_path):
            x = json.loads(l)
            seen.add((x["dir"], x["tag"], x["t"]))
    with open(done_path, "a") as fo, open(f"{Q}/flags.jsonl", "a") as ff:
        for d in sorted(glob.glob(f"{D}/data/*/*")):
            for r in rows(d):
                if "error" not in r and r.get("hidden") != "settled-v5":
                    continue  # rows from before settled-v5 have no per-boundary logs to check
                key = (os.path.relpath(d, f"{D}/data"), r.get("tag"), r.get("t"))
                if key in seen:
                    continue
                seen.add(key)
                if "error" in r:
                    rec = dict(dir=key[0], tag=key[1], t=key[2], error=r["error"][:400],
                               flags=[f"error row: {r['error'][:200]}"])
                else:
                    try:
                        rec = check(d, r)
                    except Exception as e:  # a check that crashes is itself worth a look
                        rec = dict(dir=key[0], tag=key[1], t=key[2], flags=[f"qc crashed: {e!r}"[:300]])
                fo.write(json.dumps(rec) + "\n")
                for f in rec["flags"]:
                    ff.write(json.dumps(dict(dir=rec["dir"], tag=rec["tag"], t=rec["t"], flag=f)) + "\n")
    recs = [json.loads(l) for l in open(done_path)]
    ok = [x for x in recs if "error" not in x]
    kinds = {}
    for x in recs:
        for f in x["flags"]:
            kinds[f.split(":")[0]] = kinds.get(f.split(":")[0], 0) + 1
    with open(f"{Q}/qc_summary.txt", "w") as fs:
        fs.write(f"rows checked {len(recs)} (ok {len(ok)}, error {len(recs) - len(ok)}); flags by kind {kinds}\n")


if __name__ == "__main__":
    main()
