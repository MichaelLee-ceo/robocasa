"""Aggregates scan.jsonl (+ coverage.json) into the integrity tables and the anomaly lists of report.md.
Writes analyze.json and prints the tables. usage: python -I analyze.py"""
import collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, COARSE, OTHER, N_STAGES, FIX_JOBS, DATA  # noqa: E402

SLACK = 5.0  # s, file mtime tolerance


def load():
    recs = [json.loads(l) for l in open(os.path.join(HERE, "scan.jsonl"))]
    crash = [r for r in recs if "crash" in r]
    recs = [r for r in recs if "crash" not in r]
    # current coarse rows: highest override level holding (tag, t)
    lvl = {d: (task, i) for task, lv in COARSE.items() for i, ds in enumerate(lv) for d in ds}
    best = {}
    for r in recs:
        if r["dir"] in lvl:
            key = (r["task"], r["tag"], r["t"])
            if key not in best or lvl[r["dir"]][1] > lvl[best[key]["dir"]][1]:
                best[key] = r
    for r in recs:
        if r["dir"] in lvl:
            r["status"] = "current" if best[(r["task"], r["tag"], r["t"])] is r else "superseded"
        else:
            r["status"] = OTHER[r["dir"]][0]
    return recs, crash


def group_of(r):
    return f"{r['status']}:{r['dir']}"


def main():
    recs, crash = load()
    out = dict(crash=crash)
    by = collections.defaultdict(list)
    for r in recs:
        by[group_of(r)].append(r)
    jobs_fix = {}
    for jf, fd in FIX_JOBS.items():
        for j in json.load(open(f"{DATA}/{jf}")):
            jobs_fix.setdefault((j["tag"], j["t"]), []).append(jf)

    # ---------------- per-group tables
    T = {}
    anomalies = collections.defaultdict(list)
    for g, rs in sorted(by.items()):
        c = collections.Counter()
        fd = collections.Counter()
        for r in rs:
            c["points"] += 1
            nst = N_STAGES[r["task"]]
            c["rollouts"] += len(r["rolls"])
            # structure
            if r["n_cands"] != r["N"]:
                c["n_cands_ne_N"] += 1; anomalies["n_cands_ne_N"].append((g, r["tag"], r["t"]))
            per_c = collections.Counter(o["i"] for o in r["rolls"])
            if any(v != r["M"] for v in per_c.values()):
                c["rollouts_ne_M"] += 1; anomalies["rollouts_ne_M"].append((g, r["tag"], r["t"]))
            if r["k_replay"] != r["k"]:
                c["k_replay_ne_k"] += 1; anomalies["k_replay_ne_k"].append((g, r["tag"], r["t"], r["k"], r["k_replay"]))
            if not r["has_py_state"]:
                c["py_state_missing"] += 1; anomalies["py_state_missing"].append((g, r["tag"], r["t"]))
            elif not r["py_state_len"]:
                c["py_state_empty"] += 1
            if r["state_len"] != r["main_len"]:
                c["layout_len_ne_main"] += 1
            # snap
            s = r["snap"]
            if not s["exists"]:
                c["snap_missing"] += 1; anomalies["snap_missing"].append((g, r["tag"], r["t"]))
            elif "err" in s or not s.get("crc_ok"):
                c["snap_bad"] += 1; anomalies["snap_bad"].append((g, r["tag"], r["t"], s.get("err")))
            elif not s.get("state_eq_main"):
                c["snap_state_ne_main"] += 1; anomalies["snap_state_ne_main"].append((g, r["tag"], r["t"]))
            # obs_fp
            fps = {o["obs_fp"] for o in r["rolls"]}
            if len(fps) != 1:
                c["obs_fp_not_unique"] += 1; anomalies["obs_fp_not_unique"].append((g, r["tag"], r["t"], len(fps)))
            # mtimes
            mt = [o["mtime"] for o in r["rolls"] if "mtime" in o]
            if s.get("exists") and mt:
                after = max(mt) - s["mtime"]
                before = (s["mtime"] - r["wall_s"]) - min(mt)
                if after > SLACK:
                    c["roll_newer_than_snap"] += 1; anomalies["roll_newer_than_snap"].append((g, r["tag"], r["t"], round(after, 1)))
                if before > SLACK:
                    c["roll_older_than_point"] += 1; anomalies["roll_older_than_point"].append((g, r["tag"], r["t"], round(before, 1)))
            for o in r["rolls"]:
                key = (g, r["tag"], r["t"], o["i"], o["m"])
                if o.get("npz") != "ok":
                    c["npz_" + o.get("npz", "?")[:12]] += 1; anomalies["npz_bad"].append(key + (o.get("npz"),)); continue
                if o["mp4"] != "ok":
                    c["mp4_" + o["mp4"]] += 1; anomalies["mp4_bad"].append(key + (o["mp4"],))
                if o["keys_missing"] or o["keys_extra"]:
                    c["keys_bad"] += 1; anomalies["keys_bad"].append(key)
                if o["shape_bad"]:
                    c["shape_bad"] += 1; anomalies["shape_bad"].append(key + (o["shape_bad"],))
                if o["D"] != r["main_len"]:
                    c["state_dim_ne_main"] += 1; anomalies["state_dim_ne_main"].append(key + (o["D"], r["main_len"]))
                if not o["finite"] or not o["act_finite"]:
                    c["nonfinite"] += 1; anomalies["nonfinite"].append(key)
                if o["st_dtype"] != "float64" or o["act_dtype"] != ["float32"]:
                    c["dtype"] += 1
                if o["time_dev"] is None or o["time_dev"] > 1e-9:
                    c["time_step_bad"] += 1; anomalies["time_step_bad"].append(key + (o["time_dev"],))
                if not o["s0_eq_main"]:
                    c["s0_ne_main"] += 1; anomalies["s0_ne_main"].append(key + (o.get("s0_maxdiff"), o.get("s0_argmax")))
                # row vs npz
                if o["S"] != o["steps"] + 1:
                    c["S_ne_steps+1"] += 1; anomalies["S_ne_steps+1"].append(key + (o["S"], o["steps"]))
                if o["C"] * 16 != o["steps"]:
                    c["C*16_ne_steps"] += 1; anomalies["C*16_ne_steps"].append(key + (o["C"], o["steps"]))
                if o["steps"] > r["L"]:
                    c["steps_gt_L"] += 1; anomalies["steps_gt_L"].append(key + (o["steps"], r["L"]))
                if not o["done"] and o["steps"] != r["L"]:
                    c["notdone_steps_ne_L"] += 1; anomalies["notdone_steps_ne_L"].append(key + (o["steps"], r["L"]))
                if o["done"] != (o["reached"] > r["k"] or o["success"]):
                    c["done_ne_reached_or_success"] += 1; anomalies["done_ne_reached"].append(key + (o["done"], o["reached"], o["success"]))
                if o["reached"] < r["k"] or o["reached"] > nst:
                    c["reached_out_of_range"] += 1
                if not o["done"] and o["reached"] != r["k"]:
                    c["notdone_reached_ne_k"] += 1
                if o["success"] and not o["done"]:
                    c["success_not_done"] += 1
                if o["success"] and r["k"] < nst - 1 and o["reached"] < nst:
                    c["success_before_last_stage"] += 1; anomalies["success_before_last_stage"].append(key + (o["reached"],))
                if not o["prev_ok"] and (not o["done"] or r["k"] == 0 or r["k"] >= nst - 1):
                    c["prev_ok_impossible"] += 1
                if not o["prev_ok"]:
                    c["prev_ok_false"] += 1
                if not (0 <= o["progress"] <= 1):
                    c["progress_out_of_range"] += 1
                if o["done"] and o["steps"] == 16:
                    c["done_in_first_chunk"] += 1
                fd[o["steps"] % 16] += 1
            # within candidate
            for p in r["pairs"]:
                c["pairs"] += 1
                if not p["c0_eq"]:
                    c["pair_c0_ne"] += 1; anomalies["pair_c0_ne"].append((g, r["tag"], r["t"], p["i"], p["ma"], p["mb"], p["c0_neq_keys"]))
                if not p["s16_eq"]:
                    c["pair_s16_ne"] += 1; anomalies["pair_s16_ne"].append((g, r["tag"], r["t"], p["i"], p["ma"], p["mb"], p.get("first_div"), p.get("div_max")))
                if p["both_long"]:
                    c["pairs_both_long"] += 1
                    if p["c1_eq"]:
                        c["pair_c1_eq"] += 1; anomalies["pair_c1_eq"].append((g, r["tag"], r["t"], p["i"], p["ma"], p["mb"]))
                    if p["c1_eef_eq"]:
                        c["pair_c1_eef_eq"] += 1
                    if p.get("first_div") is None:
                        c["pair_long_never_diverge"] += 1; anomalies["pair_long_never_diverge"].append((g, r["tag"], r["t"], p["i"]))
                    elif p["first_div"] != 17:
                        c[f"pair_first_div_{'le16' if p['first_div'] <= 16 else 'gt17'}"] += 1
                        if p["first_div"] > 17:
                            anomalies["pair_first_div_gt17"].append((g, r["tag"], r["t"], p["i"], p["ma"], p["mb"], p["first_div"]))
            c["cand_pairs"] += r["n_cand_pairs"]
            for x in r["cand_collisions"]:
                c["cand_c0_all_eq" if x["all"] else "cand_c0_eef_eq"] += 1
                anomalies["cand_collision"].append((g, r["tag"], r["t"], x))
        c["steps_mod16"] = dict(fd)
        T[g] = dict(c)
    out["tables"] = T
    out["anomalies"] = {k: v for k, v in anomalies.items()}
    for g, c in T.items():
        print(g, json.dumps(c))
    print("ANOMALIES")
    for k, v in anomalies.items():
        print(k, len(v), v[:6])
    json.dump(out, open(os.path.join(HERE, "analyze.json"), "w"), indent=0, default=str)


if __name__ == "__main__":
    main()
