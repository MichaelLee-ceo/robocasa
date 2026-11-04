"""Consistency checks and summary of the closed-loop episodes (out/<tag>.json).
Per episode, every chunk must be consistent:
  - t advances by 16 and the stage k never goes down;
  - steered <=> some but not all candidates finished (0 < n_done < N);
  - unsteered chunks execute candidate 0 (a steered pick is a finisher by construction; it may finish exactly at its
    budget's last boundary, e.g. KettleBoiling s19 t16, so its steps can equal the budget);
  - success <=> the reward fired or every stage check held (episodes from 2026-10-30 log both); a failed episode does not
    pass the official check at its end;
  - a successful episode ends in the last stage.
The live check that the executed chunk equals the imagined one is in closedloop.py (it raises).
Summary per task: success against the plain policy (4 seeds per scene) and imagine on the same scenes; chunks, steered
chunks, steered chunks where candidate 0 would have failed, and policy calls per episode.
Plain: policy_seed 0 from ../20261025_seen480/run, seeds 1-3 from ../20261026_imagine/{full,multi} (condition baseline).
Imagine: ../20261026_imagine/full_v2 (KettleBoiling), ../20261026_imagine/sim_pil (the other two, re-run 2026-10-30).
usage: python3 check.py"""
import collections, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)
STAGES = {"KettleBoiling": 2, "SteamInMicrowave": 4, "PackIdenticalLunches": 4}
PLANNED = {"KettleBoiling": [2, 4, 5, 8, 9, 10, 12, 17, 19, 24],  # scenes where the plain policy fails most (4 seeds)
           "SteamInMicrowave": [0, 2, 3, 5, 8, 12, 14, 15, 27, 28],
           "PackIdenticalLunches": [0, 1, 2, 4, 5, 10, 11, 13, 14, 15]}
IMAGINE = {"KettleBoiling": "full_v2", "SteamInMicrowave": "sim_pil", "PackIdenticalLunches": "sim_pil"}


def scene_of(tag):
    return int(tag.split("_s")[1].split("_")[0])


def jsonl(pattern):
    out = []
    for p in glob.glob(pattern):
        for l in open(p):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:  # a line cut off by a kill
                pass
    return out


def plain():
    """{(task, scene): [success of each plain seed]}"""
    out = collections.defaultdict(list)
    for r in jsonl(f"{R}/20261025_seen480/run/results_w*.jsonl"):
        if r.get("task") in STAGES:
            out[r["task"], int(r["scene"].split("_s")[-1].split(".")[0])].append(int(r["success"]))
    for r in jsonl(f"{R}/20261026_imagine/full/results*.jsonl") + jsonl(f"{R}/20261026_imagine/multi/results*.jsonl"):
        if r.get("condition") == "baseline" and "error" not in r:
            out[r["task"], scene_of(r["tag"])].append(int(r["success"]))
    return out


def imagine():
    """{(task, scene): success of the stage-level search, or "ERROR" when it only has error rows}"""
    out = {}
    for task, d in IMAGINE.items():
        for r in jsonl(f"{R}/20261026_imagine/{d}/results*.jsonl"):
            if r.get("condition") != "imagine" or not r.get("tag", "").startswith(task + "_s"):
                continue
            k = task, scene_of(r["tag"])
            if "error" not in r:
                out[k] = int(r["success"])
            else:
                out.setdefault(k, "ERROR")
    return out


def check(r):
    bad, log = [], r["log"]
    for i, c in enumerate(log):
        if c["t"] != 16 * i:
            bad.append(f"chunk {i}: t {c['t']}")
        if i and c["k"] < log[i - 1]["k"]:
            bad.append(f"chunk {i}: stage went down")
        crit = 0 < c["n_done"] < r["N"]
        if c["steered"] != crit:
            bad.append(f"chunk {i}: steered {c['steered']} but n_done {c['n_done']}")
        if not c["steered"] and c["pick"] != 0:
            bad.append(f"chunk {i}: unsteered pick {c['pick']}")
    if "reward_success" in r:
        if r["success"] != (r["reward_success"] or r["all_stages"]):
            bad.append(f"success {r['success']} vs reward {r['reward_success']} / all stages {r['all_stages']}")
    elif r["success"] != r["final_check"]:
        bad.append(f"success {r['success']} vs final check {r['final_check']}")
    if not r["success"] and r["final_check"]:
        bad.append("failed but the official check holds at the end")
    if r["success"] and r["stage"] < STAGES[r["task"]] - 1:
        bad.append(f"success in stage {r['stage']}")
    return bad


def main():
    P, I = plain(), imagine()
    rs = sorted((json.load(open(p)) for p in glob.glob(os.path.join(HERE, "out", "*.json"))),
                key=lambda r: (r["task"], scene_of(r["tag"])))
    launched = {os.path.basename(p).split("_s")[0] for p in glob.glob(os.path.join(HERE, "logs", "*.log"))}
    for task in sorted({r["task"] for r in rs} | (launched & set(PLANNED))):
        tr = [r for r in rs if r["task"] == task]
        missing = sorted(set(PLANNED[task]) - {scene_of(r["tag"]) for r in tr})
        if missing:
            print(f"{task}: MISSING closed-loop episodes for scenes {missing} (failed twice or still running)")
        if not tr:
            continue
        print(f"{task}\n{'scene':6s} {'ours':>5s} {'plain':>6s} {'imagine':>7s} {'steps':>5s} {'chunks':>6s} {'steered':>7s} {'plain-fail':>10s} {'calls':>6s}  checks")
        for r in tr:
            s = scene_of(r["tag"]); bad = check(r); p = P.get((task, s), [])
            print(f"s{s:<5d} {int(r['success']):5d} {sum(p):3d}/{len(p):<2d} {str(I.get((task, s), '-')):>7s} {r['steps']:5d} {r['chunks']:6d} "
                  f"{r['steered']:7d} {r['steered_plain_would_fail']:10d} {r['policy_calls']:6d}  {'ok' if not bad else '; '.join(bad[:3])}")
        n = len(tr); ours = sum(r["success"] for r in tr)
        ps = [x for r in tr for x in P.get((task, scene_of(r["tag"])), [])]
        im = [I[task, scene_of(r["tag"])] for r in tr if I.get((task, scene_of(r["tag"]))) in (0, 1)]
        print(f"{n} episodes: ours {ours}/{n} = {ours / n:.0%}; plain {sum(ps)}/{len(ps)} = {sum(ps) / max(len(ps), 1):.1%}; "
              f"imagine {sum(im)}/{len(im)}. Steered {sum(r['steered'] for r in tr)} of {sum(r['chunks'] for r in tr)} chunks "
              f"({sum(r['steered_plain_would_fail'] for r in tr)} where candidate 0 would have failed); "
              f"policy calls per episode {sum(r['policy_calls'] for r in tr) / n:.0f}\n")


if __name__ == "__main__":
    main()
