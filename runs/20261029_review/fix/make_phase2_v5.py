"""Phase 2 of the settled-v5 study: does the choice of chunk matter more at the critical chunks than at random chunks?
Critical: every relocated jump (critical_chunks.json from dense_analysis.py on the v5 data) that is complete, one chunk
wide (tb - ta = 16), |jump| >= 0.4 (one threshold for all tasks) and above 2 SE. A job at its start ta.
Control: as many uniform decision points of the same task (the coarse rows in use), drawn at random (seed 0), not within
32 steps of a critical chunk start of the same episode.
Both: 8 candidates x 4 continuations, continuation seeds salted ("c5") so they never replay earlier rollouts, settled-v5;
outputs run_confirm5_<task> and run_control5_<task> (data.py datasets confirm5 / control5).
Writes the jobs file and phase2_points.json (the selection, for the record). usage: python -I make_phase2_v5.py jobs_out.json"""
import json, os, random, sys
sys.path.insert(0, "/home/michaellee/mclee/robocasa/runs/20261028_critical")
import data  # noqa: E402

JUMP, SALT = 0.4, "c5"
HERE = os.path.dirname(os.path.abspath(__file__))
crit = json.load(open(os.path.join(data.OUT, "critical_chunks.json")))
base = {t: {(j["tag"], j["t"]): j for j in json.load(open(os.path.join(data.DATA, f)))} for t, f in data.JOBS.items()}
jobs, record = [], {}
rng = random.Random(0)
for task in data.TASKS:
    sel = [c for c in crit if c["task"] == task and c["complete"] and c["width"] == 16 and abs(c["jump"]) >= JUMP and not c["noise"]]
    starts = {(c["tag"], c["ta"]) for c in sel}
    for c in sel:
        tmpl = next(j for (tag, _), j in base[task].items() if tag == c["tag"] and j["k"] == c["k"])
        jobs.append(dict(tmpl, t=c["ta"], N=8, M=4, salt=SALT, out=f"{data.DATA}/run_confirm5_{task}", replaces="(phase 2: critical)"))
    coarse = data.rows(task, "coarse")
    pool = sorted(k for k in coarse if not any(tag == k[0] and abs(ta - k[1]) <= 32 for tag, ta in starts))
    ctrl = rng.sample(pool, min(len(sel), len(pool)))
    for tag, t in ctrl:
        r = coarse[(tag, t)]
        tmpl = base[task][(tag, t)] if (tag, t) in base[task] else next(j for (tg, _), j in base[task].items() if tg == tag and j["k"] == r["k"])
        jobs.append(dict(tmpl, t=t, k=r["k"], L=r["L"], N=8, M=4, salt=SALT, out=f"{data.DATA}/run_control5_{task}", replaces="(phase 2: control)"))
    record[task] = dict(critical=[dict(tag=c["tag"], k=c["k"], ta=c["ta"], tb=c["tb"], jump=c["jump"], se=c["se"]) for c in sel],
                        control=[dict(tag=tag, t=t, k=coarse[(tag, t)]["k"]) for tag, t in ctrl])
    print(f"{task:21s} critical {len(sel):3d}, control {len(ctrl):3d}")
json.dump(record, open(os.path.join(HERE, "phase2_points.json"), "w"), indent=1)
json.dump(jobs, open(sys.argv[1], "w"), indent=1)
print(len(jobs), "jobs ->", sys.argv[1])
