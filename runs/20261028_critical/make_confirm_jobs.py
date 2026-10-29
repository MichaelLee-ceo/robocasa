"""Confirmation jobs at the located critical chunks (critical_chunks.json from dense_analysis.py, in $CRIT_OUT): at the
start of every complete interval's largest jump with |dV| >= MIN_JUMP, a collect.py job with N = 8 candidates x M = 4
continuations. KettleBoiling was confirmed at every interval (|dV| >= 0.25), SteamInMicrowave and PackIdenticalLunches
at 0.4 (README). Writes ../20261026_critic_data/jobs/confirm_<task>.json. usage: python -I make_confirm_jobs.py <task> [min jump=0.4]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATA, JOBS, OUT  # noqa: E402


def select(task, min_jump=0.4):
    rows = [r for r in json.load(open(os.path.join(OUT, "critical_chunks.json"))) if r["task"] == task and r["complete"] and abs(r["jump"]) >= min_jump]
    base = json.load(open(os.path.join(DATA, JOBS[task])))
    jobs = []
    for r in rows:
        tmpl = next(j for j in base if j["tag"] == r["tag"] and j["k"] == r["k"])
        jobs.append(dict(tmpl, t=r["ta"], N=8, M=4))
    return list({(j["tag"], j["t"]): j for j in jobs}.values()), sum(bool(r.get("noise")) for r in rows)


def main():
    task = sys.argv[1]
    jobs, noise = select(task, float(sys.argv[2]) if len(sys.argv) > 2 else 0.4)
    json.dump(jobs, open(os.path.join(DATA, "jobs", f"confirm_{task}.json"), "w"), indent=0)
    print(task, len(jobs), f"confirmation points ({noise} of the jumps within 2 SE of noise)")


if __name__ == "__main__":
    main()
