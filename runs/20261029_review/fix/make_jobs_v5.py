"""Jobs that re-collect given decision points with the settled-v5 restore, each with the N, M, L, k and seeds of the row
it replaces (salt "" keeps the original seeds, so old and new rows of a point pair up) and an "out" dir per dataset.
usage: python make_jobs_v5.py points.json jobs_out.json   (points: [{dir, tag, t, out}] or affected rows)"""
import glob, json, sys

D = "/home/michaellee/mclee/robocasa/runs/20261026_critic_data"


def row(d, tag, t):
    for p in glob.glob(f"{D}/{d}/results_w*.jsonl"):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("tag") == tag and r.get("t") == t and "error" not in r:
                return r
    raise KeyError((d, tag, t))


def scenes():
    sc = {}
    for p in glob.glob(f"{D}/jobs*.json"):
        for j in json.load(open(p)):
            sc.setdefault(j["tag"], j["scene"])
    return sc


def main():
    pts, outp = json.load(open(sys.argv[1])), sys.argv[2]
    sc = scenes()
    jobs = []
    for p in pts:
        r = row(p["dir"], p["tag"], p["t"])
        j = {k: r[k] for k in ("task", "tag", "t", "k", "L", "N", "M", "episode_success")}
        j.update(scene=sc[r["tag"]], out=p["out"], salt=p.get("salt", ""), replaces=p["dir"])
        jobs.append(j)
    json.dump(jobs, open(outp, "w"), indent=1)
    print(len(jobs), "jobs ->", outp)


if __name__ == "__main__":
    main()
