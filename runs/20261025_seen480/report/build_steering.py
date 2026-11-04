"""Data for the report's "Steering" tab (index.html#steering): chunk-by-chunk closed-loop steering with the simulator as
oracle (runs/20261029_closedloop), against the plain policy and the stage-level search (the "Upper bound" tab).

Per task: the planned hard scenes (closedloop/check.py PLANNED) and, per episode, its outcome, cost and one code per
chunk: S = steered (some of the 8 candidates finished the stage, the earliest finisher was executed), A = all 8
finished, N = none finished (both execute candidate 0); with the chunk's stage. Episodes still running are read from
their log (status "running"). Writes report/steering.js (window.STEER). usage: python report/build_steering.py"""
import glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.dirname(os.path.dirname(HERE))
CL = f"{RUNS}/20261029_closedloop"
sys.path.insert(0, CL)
import check  # noqa: E402  (PLANNED scenes, plain baselines and imagine results, consistency checks)

HORIZON = json.load(open(f"{RUNS}/horizons_original_plus.json"))


def code(c, n):
    return "S" if c["steered"] else ("A" if c["n_done"] == n else "N")


def main():
    P, I = check.plain(), check.imagine()
    tasks = []
    for task, scenes in check.PLANNED.items():
        eps = []
        for s in scenes:
            tag = f"{task}_s{s}_single"
            out, log = f"{CL}/out/{tag}.json", f"{CL}/logs/{tag}.log"
            p = P.get((task, s), [])
            e = dict(seed=s, plain=p, imagine=I.get((task, s)))
            if os.path.exists(out):
                r = json.load(open(out))
                L = r["log"]
                e.update(status="done", ok=bool(r["success"]), steps=r["steps"], chunks=r["chunks"], steered=r["steered"],
                         plain_fail=r["steered_plain_would_fail"], calls=r["policy_calls"], checks=check.check(r),
                         codes="".join(code(c, r["N"]) for c in L), stages="".join(str(c["k"]) for c in L))
            elif os.path.exists(log):
                L = [json.loads(l) for l in open(log) if l.startswith('{"t"')]
                e.update(status="running", t=L[-1]["t"] + 16 if L else 0, chunks=len(L), steered=sum(c["steered"] for c in L),
                         codes="".join(code(c, 8) for c in L), stages="".join(str(c["k"]) for c in L))
            else:
                e.update(status="missing")
            eps.append(e)
        done = [e for e in eps if e["status"] == "done"]
        tasks.append(dict(task=task, n_stages=check.STAGES[task], horizon=HORIZON[task], episodes=eps, summary=dict(
            n=len(eps), done=len(done), ok=sum(e["ok"] for e in done),
            plain=sum(sum(e["plain"]) for e in eps) / max(sum(len(e["plain"]) for e in eps), 1),
            imagine=sum(e["imagine"] == 1 for e in eps), imagine_n=sum(e["imagine"] in (0, 1) for e in eps),
            chunks=sum(e["chunks"] for e in done), steered=sum(e["steered"] for e in done),
            plain_fail=sum(e["plain_fail"] for e in done),
            calls=sum(e["calls"] for e in done) / max(len(done), 1),
            all_checks_ok=all(not e["checks"] for e in done))))
    S = dict(updated=time.strftime("%Y-%m-%d %H:%M"), tasks=tasks)
    with open(f"{HERE}/steering.js", "w") as f:
        f.write("window.STEER = ")
        json.dump(S, f, separators=(",", ":"))
        f.write(";\n")
    for t in tasks:
        s = t["summary"]
        print(f"{t['task']}: {s['done']}/{s['n']} done, ours {s['ok']}/{s['done']}, plain {100 * s['plain']:.1f} %, imagine {s['imagine']}/{s['imagine_n']}, "
              f"steered {s['steered']}/{s['chunks']}, plain-fail {s['plain_fail']}, calls/episode {s['calls']:.0f}, checks ok {s['all_checks_ok']}")


if __name__ == "__main__":
    main()
