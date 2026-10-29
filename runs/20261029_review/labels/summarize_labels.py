"""Summary of label_check.jsonl: per task x stage x stored done, how many rollouts were re-evaluated and how many match
on every label (done, prev_ok, steps, success, reached, progress, where the rollout stops); start-state, py_state and
k_replay checks; every mismatch with its fields; stage conditions that held between chunk boundaries only.
usage: python summarize_labels.py"""
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE  # noqa: E402


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "label_check.jsonl"))]
    err = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    why = json.load(open(os.path.join(HERE, "sample_why.json")))
    for r in ok:
        r["match"]["stop"] = r["stop_at"] == r["n_saved"] or (r["stop_at"] is None and r["n_saved"] == r["L"])
        r["why"] = next((w for c, m, w in why.get(f"{r['dir']}|{r['tag']}|{r['t']}", []) if (c, m) == (r["c"], r["m"])), "?")
    cells = collections.defaultdict(lambda: [0, 0])
    for r in ok:
        key = (r["task"], r["k"], r["stored"]["done"], "dense" if "dense" in r["dir"] else "coarse")
        cells[key][0] += 1
        cells[key][1] += all(r["match"].values())
    print(f"{len(ok)} rollouts re-evaluated at {len({(r['dir'], r['tag'], r['t']) for r in ok})} points, {len(err)} point errors")
    for key in sorted(cells):
        print(f"   {key[0]:21s} S{key[1]} done={str(key[2]):5s} {key[3]:6s}: {cells[key][1]}/{cells[key][0]} match")
    bad = [r for r in ok if not all(r["match"].values())]
    print(f"mismatching rollouts: {len(bad)}")
    for r in bad:
        print(f"   {r['tag']} t{r['t']} S{r['k']} c{r['c']}m{r['m']} {r['dir']}: fields {[k for k, v in r['match'].items() if not v]} "
              f"stored {r['stored']} re {r['re']} stop_at {r['stop_at']} n {r['n_saved']}")
    print(f"start state != main run state at t: {sum(r['start_diff'] > 0 for r in ok)}; py_state differs from the replay: "
          f"{sum(bool(r['ps_diff']) for r in ok)}; k_replay differs: {sum(r['k_replay'] != r['k_replay_row'] for r in ok)}")
    nb = [r for r in ok if not r["stored"]["done"] and r["cond_steps_n"]]
    print(f"not-done rollouts whose stage condition held at some non-boundary step: {len(nb)}")
    for r in nb:
        print(f"   {r['tag']} t{r['t']} S{r['k']} c{r['c']}m{r['m']}: held at steps {r['cond_steps_first']} (n {r['cond_steps_n']})")
    dn = [r for r in ok if r["stored"]["done"]]
    print(f"done rollouts: {len(dn)}; via the stage check at the final boundary: {sum(bool(r['cond_bound']) and r['cond_bound'][-1] == r['n_saved'] for r in dn)}; "
          f"via success only (transient): {sum(r['stored']['success'] and r['stored']['reached'] == r['k'] for r in dn)}")
    by_why = collections.Counter(r["why"].split(" done")[0] for r in ok)
    print("sample composition:", dict(by_why))
    for e in err:
        print("ERROR", e["tag"], e["t"], e["error"][:300])


if __name__ == "__main__":
    main()
