"""Score a critic's picks against the measured outcomes (2026-10-30).

Input: <answers dir>/<pid>.json ({"pick", "ranking", "confidence", "reasoning"}), key.json and truth.json (each
candidate's success rate over its 4 continuations). The critic never saw the outcomes, so the rate of its pick is an
unbiased estimate of that candidate's success rate.
Per point: gain = rate(pick) - mean rate of the 8 (a random pick); ceiling = best rate - mean (the best of 8 measured
with 4 continuations; optimistic, it includes luck); Spearman correlation of the ranking with the rates.
Reported for the critical points (confirm) and the controls, per task, with 95 % bootstrap intervals over points; and
for the points where the candidates differ at all.
usage: python3 score.py <answers dir> [label]"""
import glob, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def spearman(rank, rates):
    """correlation between the critic's ranking (best first) and the rates (ties averaged)"""
    n = len(rates)
    pos = {c: i for i, c in enumerate(rank)}
    a = [-(pos[c]) for c in range(n)]  # higher = ranked better
    def rk(v):
        order = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]: j += 1
            for k in range(i, j + 1): r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rk(a), rk(rates)
    ma, mb = sum(ra) / n, sum(rb) / n
    va, vb = sum((x - ma) ** 2 for x in ra), sum((x - mb) ** 2 for x in rb)
    return None if va == 0 or vb == 0 else sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (va * vb) ** 0.5


def boot(xs, n=4000, seed=780):
    if not xs:
        return None, None
    rnd = random.Random(seed); m = len(xs)
    s = sorted(sum(rnd.choice(xs) for _ in range(m)) / m for _ in range(n))
    return s[int(0.025 * n)], s[int(0.975 * n)]


def main():
    d = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(d.rstrip("/"))
    key, truth = json.load(open(f"{HERE}/key.json")), json.load(open(f"{HERE}/truth.json"))
    rows = []
    for p in sorted(glob.glob(f"{d}/p*.json")):
        pid = os.path.basename(p)[:-5]
        try:
            a = json.load(open(p))
        except json.JSONDecodeError:
            print(f"{pid}: unreadable answer"); continue
        rates = truth[pid]; m = sum(rates) / 8
        rows.append(dict(pid=pid, **key[pid], pick=a["pick"], conf=a.get("confidence"), gain=rates[a["pick"]] - m,
                         ceil=max(rates) - m, spread=max(rates) - min(rates) > 0, rho=spearman(a["ranking"], rates)))
    print(f"{label}: {len(rows)} answers")
    for name, sel in [("critical (confirm)", lambda r: r["dataset"] == "confirm"), ("control", lambda r: r["dataset"] == "control"),
                      ("critical, candidates differ", lambda r: r["dataset"] == "confirm" and r["spread"])] + \
                     [(f"critical {t}", (lambda t: lambda r: r["dataset"] == "confirm" and r["task"] == t)(t))
                      for t in ("KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches")]:
        g = [r for r in rows if sel(r)]
        if not g:
            continue
        gains = [100 * r["gain"] for r in g]; lo, hi = boot(gains)
        rho = [r["rho"] for r in g if r["rho"] is not None]
        print(f"  {name:32s} n {len(g):2d}  gain {sum(gains) / len(g):+5.1f} pts [{lo:+.1f}, {hi:+.1f}]   "
              f"ceiling {100 * sum(r['ceil'] for r in g) / len(g):4.1f}   Spearman {sum(rho) / len(rho) if rho else float('nan'):+.2f} (n {len(rho)})")
    json.dump(rows, open(f"{HERE}/score_{label}.json", "w"), indent=0)


if __name__ == "__main__":
    main()
