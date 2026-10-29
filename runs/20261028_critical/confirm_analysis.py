"""At the located critical chunks (dense_analysis.py): does the choice of chunk there change the stage outcome, and can
the difference be seen?

Each point (data.py "confirm": the located critical chunks; "control": random chunks) has 8 candidate chunks with 4
continuations each. Per point:
  V        completion rate of all 32 rollouts
  var_b    between-candidate variance of P(done): sample variance of the candidates' rates minus their mean binomial
           variance p(1 - p) / (M - 1), so luck is taken out
  gain     gain of a perfect best-of-8 picker at this point, from its V and var_b (effect_by_event.best_of_8)
  cv_gain  cross-validated gain of picking: the candidate with the best rate over continuations 1-2, scored on
           continuations 3-4, minus V over continuations 3-4 (no winner's curse; continuations 3-4 never reuse the seeds
           of the rollouts that located the point)
  top/bottom  the best and worst candidates' rates over all 4
The summary leads with the mean cross-validated gain, then the mean per-point gain. The pooled 142 sqrt(mean var_b) is
the gain at a point of average variance and overstates the mean gain (Jensen).
Also draws overlay_confirm/<tag>_t<t>.png (overlay_critical.py), from the rollouts and snapshot of the row in use:
blue = the candidate finished in >= 3 of 4, orange = in <= 1, grey = 2. Outputs go to $CRIT_OUT.
usage: [CONFIRM_DS=confirm|control] python -I confirm_analysis.py <task> [workers for the overlays]
(outputs <ds>_<task>.json and overlay_<ds>/)"""
import json, multiprocessing, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
from data import OUT  # noqa: E402
from effect_by_event import best_of_8  # noqa: E402


DS = os.environ.get("CONFIRM_DS", "confirm")  # confirm: the critical chunks; control: random chunks, same protocol


def load(task):
    return data.rows(task, DS)


def main():
    task = sys.argv[1]
    crit = {(r["tag"], r["ta"]): r for r in json.load(open(os.path.join(OUT, "critical_chunks.json"))) if r["task"] == task}
    out, rows = [], load(task)
    for key, r in sorted(rows.items()):
        y = data.outcomes(r)  # 8 x 4
        p = y.mean(1); M = y.shape[1]
        var_b = p.var(ddof=1) - np.mean(p * (1 - p)) / (M - 1)
        first, second = y[:, :2].mean(1), y[:, 2:].mean(1)
        pick = np.flatnonzero(first == first.max())
        cv = float(second[pick].mean() - second.mean())
        gain = float(best_of_8(y.mean(), var_b)[0])
        c = crit.get(key, {})
        out.append(dict(tag=key[0], t=key[1], k=r["k"], V=float(y.mean()), var_b=float(var_b), gain=gain, cv_gain=cv, rates=p.tolist(),
                        jump=c.get("jump"), tb=c.get("tb"), noise=c.get("noise"), hidden=r.get("hidden"), dir=data.rel(r["_dir"])))
        print(f"{key[0]:26s} t{key[1]:4d} S{r['k']}  V {y.mean():.2f}  candidates' rates {' '.join(f'{x:.2f}' for x in sorted(p, reverse=True))}  "
              f"var_b {var_b:+.3f}  cv gain {cv:+.2f}  gain {100 * gain:3.0f} pts   (dense jump {c.get('jump', float('nan')):+.2f} to t{c.get('tb')}"
              f"{', within 2 SE of noise' if c.get('noise') else ''})")
    if out:
        vb, g, cv = (np.array([o[k] for o in out]) for k in ("var_b", "gain", "cv_gain"))
        rng = np.random.default_rng(0)
        ci = lambda x: np.percentile([rng.choice(x, len(x)).mean() for _ in range(2000)], [2.5, 97.5]) * 100
        print(f"{DS} {len(out)} points: mean cross-validated gain {100 * cv.mean():+.0f} pts [{ci(cv)[0]:+.0f}, {ci(cv)[1]:+.0f}]; "
              f"mean perfect best-of-8 gain per point {100 * g.mean():.0f} pts [{ci(g)[0]:.0f}, {ci(g)[1]:.0f}]; "
              f"mean var_b {vb.mean():+.3f} (pooled 142 sqrt(mean var_b): {142 * np.sqrt(max(vb.mean(), 0)):.0f} pts)")
    json.dump(out, open(os.path.join(OUT, f"{DS}_{task}.json"), "w"), indent=0)
    if len(sys.argv) > 2:
        from overlay_critical import work
        js = []
        for o in out:
            r = rows[(o["tag"], o["t"])]
            per = [2 if x >= 0.75 else 0 if x <= 0.25 else 1 for x in o["rates"]]
            js.append((dict(task=task, tag=o["tag"], t=o["t"], k=o["k"], pd=o["V"], per=per, scene=data.scenes()[o["tag"]], outdir=f"overlay_{DS}",
                             legend="candidates finishing in >= 3 / 2 / <= 1 of 4 continuations"), data.roll(r), data.snap(r)))
        with multiprocessing.get_context("spawn").Pool(int(sys.argv[2])) as pool:
            for r in pool.imap_unordered(work, js):
                print(r.get("tag"), r.get("t"), r.get("error", "ok"))


if __name__ == "__main__":
    main()
