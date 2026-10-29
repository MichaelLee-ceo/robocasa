"""Side check: the continuations m >= 1 of a candidate replay its stored first chunk from the same restored state, so their
first 16 steps must equal m0's exactly (any difference = state leaking between rollouts, as in bug 1).
Per role and dir: pairs checked, pairs with any difference in steps 1-16 (eef, finger gap, objects, base), max difference.
usage: python -I pairs.py"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics  # noqa: E402


def main():
    rows = []
    for d in metrics.load():
        for p in d["pts"]:
            by = {(r["c"], r["m"]): r for r in p["rolls"]}
            for (c, m), r in by.items():
                if m == 0 or (c, 0) not in by:
                    continue
                r0 = by[(c, 0)]
                n = min(17, r["T"], r0["T"])
                diff = max(float(np.abs(r[k][:n] - r0[k][:n]).max()) for k in ("eef_w", "gap", "obj", "base"))
                rows.append(dict(task=d["task"], role=p["role"], dir=p["dir"].split("/")[0], hidden=p["hidden"], tag=d["tag"], t=p["t"],
                                 c=c, m=m, n=n, diff=diff))
    df = pd.DataFrame(rows)
    g = df.groupby(["task", "role", "dir", "hidden"], dropna=False).agg(pairs=("diff", "size"), differ=("diff", lambda x: int((x > 0).sum())),
                                                                       max_diff=("diff", "max")).reset_index()
    print(g.to_string(index=False))
    df.to_csv(os.path.join(HERE, "pairs.csv.gz"), index=False)


if __name__ == "__main__":
    main()
