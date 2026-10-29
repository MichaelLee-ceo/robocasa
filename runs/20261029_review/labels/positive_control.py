"""Positive control for the swing detector of vseries.py: run it on the superseded rows (before the bug 2/3/4 fixes),
where the known bugs produced V swings. If the detector finds them there and nothing in the current rows, the absence
of flags in the current rows means something.
  old SteamInMicrowave: coarse run_SteamInMicrowave_v2 only, dense run_dense_SteamInMicrowave_bug3
  old PackIdenticalLunches: coarse run_PackIdenticalLunches only, dense run_dense_PackIdenticalLunches_bug3
  old KettleBoiling: coarse run_v2/s* only (bug 2 hit 1 point), dense run_dense_KettleBoiling
usage: python positive_control.py"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_rows, rolls  # noqa: E402
from vseries import series, flags_of, worst_p  # noqa: E402

OLD = {"KettleBoiling": (["run_v2/s*"], ["run_dense_KettleBoiling"]),
       "SteamInMicrowave": (["run_SteamInMicrowave_v2"], ["run_dense_SteamInMicrowave_bug3"]),
       "PackIdenticalLunches": (["run_PackIdenticalLunches"], ["run_dense_PackIdenticalLunches_bug3"])}


def main():
    for task, (cd, dd) in OLD.items():
        pts = {}
        for src, dirs in (("coarse", cd), ("dense", dd)):
            for d in dirs:
                for r in read_rows(d):
                    y = [x["done"] for _, _, x in rolls(r)]
                    pts[(r["tag"], r["t"])] = dict(task=task, tag=r["tag"], t=r["t"], k=r["k"], n=len(y), x=int(sum(y)),
                                                   V=float(np.mean(y)), src=src, dir=r["_dir"], hidden=r.get("hidden"), V_ok=0)
        by = series(list(pts.values()))
        fl = []
        for key, s in by.items():
            for f in flags_of(s):
                f["p"] = worst_p(s, f)
                fl.append((key, f, s))
        print(f"{task}: {len(pts)} old points, flags: DIP {sum(f['kind'] == 'DIP' for _, f, _ in fl)}, "
              f"SPIKE {sum(f['kind'] == 'SPIKE' for _, f, _ in fl)}, JR {sum(f['kind'] == 'JR' for _, f, _ in fl)}")
        for key, f, s in sorted(fl, key=lambda z: (z[0][1], z[1]["idx"][0])):
            print(f"    {f['kind']:5s} {key[1]:30s} S{key[2]} p_worst {f['p']:.1e}  " +
                  " ".join(f"{s[i]['t']}:{s[i]['x']}/{s[i]['n']}" for i in f["idx"]))


if __name__ == "__main__":
    main()
