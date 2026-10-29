"""Coverage and bookkeeping checks (rows and file names only; no npz is opened).

- current rows per the override order of effect_by_event.points(); every job of each jobs file has exactly one
  error-free current row; the points of the jobs_fix* lists are current from their fix dir
- superseded rows, error rows that nothing replaces, duplicate error-free rows (same dir, or same override level),
  rows for points that no jobs file lists, unparseable lines
- row fields vs the job (k, L, N, M, episode_success, task) and vs the main run (stage history, success, length)
- hidden tag per dir
- orphan roll / snap files (no error-free row in that dir), roll files outside the row's N x M, foreign file names
Writes coverage.json. usage: python -I coverage.py"""
import collections, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, SEEN, HERE, COARSE, COARSE_JOBS, FIX_JOBS, OTHER, COARSE_HIDDEN, L_TABLE, read_rows  # noqa: E402

PAT = re.compile(r"^(?P<tag>.+)_t(?P<t>\d+)_c(?P<c>\d+)_m(?P<m>\d+)\.(?P<ext>npz|mp4)$")
SPAT = re.compile(r"^(?P<tag>.+)_t(?P<t>\d+)\.npz$")


def main_runs():
    out = {}
    for p in glob.glob(f"{SEEN}/run/results_w*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            if "error" not in r and r.get("task") in L_TABLE:
                out[r["tag"]] = r
    return out


def files(d):
    roll, snap, foreign = collections.defaultdict(list), {}, []
    rd, sd = f"{DATA}/{d}/roll", f"{DATA}/{d}/snap"
    for fn in (os.listdir(rd) if os.path.isdir(rd) else []):
        mm = PAT.match(fn)
        if mm:
            roll[(mm["tag"], int(mm["t"]))].append((int(mm["c"]), int(mm["m"]), mm["ext"]))
        else:
            foreign.append(f"roll/{fn}")
    for fn in (os.listdir(sd) if os.path.isdir(sd) else []):
        mm = SPAT.match(fn)
        if mm:
            snap[(mm["tag"], int(mm["t"]))] = fn
        else:
            foreign.append(f"snap/{fn}")
    return roll, snap, foreign


def main():
    mr = main_runs()
    out = dict(dirs={}, tasks={}, issues=[])
    issue = lambda kind, **kw: out["issues"].append(dict(kind=kind, **kw))
    jobs_by_file = {f: json.load(open(f"{DATA}/{f}")) for f in
                    list(COARSE_JOBS.values()) + list(FIX_JOBS) + [v[2] for v in OTHER.values() if v[2]]}
    alldirs = [(d, task, f"coarse{lvl}") for task, lv in COARSE.items() for lvl, ds in enumerate(lv) for d in ds] + \
              [(d, v[1], v[0]) for d, v in OTHER.items()]
    rows_of = {}
    for d, task, role in alldirs:
        rows = read_rows(d)
        rows_of[d] = rows
        good = [(f, i, r) for f, i, r in rows if r is not None and "error" not in r]
        err = [(f, i, r) for f, i, r in rows if r is not None and "error" in r]
        bad = [(f, i) for f, i, r in rows if r is None]
        hid = collections.Counter(r.get("hidden") for _, _, r in good)
        keys = collections.Counter((r["tag"], r["t"]) for _, _, r in good)
        dups = {f"{k[0]} t{k[1]}": n for k, n in keys.items() if n > 1}
        roll, snap, foreign = files(d)
        goodk = {(r["tag"], r["t"]): r for _, _, r in good}
        orph_roll = {k: v for k, v in roll.items() if k not in goodk}
        orph_snap = [k for k in snap if k not in goodk]
        outside = {}
        missing = {}
        for k, r in goodk.items():
            have = {(c, m, e) for c, m, e in roll.get(k, [])}
            want = {(c, m, e) for c in range(r["N"]) for m in range(r["M"]) for e in ("npz", "mp4")}
            if have - want:
                outside[f"{k[0]} t{k[1]}"] = sorted(have - want)
            if want - have:
                missing[f"{k[0]} t{k[1]}"] = sorted(want - have)
            if k not in snap:
                missing.setdefault(f"{k[0]} t{k[1]}", []).append("snap")
        # error rows: replaced by an error-free row in the same dir?
        err_unrepl = [dict(tag=r["tag"], t=r["t"], file=f, line=i, error=r["error"][:120]) for f, i, r in err
                      if (r["tag"], r["t"]) not in goodk]
        claims = sorted(os.listdir(f"{DATA}/{d}/claims")) if os.path.isdir(f"{DATA}/{d}/claims") else []
        out["dirs"][d] = dict(task=task, role=role, rows=len(rows), good=len(good), error=len(err), unparseable=bad,
                              hidden=dict(hid), dup_rows=dups, roll_files=sum(len(v) for v in roll.values()),
                              snap_files=len(snap), foreign_files=foreign[:20], n_foreign=len(foreign),
                              orphan_roll_points={f"{k[0]} t{k[1]}": len(v) for k, v in orph_roll.items()},
                              orphan_snap=[f"{k[0]} t{k[1]}" for k in orph_snap], outside_NxM=outside,
                              missing_files=missing, error_rows_not_replaced_in_dir=err_unrepl,
                              error_rows=[dict(tag=r["tag"], t=r["t"], file=f, line=i, error=r["error"][:100]) for f, i, r in err],
                              claims=len(claims), claims_without_row=[c for c in claims if (c.rsplit("_t", 1)[0], int(c.rsplit("_t", 1)[1])) not in goodk][:50])
        # expected hidden tag
        exp = None
        if role.startswith("coarse"):
            exp = COARSE_HIDDEN[int(role[-1])]
        elif role != "bug1":
            exp = OTHER[d][3]
        for f, i, r in good:
            e = exp
            if d == "run_SteamInMicrowave_fix3" and (r["tag"], r["t"]) in {(j["tag"], j["t"]) for j in jobs_by_file["jobs_fix4_SteamInMicrowave.json"]}:
                e = "settled-v4"
            if e is not None and r.get("hidden") != e:
                issue("hidden_tag", dir=d, tag=r["tag"], t=r["t"], hidden=r.get("hidden"), expected=e)
        if dups:
            issue("duplicate_rows_in_dir", dir=d, dups=dups)
        if bad:
            issue("unparseable_lines", dir=d, lines=bad)

    # ---- coarse chain: current rows
    for task, levels in COARSE.items():
        jobs = {(j["tag"], j["t"]): j for j in jobs_by_file[COARSE_JOBS[task]]}
        hist = collections.defaultdict(list)  # (tag,t) -> [(level, dir, file, line, row)]
        for lvl, ds in enumerate(levels):
            for d in ds:
                for f, i, r in rows_of[d]:
                    if r is not None and "error" not in r:
                        hist[(r["tag"], r["t"])].append((lvl, d, f, i, r))
        cur = {k: v[-1] for k, v in hist.items()}
        # same-level duplicates (e.g. two KettleBoiling shards)
        for k, v in hist.items():
            lv = collections.Counter(x[0] for x in v)
            for l_, n in lv.items():
                if n > 1:
                    issue("duplicate_rows_same_level", task=task, tag=k[0], t=k[1], level=l_, where=[(x[1], x[2], x[3]) for x in v if x[0] == l_])
        missing = [k for k in jobs if k not in cur]
        extra = [k for k in cur if k not in jobs]
        # fix lists
        fixck = {}
        for jf, fd in FIX_JOBS.items():
            if not jf.endswith(f"_{task}.json"):
                continue
            pts = [(j["tag"], j["t"]) for j in jobs_by_file[jf]]
            notcur = []
            for k in pts:
                if k not in cur:
                    notcur.append((k, None)); continue
                if cur[k][1] != fd:
                    # acceptable only when a later fix list also holds the point and its dir is current
                    in_later = [f2 for f2, fd2 in FIX_JOBS.items() if f2.endswith(f"_{task}.json") and fd2 == cur[k][1]
                                and k in {(j["tag"], j["t"]) for j in jobs_by_file[f2]}]
                    if not in_later:
                        notcur.append((k, cur[k][1]))
            fixck[jf] = dict(points=len(pts), not_current_from_fix_dir=[f"{k[0]} t{k[1]} (current: {c})" for k, c in notcur])
            if notcur:
                issue("fix_point_not_current_from_fix_dir", jobs=jf, points=fixck[jf]["not_current_from_fix_dir"])
        # rows in fix dirs that no fix list holds
        fixpts = {}
        for jf, fd in FIX_JOBS.items():
            if jf.endswith(f"_{task}.json"):
                fixpts.setdefault(fd, set()).update((j["tag"], j["t"]) for j in jobs_by_file[jf])
        for lvl, ds in enumerate(levels):
            for d in ds:
                if d in fixpts:
                    ex = [(r["tag"], r["t"]) for f, i, r in rows_of[d] if r is not None and "error" not in r and (r["tag"], r["t"]) not in fixpts[d]]
                    if ex:
                        issue("row_in_fix_dir_not_in_fix_list", dir=d, points=[f"{a} t{b}" for a, b in ex])
        # row vs job fields; row vs main run
        fieldbad = []
        for k, v in cur.items():
            r = v[4]; j = jobs.get(k)
            if j:
                for f_ in ("k", "L", "N", "M", "episode_success", "task"):
                    if r.get(f_) != j[f_]:
                        fieldbad.append(dict(tag=k[0], t=k[1], field=f_, row=r.get(f_), job=j[f_], dir=v[1]))
            m = mr.get(k[0])
            if m:
                kk = sum(tk <= k[1] for tk, _ in m["history"])
                if kk != r.get("k"):
                    fieldbad.append(dict(tag=k[0], t=k[1], field="k_vs_main_history", row=r.get("k"), job=kk, dir=v[1]))
                if m["success"] != r.get("episode_success"):
                    fieldbad.append(dict(tag=k[0], t=k[1], field="episode_success_vs_main", row=r.get("episode_success"), job=m["success"], dir=v[1]))
                if k[1] >= m["steps"]:
                    fieldbad.append(dict(tag=k[0], t=k[1], field="t_beyond_main_run", row=k[1], job=m["steps"], dir=v[1]))
            if r.get("L") != L_TABLE[task][r["k"]]:
                fieldbad.append(dict(tag=k[0], t=k[1], field="L_vs_stage_table", row=r.get("L"), job=L_TABLE[task][r["k"]], dir=v[1]))
        for x in fieldbad:
            issue("field_mismatch", task=task, **x)
        # error rows nothing replaces anywhere in the chain
        errs = []
        for lvl, ds in enumerate(levels):
            for d in ds:
                for f, i, r in rows_of[d]:
                    if r is not None and "error" in r:
                        repl = [x for x in hist.get((r["tag"], r["t"]), []) if x[0] > lvl or (x[1] == d)]
                        if not repl:
                            errs.append(dict(dir=d, tag=r["tag"], t=r["t"], error=r["error"][:100]))
        if errs:
            issue("error_rows_not_replaced", task=task, rows=errs)
        sup = collections.Counter()
        for k, v in hist.items():
            for x in v[:-1]:
                sup[x[1]] += 1
        cur_from = collections.Counter(v[1] for v in cur.values())
        out["tasks"][task] = dict(jobs=len(jobs), current=len(cur), missing=[f"{a} t{b}" for a, b in missing],
                                  extra=[f"{a} t{b}" for a, b in extra], current_from=dict(cur_from),
                                  superseded_rows=dict(sup), fix_lists=fixck,
                                  current_hidden=dict(collections.Counter(v[4].get("hidden") for v in cur.values())),
                                  points_with_n_rows=dict(collections.Counter(len(v) for v in hist.values())))
        if missing:
            issue("job_without_current_row", task=task, points=out["tasks"][task]["missing"])
        if extra:
            issue("current_row_not_in_jobs", task=task, points=out["tasks"][task]["extra"])

    # ---- dense / confirm / dense2 vs their jobs files
    for d, (role, task, jf, _) in OTHER.items():
        if role in ("bug1", "bug3") or jf is None:
            continue
        jobs = {(j["tag"], j["t"]): j for j in json.load(open(f"{DATA}/{jf}"))}
        good = collections.defaultdict(list)
        for f, i, r in rows_of[d]:
            if r is not None and "error" not in r:
                good[(r["tag"], r["t"])].append(r)
        miss = [k for k in jobs if k not in good]
        extra = [k for k in good if k not in jobs]
        multi = [k for k, v in good.items() if len(v) > 1]
        fb = []
        for k, v in good.items():
            j = jobs.get(k)
            if j:
                for f_ in ("k", "L", "N", "M", "episode_success", "task"):
                    if v[-1].get(f_) != j[f_]:
                        fb.append(dict(tag=k[0], t=k[1], field=f_, row=v[-1].get(f_), job=j[f_]))
            m = mr.get(k[0])
            if m:
                kk = sum(tk <= k[1] for tk, _ in m["history"])
                if kk != v[-1].get("k"):
                    fb.append(dict(tag=k[0], t=k[1], field="k_vs_main_history", row=v[-1].get("k"), job=kk))
        out["dirs"][d].update(jobs=len(jobs), jobs_missing=[f"{a} t{b}" for a, b in miss],
                              rows_not_in_jobs=[f"{a} t{b}" for a, b in extra], multi=[f"{a} t{b}" for a, b in multi],
                              field_mismatch=fb)
        if miss and role != "dense2":
            issue("job_without_row", dir=d, points=out["dirs"][d]["jobs_missing"])
        if extra:
            issue("row_not_in_jobs", dir=d, points=out["dirs"][d]["rows_not_in_jobs"])
        for x in fb:
            issue("field_mismatch", dir=d, **x)
    json.dump(out, open(os.path.join(HERE, "coverage.json"), "w"), indent=1)
    for t, v in out["tasks"].items():
        print(t, {k: v[k] for k in ("jobs", "current", "current_from", "superseded_rows", "current_hidden", "points_with_n_rows")},
              "missing", len(v["missing"]), "extra", len(v["extra"]))
    for d, v in out["dirs"].items():
        print(f"{d:38s} rows {v['rows']:4d} good {v['good']:4d} err {v['error']:2d} unparse {len(v['unparseable'])} dups {len(v['dup_rows'])} "
              f"roll {v['roll_files']:5d} snap {v['snap_files']:4d} foreign {v['n_foreign']} orphan_pts {len(v['orphan_roll_points'])} "
              f"orphan_snap {len(v['orphan_snap'])} outsideNxM {len(v['outside_NxM'])} missing {len(v['missing_files'])} "
              f"err_not_repl_in_dir {len(v['error_rows_not_replaced_in_dir'])} claims {v['claims']} claims_wo_row {len(v['claims_without_row'])} hidden {v['hidden']}")
    print("ISSUES")
    for x in out["issues"]:
        print(json.dumps(x)[:600])


if __name__ == "__main__":
    main()
