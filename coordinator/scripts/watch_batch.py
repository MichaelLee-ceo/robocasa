"""Watch a batch of `run.py point` runs until every one has a valid result, independent of any interactive session
(start it with setsid nohup). Every `--interval` seconds:
  1. the per-run result files (<out>/<config>/<id>_p<seed>.json) are the record: a result with an "error" is deleted
     (and logged in <out>/<config>_errors.jsonl);
  2. a run with no valid result that is not running is (re)started, at most `--tries` times, on the render GPU with room
     (`--gpus 1:19,2:13`: GPU -> at most that many runs rendering on it, counting runs started by anyone);
  3. a run that has taken more than `--timeout` seconds is stopped (logged as an error) and so started again;
  4. <out>/<config>_STATUS.txt says how far it is.
When every run is valid (or out of tries), <out>/<config>.jsonl is rewritten from the result files (valid ones only)
and <out>/<config>_SUMMARY.md compares the runs with loop.py's retrace arm and control arm on the same points and seeds.

usage: setsid nohup python3 coordinator/scripts/watch_batch.py --out DIR --config oracle_retrace --runs LIST
         [--gpus 1:19,2:13] [--vlm SPEC] [--tries 3] [--timeout 2700] [--interval 60] > DIR/watch.log 2>&1 < /dev/null &
       LIST: one run per line, "<point id> <policy seed> <GR00T port>" """
import argparse, collections, glob, json, os, signal, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.path.join(ROOT, "envs", "groot", "bin", "python")


def running(out, config):
    """{(id, seed): (render GPU, pid, seconds running)} of the run.py processes writing to this batch (started by
    anyone)"""
    found, hz, up = {}, os.sysconf("SC_CLK_TCK"), float(open("/proc/uptime").read().split()[0])
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            args = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            args = [a.decode() for a in args if a]
            if "coordinator/scripts/run.py" not in args or "point" not in args or config not in args:
                continue
            if "--out" not in args or os.path.normpath(args[args.index("--out") + 1]) != os.path.normpath(out):
                continue
            i = args.index("point")
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
            gpu = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"CUDA_VISIBLE_DEVICES=")), "?")
            started = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19]) / hz
            found[(args[i + 1], int(args[i + 3]))] = (gpu, int(pid), up - started)
        except (OSError, ValueError, IndexError):
            continue
    return found


def results(out, config):
    """{(id, seed): result} of the valid results; deletes the results with an error (logged)"""
    ok, d = {}, os.path.join(out, config)
    for f in glob.glob(os.path.join(d, "*_p*.json")):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue                                  # being written
        if "error" in r:
            with open(os.path.join(out, f"{config}_errors.jsonl"), "a") as log:
                log.write(json.dumps(dict(file=os.path.basename(f), error=str(r["error"])[:300],
                                          time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
            os.remove(f)
            continue
        ok[(r.get("id"), int(r.get("pseed")))] = r
    return ok


def start(run_id, seed, port, gpu, out, config, vlm):
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu,
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", POLICY_PORT=str(port))
    cmd = f"{PY} -I coordinator/scripts/run.py point {run_id} {config} {seed} --out {out}" + (f" --vlm {vlm}" if vlm else "")
    cmd += f" 2>/dev/null | grep '^{{' >> {out}/{config}.jsonl"
    subprocess.Popen(["bash", "-c", cmd], cwd=ROOT, env=env, start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def summary(out, config, res):
    """the runs against loop.py's retrace and control arms, paired by point and seed"""
    from math import comb
    load = lambda arm: {(d["id"], d["pseed"]): d for d in (json.load(open(f)) for f in glob.glob(  # noqa: E731
        os.path.join(ROOT, "runs", "20261105_recovery", "loop", arm, "*.json")))}
    v2, ctl = load("retrace"), load("control")
    ks = sorted(k for k in res if k in v2)
    sign = lambda a, b: min(1, 2 * sum(comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b)) if a + b else 1  # noqa: E731
    L = [f"# {config}: {len(res)} runs (paired with loop.py's retrace arm: {len(ks)})", "",
         "| | control (GR00T alone) | v2 retrace (loop.py) | this batch | runs only this / only v2 (sign p) |", "|---|---|---|---|---|"]
    for key in ("fixed", "success"):
        a = sum(bool(v2[k][key]) for k in ks); b = sum(bool(res[k][key]) for k in ks)
        c = sum(bool(ctl[k][key]) for k in ks if k in ctl)
        up = sum(bool(res[k][key]) and not bool(v2[k][key]) for k in ks); dn = sum(bool(v2[k][key]) and not bool(res[k][key]) for k in ks)
        L.append(f"| {key} | {c}/{len(ks)} | {a}/{len(ks)} ({a / max(1, len(ks)):.0%}) | {b}/{len(ks)} ({b / max(1, len(ks)):.0%}) | {up} / {dn} ({sign(up, dn):.2f}) |")
    pts = collections.defaultdict(lambda: [0, 0, 0, 0])
    for k in ks:
        p = pts[k[0]]
        p[0] += bool(res[k]["fixed"]); p[1] += bool(v2[k]["fixed"]); p[2] += bool(res[k]["success"]); p[3] += bool(v2[k]["success"])
    for nm, i, j in (("fixed", 0, 1), ("success", 2, 3)):
        b = sum(p[i] > p[j] for p in pts.values()); w = sum(p[i] < p[j] for p in pts.values())
        L.append(f"\nPer failure point, {nm}: this batch better at {b}, worse at {w}, same at {len(pts) - b - w} (sign p {sign(b, w):.2f})")
    by = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for k in ks:
        t = by[res[k]["task"]]; t[0] += 1; t[1] += bool(v2[k]["fixed"]); t[2] += bool(res[k]["fixed"]); t[3] += bool(v2[k]["success"]); t[4] += bool(res[k]["success"])
    L += ["", "| task | runs | fixed v2 / this | success v2 / this |", "|---|---|---|---|"]
    L += [f"| {t} | {v[0]} | {v[1]} / {v[2]} | {v[3]} / {v[4]} |" for t, v in sorted(by.items())]
    rs = [iv["result"] for r in res.values() for iv in r["memory"]["interventions"] if iv["kind"] == "retrace"]
    us = [u for r in rs for u in r.get("unstick", [])]
    ch = sorted(r["chunks"] for r in rs)
    L += ["", f"Retraces: {len(rs)}; arrived {sum(r['arrived'] for r in rs)}; blocked at first {sum(1 for r in rs if r.get('unstick'))}, "
          f"arrived after unstick {sum(1 for r in rs if r.get('unstick') and r['arrived'])}; not arrived without a block "
          f"{sum(1 for r in rs if not r['arrived'] and not r.get('unstick'))}",
          f"Retrace chunks: median {ch[len(ch) // 2] if ch else None}, p90 {ch[int(.9 * (len(ch) - 1))] if ch else None}, max {ch[-1] if ch else None}; "
          f"at the 400-chunk cap: {sum(c >= 400 for c in ch)}",
          f"Unstick sentences: {collections.Counter(u['instruction'] for u in us).most_common(8)}",
          f"Run time: median {sorted(r['seconds'] for r in res.values())[len(res) // 2]:.0f} s, max {max(r['seconds'] for r in res.values()):.0f} s"]
    open(os.path.join(out, f"{config}_SUMMARY.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--gpus", default="1:19,2:13")
    ap.add_argument("--vlm", default=None)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=2700)
    a = ap.parse_args()
    os.chdir(ROOT)
    todo = [ln.split() for ln in open(a.runs) if ln.strip()]
    todo = [(r, int(s), int(p)) for r, s, p in todo]
    cap = {g: int(n) for g, n in (x.split(":") for x in a.gpus.split(","))}
    tries_file = os.path.join(a.out, f"{a.config}_tries.json")
    tries = json.load(open(tries_file)) if os.path.exists(tries_file) else {}
    while True:
        res, run = results(a.out, a.config), running(a.out, a.config)
        for (r, s), (gpu, pid, secs) in list(run.items()):
            if secs > a.timeout:                      # too long (e.g. a broken render context): stop it, start again
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
                with open(os.path.join(a.out, f"{a.config}_errors.jsonl"), "a") as log:
                    log.write(json.dumps(dict(file=f"{r}_p{s}.json", error=f"timeout: stopped after {secs:.0f} s",
                                              time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
                del run[(r, s)]
        load = collections.Counter(g for g, _, _ in run.values())
        pending = [(r, s, p) for r, s, p in todo if (r, s) not in res and (r, s) not in run and tries.get(f"{r}:{s}", 0) < a.tries]
        for r, s, p in pending:
            gpu = next((g for g in cap if load[g] < cap[g]), None)
            if gpu is None:
                break
            start(r, s, p, gpu, a.out, a.config, a.vlm)
            tries[f"{r}:{s}"] = tries.get(f"{r}:{s}", 0) + 1
            load[gpu] += 1
        json.dump(tries, open(tries_file, "w"))
        gave_up = [f"{r}:{s}" for r, s, p in todo if (r, s) not in res and (r, s) not in run and tries.get(f"{r}:{s}", 0) >= a.tries]
        errs = sum(1 for _ in open(os.path.join(a.out, f"{a.config}_errors.jsonl"))) if os.path.exists(
            os.path.join(a.out, f"{a.config}_errors.jsonl")) else 0
        open(os.path.join(a.out, f"{a.config}_STATUS.txt"), "w").write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}  valid {len([k for k in res if k in {(r, s) for r, s, _ in todo}])}/{len(todo)}"
            f"  running {len(run)} {dict(load)}  waiting {len(pending)}  errors deleted {errs}  gave up {gave_up}\n")
        if not run and not [1 for r, s, p in todo if (r, s) not in res and tries.get(f"{r}:{s}", 0) < a.tries]:
            break
        time.sleep(a.interval)
    res = results(a.out, a.config)
    keys = {(r, s) for r, s, _ in todo}
    with open(os.path.join(a.out, f"{a.config}.jsonl"), "w") as f:
        for k in sorted(k for k in res if k in keys):
            r = res[k]
            f.write(json.dumps({x: v for x, v in r.items() if x != "memory"}) + "\n")
    summary(a.out, a.config, {k: v for k, v in res.items() if k in keys})
    open(os.path.join(a.out, f"{a.config}_STATUS.txt"), "a").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  done\n")


if __name__ == "__main__":
    main()
