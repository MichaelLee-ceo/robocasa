"""Watch an offline tracker evaluation (eval_tracker.py) until every run has a valid result, independent of any
interactive session (start it with setsid nohup). Arms = windows; each writes <out>/<window>/<run>.json.
Every `--interval` seconds:
  1. a result is invalid if its replay does not match the record, or if VLM calls failed for reasons other than the
     model's answer (the server unreachable, timeouts, HTTP errors) at any check: it is deleted (logged in
     <out>/errors.jsonl) and the run is started again; answers that do not parse or fit the schema are the model's
     and stay;
  2. a run with no valid result that is not running is (re)started, at most `--tries` times, while fewer than
     `--workers` evaluations render on the GPU (`--gpu`); runs are started longest first (PackIdenticalLunches), the
     arms interleaved;
  3. a run that has taken more than `--timeout` seconds is stopped (logged) and so started again;
  4. <out>/STATUS.txt says how far it is.
When every run is valid (or out of tries), score_tracker.py scores each arm into <out>/SUMMARY.md.

usage: setsid nohup python3 coordinator/scripts/watch_eval.py --out DIR --vlm SPEC [--windows uniform,recent]
         [--plan vlm] [--runs all] [--gpu 1] [--workers 16] [--tries 3] [--timeout 9000] [--interval 60] \\
         > DIR/watch.log 2>&1 < /dev/null &"""
import argparse, glob, json, os, signal, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.path.join(ROOT, "envs", "groot", "bin", "python")
RECORDS = os.path.join(ROOT, "runs", "20261105_recovery", "phase0", "records")
TRANSPORT = ("cannot reach", "server error", "connection", "timed out", "timeout", "http ", "status 5", "500", "502", "503", "504")
ORDER = {"PackIdenticalLunches": 0, "SteamInMicrowave": 1, "KettleBoiling": 2}


def all_runs():
    ids = [os.path.basename(f)[:-len(".pkl.gz")] for f in glob.glob(os.path.join(RECORDS, "*.pkl.gz"))]
    return sorted(ids, key=lambda r: (ORDER.get(r.rsplit("_s", 1)[0], 9), int(r.rsplit("_s", 1)[1])))


def running(out):
    """{(window, run): (pid, seconds running)} of the eval_tracker.py processes writing under out"""
    found, hz, up = {}, os.sysconf("SC_CLK_TCK"), float(open("/proc/uptime").read().split()[0])
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            args = [a.decode() for a in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if a]
            if "coordinator/scripts/eval_tracker.py" not in args or "--out" not in args:
                continue
            d = os.path.normpath(args[args.index("--out") + 1])
            if os.path.dirname(d) != os.path.normpath(out):
                continue
            started = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19]) / hz
            found[(os.path.basename(d), args[-1])] = (int(pid), up - started)
        except (OSError, ValueError, IndexError):
            continue
    return found


def problem(r):
    """why a result is invalid (None: valid)"""
    if not r.get("replay_matches_record"):
        return "replay does not match the record"
    for c in r.get("checks", []):
        for x in c.get("calls", []):
            e = (x.get("error") or "").lower()
            if e and not e.startswith(("unparseable", "answer does not match")) and any(t in e for t in TRANSPORT):
                return f"VLM call failed at chunk {c['chunk']}: {x['error'][:200]}"
    return None


def results(out, windows):
    ok = {}
    for w in windows:
        for f in glob.glob(os.path.join(out, w, "*.json")):
            if f.endswith("score.json"):
                continue
            try:
                r = json.load(open(f))
            except (OSError, ValueError):
                continue                              # being written
            why = problem(r)
            if why:
                with open(os.path.join(out, "errors.jsonl"), "a") as log:
                    log.write(json.dumps(dict(window=w, run=r.get("run"), error=why,
                                              time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
                os.remove(f)
                continue
            ok[(w, r["run"])] = r
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--windows", default="uniform")
    ap.add_argument("--plan", default="vlm")
    ap.add_argument("--runs", default="all", help="all, or a file with one run id per line")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=9000)
    ap.add_argument("--interval", type=int, default=60)
    a = ap.parse_args()
    out, windows = os.path.abspath(a.out), a.windows.split(",")
    runs = all_runs() if a.runs == "all" else [l.strip() for l in open(a.runs) if l.strip()]
    todo = [(w, r) for r in runs for w in windows]                   # arms interleaved, longest runs first
    tries_f = os.path.join(out, "tries.json")
    tries = {tuple(k.split("|")): v for k, v in json.load(open(tries_f)).items()} if os.path.exists(tries_f) else {}
    for w in windows:
        os.makedirs(os.path.join(out, w), exist_ok=True)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=a.gpu, MUJOCO_EGL_DEVICE_ID=a.gpu,
               OMP_NUM_THREADS="1")
    while True:
        ok, run_now = results(out, windows), running(out)
        for key, (pid, secs) in run_now.items():
            if secs > a.timeout:
                os.kill(pid, signal.SIGTERM)
                with open(os.path.join(out, "errors.jsonl"), "a") as log:
                    log.write(json.dumps(dict(window=key[0], run=key[1], error=f"timeout after {secs:.0f} s",
                                              time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
        missing = [k for k in todo if k not in ok and k not in run_now]
        startable = [k for k in missing if tries.get(k, 0) < a.tries]
        for k in startable[:max(0, a.workers - len(run_now))]:
            w, r = k
            log = open(os.path.join(out, w, f"{r}.log"), "a")
            subprocess.Popen([PY, "-I", "coordinator/scripts/eval_tracker.py", "--vlm", a.vlm, "--window", w,
                              "--plan", a.plan, "--out", os.path.join(out, w), r], cwd=ROOT, env=env, stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            tries[k] = tries.get(k, 0) + 1
        json.dump({"|".join(k): v for k, v in tries.items()}, open(tries_f, "w"))
        done = sum(k in ok for k in todo)
        failed = [k for k in missing if tries.get(k, 0) >= a.tries]
        with open(os.path.join(out, "STATUS.txt"), "w") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  valid {done}/{len(todo)}, running {len(run_now)}, "
                    f"out of tries {len(failed)}\n")
            for w in windows:
                f.write(f"  {w}: {sum(k in ok for k in todo if k[0] == w)}/{len(runs)}\n")
            for k in failed:
                f.write(f"  out of tries: {k}\n")
        if not run_now and not startable:
            break
        time.sleep(a.interval)
    lines = ["# Offline evaluation of the to-do list tracker", "", f"VLM {a.vlm}, plan {a.plan}, windows {windows}", ""]
    sc = subprocess.run(["python3", "coordinator/scripts/score_tracker.py"] + [os.path.join(out, w) for w in windows],
                        cwd=ROOT, capture_output=True, text=True)
    lines += ["```", sc.stdout.strip(), sc.stderr.strip()[-2000:], "```"]
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
