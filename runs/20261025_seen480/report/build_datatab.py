"""Data for the report's "Data" tab (index.html#data): what the VLM-critic collection (runs/20261026_critic_data)
contains, shown on one real episode and one real decision point, plus every decision point of each task.

Writes report/datatab.js (window.DATATAB) and report/dt/:
- film.jpg: frames of the example episode (main run, seed 0) at the decision-point steps, from its video;
- snap.jpg: the example decision point's saved views (3 RGB cameras, wrist depth, wrist depth without the robot);
- grid.mp4: four rollouts from that decision point side by side (two candidates x two continuations).
usage (groot env, for numpy / PIL / av): python report/build_datatab.py; build_pub.py copies the output."""
import glob, json, math, os, subprocess
import numpy as np
import av
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.dirname(os.path.dirname(HERE))
DATA = f"{RUNS}/20261026_critic_data"
MAIN = f"{RUNS}/20261025_seen480/run"
EX_TAG, EX_T, EX_CANDS = "KettleBoiling_s0_single", 160, (0, 2)
FILM_STEPS = [0, 80, 160, 240, 320, 400, 416]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
os.makedirs(f"{HERE}/dt", exist_ok=True)
seed_of = lambda tag: int(tag.split("_s")[1].split("_")[0])


def task_block(task, run_dir):
    rows = [r for p in glob.glob(f"{DATA}/{run_dir}/results_w*.jsonl") for r in map(json.loads, open(p)) if "error" not in r]
    jobs = json.load(open(f"{DATA}/jobs.json" if task == "KettleBoiling" else f"{DATA}/jobs_{task}.json"))
    main = {r["tag"]: r for p in glob.glob(f"{MAIN}/results_w*.jsonl") for r in map(json.loads, open(p)) if r.get("task") == task}
    horizon = next(r["horizon"] for r in main.values())
    eps = {}
    for tag in sorted({j["tag"] for j in jobs}, key=seed_of):
        m = main[tag]
        eps[tag] = dict(seed=seed_of(tag), steps=m["steps"], ok=m["success"], hist=m["history"], points=[])
    Y, steps_all = [], []
    for r in sorted(rows, key=lambda r: (seed_of(r["tag"]), r["t"])):
        g = [[int(x["done"]), x["steps"], x["progress"]] for c in r["cands"] for x in c["rollouts"]]
        eps[r["tag"]]["points"].append(dict(t=r["t"], k=r["k"], L=r["L"], g=g))
        Y.append(np.array([[x["done"] for x in c["rollouts"]] for c in r["cands"]], float))
        steps_all += [x["steps"] for c in r["cands"] for x in c["rollouts"]]
    mix = [("all" if y.min() == 1 else "none" if y.max() == 0 else "mixed") for y in Y]
    same = float(np.mean([y[i, 0] == y[i, 1] for y in Y for i in range(len(y))])) if Y else None
    cross = float(np.mean([y[i, a] == y[j, b] for y in Y for i in range(len(y)) for j in range(len(y)) if i != j
                           for a in range(2) for b in range(2)])) if Y else None
    size = int(subprocess.run(["du", "-sb", f"{DATA}/{run_dir}"], capture_output=True, text=True).stdout.split()[0])
    return dict(task=task, horizon=horizon, n_jobs=len(jobs), n_points=len(rows), n_rollouts=len(steps_all),
                sim_steps=int(sum(steps_all)), chunk_samples=int(sum(math.ceil(s / 16) for s in steps_all)),
                per_stage=[sum(j["k"] == k for j in jobs) for k in range(max(j["k"] for j in jobs) + 1)],
                mix=dict(all=mix.count("all"), none=mix.count("none"), mixed=mix.count("mixed")),
                p_done=float(np.mean([y.mean() for y in Y])) if Y else None, same=same, cross=cross,
                bytes=size, every=jobs[1]["t"] - jobs[0]["t"], L=sorted({(j["k"], j["L"]) for j in jobs}),
                episodes=list(eps.values()))


def frames_of(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


# --- example episode: film strip at the decision-point steps (video: 1 frame per 4 steps, frame i = after step 4(i+1))
fr = frames_of(f"{MAIN}/KettleBoiling/{EX_TAG}.mp4")
tiles = [Image.fromarray(fr[min(max(s // 4 - 1, 0), len(fr) - 1)]).resize((384, 192), Image.LANCZOS) for s in FILM_STEPS]
film = Image.new("RGB", (384 * len(tiles), 192))
for i, t in enumerate(tiles):
    film.paste(t, (384 * i, 0))
film.save(f"{HERE}/dt/film.jpg", quality=80, optimize=True)

# --- example decision point: saved views
S = np.load(f"{DATA}/run/snap/{EX_TAG}_t{EX_T}.npz")
def depth_img(d, lo, hi):
    v = np.clip((d - lo) / (hi - lo), 0, 1)[..., None]
    near, far = np.array([250, 110, 40]), np.array([40, 70, 160])  # near = orange, far = blue
    return (near * (1 - v) + far * v).astype(np.uint8)
w = "robot0_eye_in_hand"
lo, hi = float(np.percentile(S[f"depth_env/{w}"], 1)), float(np.percentile(S[f"depth_env/{w}"], 99))
views = [S["rgb/robot0_agentview_left"], S["rgb/robot0_agentview_right"], S[f"rgb/{w}"], depth_img(S[f"depth/{w}"], lo, hi),
         depth_img(S[f"depth_env/{w}"], lo, hi)]
snap = Image.new("RGB", (320 * len(views), 320))
for i, v in enumerate(views):
    snap.paste(Image.fromarray(v).resize((320, 320), Image.LANCZOS), (320 * i, 0))
snap.save(f"{HERE}/dt/snap.jpg", quality=82, optimize=True)
snap_keys = [[k, list(S[k].shape), str(S[k].dtype)] for k in S.files]

# --- example rollouts side by side, padded to the longest
rows = [r for p in glob.glob(f"{DATA}/run/results_w*.jsonl") for r in map(json.loads, open(p)) if r.get("tag") == EX_TAG and r.get("t") == EX_T]
ex = rows[0]
paths, labels = [], []
for c in EX_CANDS:
    for m in range(2):
        x = ex["cands"][c]["rollouts"][m]
        paths.append(f"{DATA}/run/roll/{EX_TAG}_t{EX_T}_c{c}_m{m}.mp4")
        labels.append(f"candidate {c} - continuation {m} - {'finished S0 in ' + str(x['steps']) + ' steps' if x['done'] else 'did not finish (' + str(x['steps']) + ' steps)'}")
n = max(len(frames_of(p)) for p in paths)
inp = sum([["-i", p] for p in paths], [])
fl = ";".join(f"[{i}:v]tpad=stop_mode=clone:stop={n},trim=end_frame={n},drawtext=fontfile={FONT}:text='{labels[i]}':x=8:y=8:fontsize=15:"
              f"fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=4[v{i}]" for i in range(4))
fl += ";[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]"
subprocess.run(["ffmpeg", "-y", "-v", "error", *inp, "-filter_complex", fl, "-map", "[out]", "-c:v", "libx264", "-crf", "30",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", f"{HERE}/dt/grid.mp4"], check=True)

R = np.load(f"{DATA}/run/roll/{EX_TAG}_t{EX_T}_c{EX_CANDS[0]}_m0.npz")
roll_keys = [[k, list(R[k].shape), str(R[k].dtype)] for k in R.files]
chunk = {k: np.round(R[k][0], 3).tolist() for k in R.files if k.startswith("action.")}
row_example = {k: v for k, v in ex.items() if k not in ("py_state", "cands")}
row_example["cands"] = ex["cands"][:1]

D = dict(example=dict(tag=EX_TAG, seed=seed_of(EX_TAG), t=EX_T, cands=list(EX_CANDS), film_steps=FILM_STEPS,
                      grid=[[x["done"], x["steps"], x["progress"]] for c in ex["cands"] for x in c["rollouts"]],
                      instr=ex["instr"], chunk=chunk, snap_keys=snap_keys, roll_keys=roll_keys, row=row_example),
         tasks=[task_block("KettleBoiling", "run"), task_block("SteamInMicrowave", "run_SteamInMicrowave")])
with open(f"{HERE}/datatab.js", "w") as f:
    f.write("window.DATATAB = ")
    json.dump(D, f, separators=(",", ":"))
    f.write(";\n")
for t in D["tasks"]:
    print(t["task"], t["n_points"], "/", t["n_jobs"], "points,", t["n_rollouts"], "rollouts,", t["sim_steps"], "steps,",
          t["chunk_samples"], "chunk samples,", t["mix"], f"{t['bytes'] / 1e9:.1f} GB")
print({p: os.path.getsize(f"{HERE}/dt/{p}") // 1024 for p in os.listdir(f"{HERE}/dt")}, "datatab.js", os.path.getsize(f"{HERE}/datatab.js") // 1024, "KB")
