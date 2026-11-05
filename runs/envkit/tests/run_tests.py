"""The environment test suite (2026-11-04): what must hold, for every task and frozen scene, before an experiment.

freeze <task> <seed>
  An official reset in a fresh process, frozen to ../scenes/<task>_s<seed>.pkl.gz (envkit.freeze). Then 8 chunks
  of seeded random actions are run from it; after each chunk the state hash (without camera images), the proprio
  observation hashes and the image hashes are the scene's reference trajectory (<task>_s<seed>.ref.json).
check <task> <seed>
  In a fresh process:
  reset   (reported, not failed) whether RoboCasa's official reset, seeded as envkit seeds it, reproduces the frozen
          state by itself (reset_reproducible, and the first differing keys). Two metadata keys are left out: the
          path of a temporary object xml (MujocoXML.file) and the order of an object's group-name list
          (groups_containing_sampled_obj, used only for membership tests); reset_meta_diff counts them.
  load    the loaded scene's state equals the frozen one (MjData arrays and contacts, and the whole Python graph:
          task attributes, fixtures, controllers, observables, wrappers, RNG states);
  replay  the reference chunks give the reference trajectory, chunk by chunk (state and proprio observations);
  restore at chunk 2: a snapshot, 3 chunks, a restore: the state equals the snapshot (round trip); the same 3 chunks
          again equal the first run (continuation); after another branch and a restore, again equal (no leak);
  idem    a snapshot and restore onto the identical state change nothing later (the run equals the reference).
stages <task> <seed>
  The stage tracker (../stages.py) on the official reset and after 3 random chunks: it runs, it is pure (the
  full state digest, images included, is unchanged by a call), and what it reports at t = 0.
policy <task> <seed>   (a seeded GR00T server: ../servers.sh; port from POLICY_PORT, default 5581)
  GR00T on the frozen scene (envkit.Episode, the path experiments use) to the horizon, the stage tracker after every
  chunk. Then:
  agree   at every chunk end, the official success implies every stage done, and a later stage done implies the
          earlier ones (the checks are self-contained);
  replay  the recorded actions in a new episode reproduce every chunk: success, tracker output, proprio observations,
          and the final state (the whole state without images; the differing keys are reported);
  restore at the first chunk of every new stage and the middle: a snapshot, the next 8 recorded chunks, a restore
          (round trip), the same chunks again (equal: state, tracker, success), another branch, a restore, the same
          chunks again (equal).
  The record (actions, tracker, success per chunk) goes to results/policy/<task>_s<seed>.pkl.gz.
sampling <task> <seed>   (two seeded GR00T servers: ports from POLICY_PORTS, default "5581 5582")
  Policy sampling on the frozen scene after 2 random chunks: the same observation and seed give the identical chunk
  (twice on one server, and on the other server); another seed gives a different chunk; after a step and a restore,
  the same seed gives the identical chunk again (the observation is part of the snapshot).
State comparisons leave out camera images: the official rendering (MSAA 4) can differ by +-1 on a few pixels across
processes. Such image differences are reported (render_noise: chunks, cameras, largest difference of the 32x32 grey
thumbnails, about 0.01 for MSAA noise). A thumbnail difference above IMAGE_TOL fails (test "images"): a real visual
difference, e.g. a stale or garbage image from a freed render context. The run calls gc.collect() after every chunk,
so a garbage collection mid-episode is covered.
Each command prints one json line (ok, failures, render noise, timing).
usage: python -I run_tests.py freeze|check|stages|policy|sampling <task> <seed>   (GPU rendering: set CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID)"""
import gc, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # runs/envkit
import numpy as np  # noqa: E402
import envkit  # noqa: E402
import stages  # noqa: E402

N_CHUNKS = 8
IMAGE_TOL = 0.5  # largest allowed 32x32 thumbnail difference (0-255 grey levels)


def chunks(ep, seed, n=N_CHUNKS):
    """seeded random actions in the policy's action space (small, so the robot stays near its start)"""
    rng = np.random.default_rng([1234, seed]); sp = ep.env.action_space
    return [{k: rng.uniform(-0.3, 0.3, size=sp[k].shape).astype(sp[k].dtype) for k in sp.spaces} for _ in range(n)]


def full_hash(d):
    return hashlib.sha1(json.dumps(sorted(d.items())).encode()).hexdigest()[:16]


def diff(a, b, n=6):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))[:n]


def thumb(img):
    """32x32 grey mean-pool of an image (0-255), to tell render noise (+-1 on a few pixels) from a visual change"""
    g = np.asarray(img, float).reshape(-1, *np.shape(img)[-3:])[-1].mean(-1)
    h, w = g.shape
    return np.round(g[: h // 32 * 32, : w // 32 * 32].reshape(32, h // 32, 32, w // 32).mean((1, 3)), 2).tolist()


def point(ep, obs):
    """what is compared after a chunk: the state without images, the proprio observations, the images (hashes and
    thumbnails)"""
    o = envkit.obs_digest(obs)
    return dict(state=full_hash(envkit.state_digest(ep, images=False)),
                proprio={k: v for k, v in o.items() if not k.startswith("video")},
                images={k: v for k, v in o.items() if k.startswith("video")},
                thumbs={k: thumb(v) for k, v in obs.items() if k.startswith("video")})


def ref_path(task, seed):
    return envkit.scene_path(task, seed).replace(".pkl.gz", ".ref.json")


def cmd_freeze(task, seed):
    t0 = time.time()
    ep, rec = envkit.freeze(task, seed)
    traj = [point(ep, ep.obs)]
    for c in chunks(ep, seed):
        ep.step(c); traj.append(point(ep, ep.obs))
    json.dump(dict(task=task, seed=seed, traj=traj, steps=ep.t, machine=envkit.machine()), open(ref_path(task, seed), "w"))
    return dict(ok=True, seconds=round(time.time() - t0, 1))


def cmd_check(task, seed):
    t0, fails, noise = time.time(), [], []
    ref = json.load(open(ref_path(task, seed)))["traj"]
    off = envkit.Episode.official(task, seed)
    d_reset = envkit.state_digest(off, images=False)
    off.close(); del off
    ep = envkit.Episode(task, seed)
    d0 = envkit.state_digest(ep, images=False)
    frozen = {k: v for k, v in ep.scene_digest.items() if not envkit.is_image_key(k)}
    meta = lambda k: k.endswith(".file") or "groups_containing_sampled_obj" in k
    reset_all = [k for k in set(frozen) | set(d_reset) if frozen.get(k) != d_reset.get(k)]
    reset_diff = sorted(k for k in reset_all if not meta(k))[:6]
    if d0 != frozen:
        fails.append(dict(test="load", keys=diff(frozen, d0)))
    C = chunks(ep, seed)
    run = []
    for i, c in enumerate(C):
        if i == 2:
            S, D2 = ep.snapshot(), envkit.state_digest(ep, images=False)
        ep.step(c); gc.collect(); run.append(point(ep, ep.obs))
    for i, (r, x) in enumerate(zip(ref[1:], run)):
        if r["state"] != x["state"] or r["proprio"] != x["proprio"]:
            fails.append(dict(test="replay", chunk=i, state=r["state"] == x["state"], proprio=diff(r["proprio"], x["proprio"]))); break
        if r["images"] != x["images"]:
            cams = diff(r["images"], x["images"])
            worst = round(max(float(np.abs(np.array(r["thumbs"][c]) - np.array(x["thumbs"][c])).max()) for c in cams), 3)
            noise.append(dict(chunk=i, cameras=cams, max_thumb_diff=worst))
            if worst > IMAGE_TOL:
                fails.append(dict(test="images", chunk=i, cameras=cams, max_thumb_diff=worst))
    ep.restore(S)
    if envkit.state_digest(ep, images=False) != D2:
        fails.append(dict(test="restore round trip", keys=diff(D2, envkit.state_digest(ep, images=False))))
    again = []
    for c in C[2:5]:
        ep.step(c); again.append(point(ep, ep.obs))
    if [(a["state"], a["proprio"]) for a in again] != [(a["state"], a["proprio"]) for a in run[2:5]]:
        fails.append(dict(test="restore continuation"))
    ep.restore(S)
    for c in chunks(ep, seed + 1000, 2):  # another branch
        ep.step(c)
    ep.restore(S)
    leak = []
    for c in C[2:5]:
        ep.step(c); leak.append(point(ep, ep.obs))
    if [(a["state"], a["proprio"]) for a in leak] != [(a["state"], a["proprio"]) for a in run[2:5]]:
        fails.append(dict(test="restore after a branch"))
    ep.close()
    ep = envkit.Episode(task, seed)
    for i, c in enumerate(C):
        if i == 2:
            ep.restore(ep.snapshot())
        ep.step(c)
    if point(ep, ep.obs)["state"] != ref[-1]["state"]:
        fails.append(dict(test="restore idempotence"))
    ep.close()
    return dict(ok=not fails, fails=fails, reset_reproducible=not reset_diff, reset_diff=reset_diff,
                reset_meta_diff=len(reset_all) - len([k for k in reset_all if not meta(k)]), render_noise=noise,
                seconds=round(time.time() - t0, 1))


def cmd_stages(task, seed):
    t0, fails, out = time.time(), [], []
    ep = envkit.Episode.official(task, seed)
    for i, c in enumerate([None] + chunks(ep, seed, 3)):
        if c is not None:
            ep.step(c)
        before = envkit.state_digest(ep)
        tr = stages.track(ep); stages.track(ep)
        after = envkit.state_digest(ep)
        if before != after:
            fails.append(dict(test="pure", chunk=i, keys=diff(before, after)))
        out.append(dict(t=tr["t"], stage=tr["stage"], done=tr["done"], info=tr["info"]))
    instr = stages.instructions(ep)
    ep.close()
    return dict(ok=not fails, fails=fails, track=out, instructions=instr, seconds=round(time.time() - t0, 1))


def _seg(ep, C):
    """run recorded chunks; per chunk: (state hash, tracker output, success)"""
    out = []
    for c in C:
        if ep.done:
            break
        ep.step(c)
        out.append((full_hash(envkit.state_digest(ep, images=False)), json.dumps(stages.track(ep), sort_keys=True),
                    ep.chunks[-1]["success"]))
    return out


def cmd_policy(task, seed):
    import gzip, pickle
    import policy
    t0, fails = time.time(), []
    client = policy.Client(int(os.environ.get("POLICY_PORT", 5581)))
    ep = envkit.Episode(task, seed)
    tracks = [stages.track(ep)]
    policy.rollout(ep, client, tag=f"{task}_s{seed}", on_chunk=lambda e: tracks.append(stages.track(e)))
    rec = ep.record(); rec["tracks"] = tracks
    final = envkit.state_digest(ep, images=False); rec["final_state"] = full_hash(final)
    ep.close(); del ep
    live_s = round(time.time() - t0, 1)
    print(f"[policy] live run: {len(rec['chunks'])} chunks, success {rec['success']}, {live_s} s", file=sys.stderr, flush=True)
    # agree: the official success implies every stage; a stage implies the earlier ones
    for tr in tracks:
        d = tr["done"]
        if tr["official"] and not all(d):
            fails.append(dict(test="official implies stages", t=tr["t"], done=d)); break
    gaps = [dict(t=tr["t"], done=tr["done"]) for tr in tracks if any(tr["done"][k] and not tr["done"][k - 1] for k in range(1, len(tr["done"])))]
    # replay with restore A/B at stage changes
    C = [c["actions"] for c in rec["chunks"]]
    best, nodes = 0, {len(C) // 2}
    for i, tr in enumerate(tracks[1:]):
        if tr["stage"] > best:
            best = tr["stage"]; nodes.add(i + 1)
    nodes = sorted(n for n in nodes if n < len(C))
    ep = envkit.Episode(task, seed)
    rt = [stages.track(ep)]
    for i, c in enumerate(C):
        if i in nodes:
            print(f"[policy] restore A/B at chunk {i} ({round(time.time() - t0)} s)", file=sys.stderr, flush=True)
            S, D = ep.snapshot(), envkit.state_digest(ep, images=False)
            first = _seg(ep, C[i:i + 8]); ep.restore(S)
            if envkit.state_digest(ep, images=False) != D:
                fails.append(dict(test="restore round trip", chunk=i, keys=diff(D, envkit.state_digest(ep, images=False))))
            if _seg(ep, C[i:i + 8]) != first:
                fails.append(dict(test="restore continuation", chunk=i))
            ep.restore(S); _seg(ep, chunks(ep, seed + 7 * i, 3)); ep.restore(S)
            if _seg(ep, C[i:i + 8]) != first:
                fails.append(dict(test="restore after a branch", chunk=i))
            ep.restore(S)
        ep.step(c); rt.append(stages.track(ep))
        r, x = rec["chunks"][i], ep.chunks[-1]
        if r["success"] != x["success"] or r["n"] != x["n"] or \
                {k: v for k, v in r["obs"].items() if not k.startswith("video")} != {k: v for k, v in x["obs"].items() if not k.startswith("video")}:
            fails.append(dict(test="replay", chunk=i, success=(r["success"], x["success"]))); break
    if json.dumps(rt, sort_keys=True) != json.dumps(tracks, sort_keys=True):
        k = next(i for i, (a, b) in enumerate(zip(rt, tracks)) if a != b)
        fails.append(dict(test="replay tracker", chunk=k, live=tracks[k], replay=rt[k]))
    D = envkit.state_digest(ep, images=False)
    if D != final:
        fails.append(dict(test="replay final state", keys=diff(final, D)))
    ep.close()
    os.makedirs(os.path.join(HERE, "results", "policy"), exist_ok=True)
    with gzip.open(os.path.join(HERE, "results", "policy", f"{task}_s{seed}.pkl.gz"), "wb") as f:
        pickle.dump(rec, f)
    first_done = {}
    for tr in tracks:
        for k, d in enumerate(tr["done"]):
            if d and k not in first_done:
                first_done[k] = tr["t"]
    return dict(ok=not fails, fails=fails, success=rec["success"], success_step=rec["success_step"], steps=rec["steps"],
                chunks=len(C), max_stage=max(tr["stage"] for tr in tracks), final_stage=tracks[-1]["stage"],
                first_done=first_done, gaps=gaps[:5], n_gaps=len(gaps), nodes=nodes, info_end=tracks[-1]["info"],
                live_seconds=live_s, seconds=round(time.time() - t0, 1))


def cmd_sampling(task, seed):
    import policy
    t0, fails = time.time(), []
    ports = [int(p) for p in os.environ.get("POLICY_PORTS", "5581 5582").split()]
    A, B = policy.Client(ports[0]), policy.Client(ports[-1])
    ep = envkit.Episode(task, seed)
    for c in chunks(ep, seed, 2):
        ep.step(c)
    same = lambda x, y: all(np.array_equal(x[k], y[k]) for k in x)
    s = policy.chunk_seed(f"sampling/{task}", seed, len(ep.chunks))
    a, a2, b = policy.act(A, ep.obs, s), policy.act(A, ep.obs, s), policy.act(B, ep.obs, s)
    other = policy.act(A, ep.obs, s + 1)
    if not same(a, a2):
        fails.append(dict(test="same seed, same server"))
    if not same(a, b):
        fails.append(dict(test="same seed, other server"))
    if same(a, other):
        fails.append(dict(test="another seed gives another chunk"))
    S = ep.snapshot(); ep.step(a); ep.restore(S)
    if not same(a, policy.act(A, ep.obs, s)):
        fails.append(dict(test="same seed after a restore"))
    diff_other = max(float(np.abs(a[k] - other[k]).max()) for k in a)
    ep.close()
    return dict(ok=not fails, fails=fails, ports=ports, max_diff_other_seed=round(diff_other, 4),
                seconds=round(time.time() - t0, 1))


def main():
    cmd, task, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    try:
        out = dict(freeze=cmd_freeze, check=cmd_check, stages=cmd_stages, policy=cmd_policy,
                   sampling=cmd_sampling)[cmd](task, seed)
    except Exception as e:  # report, do not hide
        import traceback
        out = dict(ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
    print(json.dumps(dict(cmd=cmd, task=task, seed=seed, **out)), flush=True)


if __name__ == "__main__":
    main()
