"""Data for the VLM critic (KettleBoiling, SteamInMicrowave, ...), in one pass: everything else (gripper paths, finger
glyphs, contacts, progress curves, failure events, overlays, GR00T features) is derived offline from what this saves.

Decision points (make_jobs.py): saved sim states of the seed-0 episodes of runs/20261025_seen480. What the sim state does
not hold (python-side fixture state such as microwave._turned_on, and the stage checks' memory S) is rebuilt by replaying
the episode's saved states up to the decision point (replay()); the replayed stage index is logged as k_replay. The
robot controllers' internal state is set to a canonical one (hidden_state.settle(), logged as hidden="settled", "settled-basefix" after the 2026-10-28 fix of the base-mode arm goal, "settled-v3" after the fix of the base controller start pose, "settled-v4" after the fix of the open-gripper threshold, "settled-v5" after the 2026-10-29 review: gripper ramp
from the servo target and contacts, observation clocks restored, and no settle at t = 0, where the state right after
load_scene is the main run's exactly; row["code"] holds the hashes of the restore code) and restored before every
rollout, so all rollouts of a decision point start identically. Runs before 2026-10-27 lacked this: each rollout
inherited the controllers of the rollout before it (the first collection, deleted; see README). row["restore"]: the
gripper ramp it starts from, the finger gap and the fingers' contacts at t.
At each decision point, saved once in snap/<tag>_t<t>.npz:
  the sim state; RGB and depth (m) of the three robot cameras at 480x480 with intrinsics K and extrinsics T (camera ->
  world); the wrist depth with the robot hidden (environment only, as in ../20261026_vlm_critic/depth_overlay.py); the
  exact request the policy received (256x256 images, proprio, instruction), keys prefixed req/.
Candidates: N first chunks, one policy noise seed each. Each candidate is continued M times: its first chunk is executed,
then the plain policy runs with fresh noise (a different noise stream per continuation) until stage k's official
condition holds or L steps pass. Continuation m >= 1 replays the stored first chunk, so all continuations of a candidate
start with exactly the same actions. Seed of chunk ci: sha256(f"{tag}|critic|t{t}|c{i}|m{m if ci else 0}|{ci}"), with
the job's "salt" appended for ci >= 1 (a confirmation run must not replay the continuations that located its point).
Checked live: every rollout starts at the main run's state at t, and the continuations of a candidate share their first
chunk's states exactly (a leak of hidden or fixture state between rollouts breaks either).
Per rollout, roll/<tag>_t<t>_c<i>_m<m>.npz: every chunk received (all action keys, chunks x 16 x dim) and the flattened
sim state at the start and after every step; .mp4: agent-view-left + wrist video, every 4th step.
results_w<W>.jsonl, one row per decision point: labels per rollout (done = stage k finished, prev_ok = the earlier
stages still hold when it finished, steps, task success, progress in [0, 1] at the end, generic per stage kind, see
../20261026_imagine/imagine.py:progress), and per chunk boundary [t, bitmask of the stages whose check holds (on a copy
of the stage memory), steps of the chunk with task success, success at the boundary step], so other definitions of
"finished" can be computed without re-collecting.
With --claim, workers share one queue instead of a fixed slice: each takes the last job that is neither finished nor
claimed, and claims it by creating claims/<tag>_t<t> (O_EXCL; holds host:pid), so runs on several GPUs can share an out
dir. pool.sh clears claims left by dead processes before it starts workers.
A job's "out" field, if present, overrides --out for that job (one queue filling several dirs; the env is rebuilt when
the task changes).
Units: 1 step = 0.05 s; chunk = 16 steps. usage: python collect.py --jobs jobs.json --worker W --n_workers N --out DIR --port P [--claim]"""
import argparse, copy, glob, hashlib, json, os, socket, sys, time, traceback
import numpy as np
import gymnasium as gym

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "20261026_imagine"))
from atomic_eval import load_scene, rollout, fingerprint  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
from groot_eval import Client, write_video  # noqa: E402
from oracle import set_state  # noqa: E402
from imagine import progress, progress_ref, py_state, set_py_state  # noqa: E402
import hidden_state  # noqa: E402
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map  # noqa: E402

RUN = os.path.join(HERE, "..", "20261025_seen480", "run")
CAMS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
RES = 480  # offscreen buffer is 640x480


class Recorder:
    """client wrapper: keeps the first request and every chunk received; with `first`, answers the first request with
    that chunk instead of asking the server."""
    def __init__(self, client, first=None):
        self.client, self.first, self.req0, self.chunks = client, first, None, []

    def get_action(self, req):
        if self.req0 is None:
            self.req0 = req
        if self.first is not None and not self.chunks:
            resp = {k: np.asarray(v)[None] for k, v in self.first.items()}
            resp["_seed_echo"] = req.get("_seed")
        else:
            resp = self.client.get_action(req)
        self.chunks.append({k: np.asarray(v)[0] for k, v in resp.items() if k.startswith("action")})
        return resp


def snapshot(kenv):
    out = {}
    for c in CAMS:
        rgb, dep = kenv.sim.render(camera_name=c, width=RES, height=RES, depth=True)
        out[f"rgb/{c}"] = rgb[::-1]
        out[f"depth/{c}"] = get_real_depth_map(kenv.sim, dep[::-1]).astype(np.float32)
        out[f"K/{c}"] = get_camera_intrinsic_matrix(kenv.sim, c, RES, RES)
        out[f"T/{c}"] = get_camera_extrinsic_matrix(kenv.sim, c)
    # robot geoms and site markers made invisible, rendered with the env's own context (a second GL context per worker
    # does not fit on the 24 GB GPU next to the policy servers)
    m = kenv.sim.model._model
    robot = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(("robot0_", "gripper0_", "mobilebase0_", "mount0_"))]
    g_alpha, s_alpha = m.geom_rgba[robot, 3].copy(), m.site_rgba[:, 3].copy()
    m.geom_rgba[robot, 3] = 0; m.site_rgba[:, 3] = 0
    _, dep = kenv.sim.render(camera_name="robot0_eye_in_hand", width=RES, height=RES, depth=True)
    m.geom_rgba[robot, 3] = g_alpha; m.site_rgba[:, 3] = s_alpha
    out["depth_env/robot0_eye_in_hand"] = get_real_depth_map(kenv.sim, dep[::-1]).astype(np.float32)
    return out


def seed(tag, t, i, m, ci, salt=""):
    key = f"{tag}|critic|t{t}|c{i}|m{m if ci else 0}|{ci}" + (salt if ci else "")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def code_hashes():
    import atomic_eval, imagine
    return {os.path.basename(f): hashlib.sha1(open(f, "rb").read()).hexdigest()[:10]
            for f in (__file__, hidden_state.__file__, atomic_eval.__file__, imagine.__file__)}


def restore_info(kenv, H0, prev_state):
    r = kenv.robots[0]
    f1, f2, n_open, n_close, fixture = hidden_state.finger_contacts(kenv, r, "right")
    q = kenv.sim.data.qpos[r._ref_gripper_joint_pos_indexes["right"]]
    out = dict(ramp=round(float(H0["robots"][0]["grippers"]["right"]["current_action"][1]), 3),
               gap_mm=round(1000 * float(abs(q[0] - q[1])), 2), held=bool(f1 & f2), obj=bool(f1 | f2), fixture=bool(fixture))
    if prev_state is not None:
        qp = prev_state[1 + np.asarray(r._ref_gripper_joint_pos_indexes["right"])]
        out["dgap_mm"] = round(1000 * float(abs(q[0] - q[1]) - abs(qp[0] - qp[1])), 3)
    return out


def replay(kenv, states, t, steps):
    """python fixture state and stage-check memory at step t of a saved episode: replay its saved states from the start
    (set_state runs every fixture's update_state, as each env step does, so e.g. microwave._turned_on follows the button
    presses), checking the stages at chunk boundaries as rollout() does. Leaves the env at step t; returns (S, k)."""
    S, k = {}, 0
    for i in range(t + 1):
        set_state(kenv, states[i])
        if i and i % 16 == 0:
            while k < len(steps) and steps[k]["check"](kenv, S):
                k += 1
    return S, k


def decision_point(client, base, kenv, j, out):
    task, tag, t, k = j["task"], j["tag"], j["t"], j["k"]
    steps = TASKS[task]
    load_scene(kenv, j["scene"])
    instr = kenv.get_ep_meta()["lang"]
    ep_states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
    state = ep_states[t]
    S0, k_replay = replay(kenv, ep_states, t, steps)
    if t:  # at t = 0 the hidden state after load_scene is the main run's (the gripper ramp is 0)
        hidden_state.settle(kenv, ep_states[t - 1])
    H0 = hidden_state.capture(kenv)
    rinfo = restore_info(kenv, H0, ep_states[t - 1] if t else None)
    ps, ref = py_state(kenv), progress_ref(task, k, kenv)
    snap = snapshot(kenv)
    name = f"{tag}_t{t}"
    cands = []
    for i in range(j["N"]):
        first, rolls = None, []
        for m in range(j["M"]):
            rec, states, S, bounds = Recorder(client, first), [], copy.deepcopy(S0), []

            def at_boundary(tb, e, S_, rew):
                mask = sum(1 << q for q in range(len(steps)) if steps[q]["check"](e, copy.deepcopy(S_)))
                bounds.append([int(tb), int(mask), int(np.sum(rew > 0)), bool(rew[-1] > 0)])
            set_py_state(kenv, ps)  # rollout() then sets the sim state (and runs update_state)
            r, frames = rollout(rec, base, kenv, S, steps, state, j["L"], "task", instr, k0=k, k_stop=k + 1,
                                seed_fn=lambda ci, i=i, m=m: seed(tag, t, i, m, ci, j.get("salt", "")), step_states=states,
                                hidden=H0, boundary_hook=at_boundary)
            if not np.array_equal(states[0], state):
                raise RuntimeError(f"c{i} m{m}: start state differs from the main run's at t (max {np.abs(states[0] - state).max():.3g})")
            if m == 0:
                first_states = np.array(states[:17])
            elif not np.array_equal(np.array(states[:17]), first_states):
                raise RuntimeError(f"c{i} m{m}: the replayed first chunk diverged from m0 (max {np.abs(np.array(states[:17]) - first_states).max():.3g})")
            done = bool(r["reached"] > k or r["success"])
            prev_ok = all(steps[q]["check"](kenv, copy.deepcopy(S)) for q in range(k)) if done and k < len(steps) - 1 else True
            first = rec.chunks[0]
            if i == 0 and m == 0:
                snap.update({f"req/{kk}": np.asarray(v) for kk, v in rec.req0.items() if kk != "_seed"})
            np.savez_compressed(f"{out}/roll/{name}_c{i}_m{m}.npz", states=np.array(states),
                                **{kk: np.stack([c[kk] for c in rec.chunks]) for kk in rec.chunks[0]})
            write_video(f"{out}/roll/{name}_c{i}_m{m}.mp4", frames, fps=10)
            rolls.append(dict(m=m, done=done, prev_ok=prev_ok, steps=int(r["steps"]),
                              success=bool(r["success"]), reached=int(r["reached"]),
                              progress=round(progress(task, k, kenv, ref), 3),
                              obs_fp=fingerprint({kk: v for kk, v in rec.req0.items() if kk != "_seed"}),
                              obs_fp_keys={kk: fingerprint(v) for kk, v in rec.req0.items() if kk != "_seed"}, bounds=bounds))
        cands.append(dict(cand=i, rollouts=rolls))
    np.savez_compressed(f"{out}/snap/{name}.npz", state=state, **snap)
    return dict(instr=instr, py_state=ps, k_replay=k_replay, hidden="settled-v5", code=CODE, restore=rinfo,
                salt=j.get("salt", ""), cands=cands)


def finished(out):
    rows = []
    for p in glob.glob(f"{out}/results*.jsonl"):  # results.jsonl (data/ layout) and the workers' results_w<W>.jsonl
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line another worker is still writing
                pass
    return {(r["tag"], r["t"]) for r in rows if "error" not in r}


def claim(out, j):
    os.makedirs(f"{out}/claims", exist_ok=True)
    try:
        fd = os.open(f"{out}/claims/{j['tag']}_t{j['t']}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{socket.gethostname()}:{os.getpid()}".encode()); os.close(fd)
    return True


CODE = code_hashes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1); ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--claim", action="store_true")
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    jobs = jobs[::-1] if a.claim else jobs[a.worker::a.n_workers]
    done = {}  # out dir -> points already finished there (a job's "out" overrides --out, so one queue can fill several dirs)
    client = Client(port=a.port, timeout_ms=600000)
    base, task = None, None
    for j in jobs:
        out = j.get("out", a.out)
        if out not in done:
            done[out] = finished(out)
            os.makedirs(f"{out}/snap", exist_ok=True); os.makedirs(f"{out}/roll", exist_ok=True)
        if (j["tag"], j["t"]) in done[out]:
            continue
        if a.claim and ((j["tag"], j["t"]) in finished(out) or not claim(out, j)):
            continue
        t0 = time.time()
        try:
            if base is None or j["task"] != task:  # one env per task; a mixed queue rebuilds it when the task changes
                if base is not None:
                    base.close()
                base, task = gym.make(f"robocasa/{j['task']}", split="target", enable_render=True).unwrapped, j["task"]
            r = decision_point(client, base, base.env, j, out)
            r.update({kk: j[kk] for kk in ("tag", "task", "t", "k", "L", "N", "M", "episode_success", "replaces") if kk in j})
        except Exception as e:
            r = dict(tag=j["tag"], t=j["t"], error=repr(e), tb=traceback.format_exc()[-1500:])
        r["wall_s"] = round(time.time() - t0, 1)
        with open(f"{out}/results_w{a.worker}.jsonl", "a") as f:
            f.write(json.dumps(r, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
        print(json.dumps({kk: r.get(kk) for kk in ("tag", "t", "k", "error", "wall_s")}), flush=True)


if __name__ == "__main__":
    main()
