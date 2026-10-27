"""Data for the VLM critic (KettleBoiling, SteamInMicrowave, ...), in one pass: everything else (gripper paths, finger
glyphs, contacts, progress curves, failure events, overlays, GR00T features) is derived offline from what this saves.

Decision points (make_jobs.py): saved sim states of the seed-0 episodes of runs/20261025_seen480. What the sim state does
not hold (python-side fixture state such as microwave._turned_on, and the stage checks' memory S) is rebuilt by replaying
the episode's saved states up to the decision point (replay()); the replayed stage index is logged as k_replay. The
robot controllers' internal state is set to a canonical one (hidden_state.settle(), logged as hidden="settled") and
restored before every rollout, so all rollouts of a decision point start identically. Runs before 2026-10-27 lacked
this: each rollout inherited the controllers of the rollout before it (run/, run_SteamInMicrowave/; see README).
At each decision point, saved once in snap/<tag>_t<t>.npz:
  the sim state; RGB and depth (m) of the three robot cameras at 480x480 with intrinsics K and extrinsics T (camera ->
  world); the wrist depth with the robot hidden (environment only, as in ../20261026_vlm_critic/depth_overlay.py); the
  exact request the policy received (256x256 images, proprio, instruction), keys prefixed req/.
Candidates: N first chunks, one policy noise seed each. Each candidate is continued M times: its first chunk is executed,
then the plain policy runs with fresh noise (a different noise stream per continuation) until stage k's official
condition holds or L steps pass. Continuation m >= 1 replays the stored first chunk, so all continuations of a candidate
start with exactly the same actions. Seed of chunk ci: sha256(f"{tag}|critic|t{t}|c{i}|m{m if ci else 0}|{ci}").
Per rollout, roll/<tag>_t<t>_c<i>_m<m>.npz: every chunk received (all action keys, chunks x 16 x dim) and the flattened
sim state at the start and after every step; .mp4: agent-view-left + wrist video, every 4th step.
results_w<W>.jsonl, one row per decision point: labels per rollout (done = stage k finished, prev_ok = the earlier
stages still hold when it finished, steps, task success, progress in [0, 1] at the end, generic per stage kind, see
../20261026_imagine/imagine.py:progress).
Units: 1 step = 0.05 s; chunk = 16 steps. usage: python collect.py --jobs jobs.json --worker W --n_workers N --out DIR --port P"""
import argparse, copy, glob, hashlib, json, os, sys, time, traceback
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


def seed(tag, t, i, m, ci):
    return int.from_bytes(hashlib.sha256(f"{tag}|critic|t{t}|c{i}|m{m if ci else 0}|{ci}".encode()).digest()[:8], "little")


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


def decision_point(client, base, kenv, j, a):
    task, tag, t, k = j["task"], j["tag"], j["t"], j["k"]
    steps = TASKS[task]
    load_scene(kenv, j["scene"])
    instr = kenv.get_ep_meta()["lang"]
    ep_states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
    state = ep_states[t]
    S0, k_replay = replay(kenv, ep_states, t, steps)
    hidden_state.settle(kenv, ep_states[t - 1] if t else None)
    H0 = hidden_state.capture(kenv)
    ps, ref = py_state(kenv), progress_ref(task, k, kenv)
    snap = snapshot(kenv)
    name = f"{tag}_t{t}"
    cands = []
    for i in range(j["N"]):
        first, rolls = None, []
        for m in range(j["M"]):
            rec, states, S = Recorder(client, first), [], copy.deepcopy(S0)
            set_py_state(kenv, ps)  # rollout() then sets the sim state (and runs update_state)
            r, frames = rollout(rec, base, kenv, S, steps, state, j["L"], "task", instr, k0=k, k_stop=k + 1,
                                seed_fn=lambda ci, i=i, m=m: seed(tag, t, i, m, ci), step_states=states, hidden=H0)
            done = bool(r["reached"] > k or r["success"])
            prev_ok = all(steps[q]["check"](kenv, copy.deepcopy(S)) for q in range(k)) if done and k < len(steps) - 1 else True
            first = rec.chunks[0]
            if i == 0 and m == 0:
                snap.update({f"req/{kk}": np.asarray(v) for kk, v in rec.req0.items() if kk != "_seed"})
            np.savez_compressed(f"{a.out}/roll/{name}_c{i}_m{m}.npz", states=np.array(states),
                                **{kk: np.stack([c[kk] for c in rec.chunks]) for kk in rec.chunks[0]})
            write_video(f"{a.out}/roll/{name}_c{i}_m{m}.mp4", frames, fps=10)
            rolls.append(dict(m=m, done=done, prev_ok=prev_ok, steps=int(r["steps"]),
                              success=bool(r["success"]), reached=int(r["reached"]),
                              progress=round(progress(task, k, kenv, ref), 3),
                              obs_fp=fingerprint({kk: v for kk, v in rec.req0.items() if kk != "_seed"})))
        cands.append(dict(cand=i, rollouts=rolls))
    np.savez_compressed(f"{a.out}/snap/{name}.npz", state=state, **snap)
    return dict(instr=instr, py_state=ps, k_replay=k_replay, hidden="settled", cands=cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1); ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))[a.worker::a.n_workers]
    done = {(r["tag"], r["t"]) for p in glob.glob(f"{a.out}/results_w*.jsonl") for r in map(json.loads, open(p)) if "error" not in r}
    client = Client(port=a.port, timeout_ms=600000)
    os.makedirs(f"{a.out}/snap", exist_ok=True); os.makedirs(f"{a.out}/roll", exist_ok=True)
    base = None
    for j in jobs:
        if (j["tag"], j["t"]) in done:
            continue
        t0 = time.time()
        try:
            if base is None:
                base = gym.make(f"robocasa/{j['task']}", split="target", enable_render=True).unwrapped
            r = decision_point(client, base, base.env, j, a)
            r.update({kk: j[kk] for kk in ("tag", "task", "t", "k", "L", "N", "M", "episode_success")})
        except Exception as e:
            r = dict(tag=j["tag"], t=j["t"], error=repr(e), tb=traceback.format_exc()[-1500:])
        r["wall_s"] = round(time.time() - t0, 1)
        with open(f"{a.out}/results_w{a.worker}.jsonl", "a") as f:
            f.write(json.dumps(r, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
        print(json.dumps({kk: r.get(kk) for kk in ("tag", "t", "k", "error", "wall_s")}), flush=True)


if __name__ == "__main__":
    main()
