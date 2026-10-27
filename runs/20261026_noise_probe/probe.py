"""How much does the action noise matter at the moment a KettleBoiling episode goes wrong?

For each probe point (an episode of runs/20261025_seen480 and a step t0 just before its failure, or a matched point in
a successful episode), restore the saved sim state at t0 and run the frozen policy to the original horizon with N
different noise seeds (every chunk freshly seeded). Records the first chunk's actions (all candidates see the same
observation, checked by fingerprint), the stage completions, the kettle's highest lift and task success.
This is the "roll back and resample" upper bound, and its first chunk shows the action spread over noise.
usage: python probe.py --jobs jobs.json --worker W --n_workers N --out DIR --port P"""
import argparse, glob, hashlib, json, os, sys, time, traceback
import numpy as np
import gymnasium as gym

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from atomic_eval import load_scene, rollout  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
from groot_eval import Client, write_video  # noqa: E402
from oracle import set_state  # noqa: E402
import hidden_state  # noqa: E402

RUN = os.path.join(HERE, "..", "20261025_seen480", "run")


class FirstAction:
    """client wrapper that keeps the first response (the chunk sampled at t0)."""
    def __init__(self, client):
        self.client, self.first = client, None

    def get_action(self, req):
        r = self.client.get_action(req)
        if self.first is None:
            self.first = {k: np.asarray(v)[0].tolist() for k, v in r.items() if k.startswith("action")}
        return r


def kettle_lift(kenv, states, z0, every=4):
    """highest kettle height above its height at t0, and the first step (relative) it was >= 5 cm up."""
    bid = kenv.obj_body_id["obj"]
    best, first = 0.0, None
    for i in range(0, len(states), every):
        set_state(kenv, states[i])
        dz = float(kenv.sim.data.body_xpos[bid][2] - z0)
        best = max(best, dz)
        if first is None and dz >= 0.05:
            first = i
    return round(best, 3), first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1); ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))[a.worker::a.n_workers]
    done = {(r["tag"], r["t0"], r["cand"]) for p in glob.glob(f"{a.out}/results_w*.jsonl") for r in map(json.loads, open(p)) if "error" not in r}
    client = FirstAction(Client(port=a.port, timeout_ms=600000))
    base = gym.make("robocasa/KettleBoiling", split="target", enable_render=True).unwrapped
    kenv = base.env
    steps = TASKS["KettleBoiling"]
    os.makedirs(f"{a.out}/videos", exist_ok=True)
    for j in jobs:
        key = (j["tag"], j["t0"], j["cand"])
        if key in done:
            continue
        t_start = time.time()
        try:
            load_scene(kenv, j["scene"])
            states = np.load(f"{RUN}/KettleBoiling/{j['tag']}_steps.npz")["states"]
            start = states[j["t0"]]
            set_state(kenv, start)
            hidden_state.settle(kenv, states[j["t0"] - 1])  # controller state as at t0 of the saved run, approximately
            hidden = hidden_state.capture(kenv)
            z0 = float(kenv.sim.data.body_xpos[kenv.obj_body_id["obj"]][2])
            instr = kenv.get_ep_meta()["lang"]
            seed_fn = lambda ci, j=j: int.from_bytes(hashlib.sha256(f"{j['tag']}|probe|{j['t0']}|{j['cand']}|{ci}".encode()).digest()[:8], "little")
            client.first = None
            digest, st = [], []
            r, frames = rollout(client, base, kenv, {}, steps, start, j["horizon"] - j["t0"], "task", instr,
                                k0=j["k0"], seed_fn=seed_fn, digest=digest, step_states=st, hidden=hidden)
            lift, lift_at = kettle_lift(kenv, st, z0)
            out = dict(tag=j["tag"], t0=j["t0"], cand=j["cand"], role=j["role"], success=r["success"],
                       steps_after_t0=r["steps"], end_step=j["t0"] + r["steps"],
                       stage_steps=[[j["t0"] + t, k] for t, k in r["history"]], reached=r["reached"],
                       kettle_lift_m=lift, lift_step=(j["t0"] + lift_at) if lift_at is not None else None,
                       obs0_fp=digest[0][0], first_chunk=client.first, wall_s=round(time.time() - t_start, 1))
            write_video(f"{a.out}/videos/{j['tag']}_t{j['t0']}_c{j['cand']}.mp4", frames, fps=10)
        except Exception as e:
            out = dict(tag=j["tag"], t0=j["t0"], cand=j["cand"], error=repr(e), tb=traceback.format_exc()[-1500:])
        with open(f"{a.out}/results_w{a.worker}.jsonl", "a") as f:
            f.write(json.dumps(out) + "\n")
        print(json.dumps({k: out.get(k) for k in ("tag", "t0", "cand", "success", "kettle_lift_m", "error", "wall_s")}), flush=True)


if __name__ == "__main__":
    main()
