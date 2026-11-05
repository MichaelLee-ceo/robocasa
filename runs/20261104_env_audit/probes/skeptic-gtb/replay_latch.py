"""Replay seen480 GetToastedBread episodes: official toaster_on latch vs our _toast_done (tracker-style, chunk boundaries)."""
import glob, json, multiprocessing, os, sys
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts"); sys.path.insert(0, f"{R}/runs/20261025_seen480")
RUN = f"{R}/runs/20261025_seen480/run"

def work(r):
    from analyze_stages import replay_env
    import atomic_eval  # noqa
    from oracle import reset_to, set_state
    from paths import local
    from atomic_steps import TASKS
    import robocasa.utils.object_utils as OU
    task, tag = r["task"], r["tag"]
    z = np.load(local(r["scene"]), allow_pickle=True)
    xml, ep_meta = str(z["xml"]), json.loads(str(z["ep_meta"]))
    states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
    env = replay_env(task)
    reset_to(env, xml, ep_meta)
    chk = TASKS[task][0]["check"]
    S = {}; k = 0; t0_pass = None; latch_first = None
    contact_on = []  # frames where bread touches the slot pair that is on
    for f in range(len(states)):
        set_state(env, states[f])
        env._check_success()
        if env.toaster_on and latch_first is None:
            latch_first = f
        st = env.toaster.get_state(env)
        for i, sp in enumerate(st.keys()):
            if st[sp]["turned_on"] and env.toaster.check_slot_contact(env, "obj", slot_pair=i):
                contact_on.append(f); break
        if f > 0 and f % 16 == 0 and k == 0:  # tracker: stage 0 checked only at chunk boundaries until it passes
            if chk(env, S):
                k = 1; t0_pass = f
    return dict(tag=tag, n=len(states), toaster_on_end=bool(env.toaster_on), latch_first=latch_first,
                tracker_stage0=t0_pass, S=S, n_contact_on=len(contact_on),
                contact_on_span=[contact_on[0], contact_on[-1]] if contact_on else None)

if __name__ == "__main__":
    rows = [json.loads(l) for p in sorted(glob.glob(f"{RUN}/results_w*.jsonl")) for l in open(p)]
    rows = [r for r in rows if r.get("task") == "GetToastedBread" and "error" not in r]
    print(len(rows), "episodes", flush=True)
    with multiprocessing.get_context("spawn").Pool(int(sys.argv[1])) as pool:
        for res in pool.imap_unordered(work, rows):
            print(json.dumps(res, default=str), flush=True)
