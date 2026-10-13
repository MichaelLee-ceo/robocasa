"""Official atomic-task eval (run_eval client protocol, split=target, official _check_success) for a list of tasks,
saving per-episode successes. Horizon: original (pre-1.5x) value from git 29f7ce8^, as in the composite repro."""
import sys, os, json
sys.path.insert(0, os.path.expanduser("~/Desktop/robocasa/third_party/Isaac-GR00T"))
from gr00t.eval.simulation import SimulationConfig, VideoConfig, MultiStepConfig, SimulationInferenceClient
tasks, n, port, out = sys.argv[1].split(","), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
H = json.load(open(os.path.expanduser("~/Desktop/robocasa/runs/atomic_horizons_original.json")))
client = SimulationInferenceClient(host="localhost", port=port)
for t in tasks:
    d = os.path.join(out, t); os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{d}/stats.json"):
        continue
    cfg = SimulationConfig(env_name=f"robocasa/{t}", split="target", n_episodes=n, n_envs=1,
                           video=VideoConfig(video_dir=d), multistep=MultiStepConfig(n_action_steps=16, max_episode_steps=H[t]))
    try:
        _, succ = client.run_simulation(cfg)
    except Exception as e:
        print("ERROR", t, repr(e), flush=True); continue
    json.dump(dict(task=t, horizon=H[t], n=len(succ), k=int(sum(succ)), episodes=[bool(x) for x in succ]), open(f"{d}/stats.json", "w"))
    print("DONE", t, int(sum(succ)), len(succ), flush=True)
