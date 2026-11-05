# envkit: the environment for every experiment

Use this folder for every new experiment on the 16 RoboCasa Composite-Seen tasks with the frozen GR00T N1.5 policy.
It runs episodes exactly as the official GR00T eval does. It also gives exact snapshots, restores and replays, and a
stage tracker that agrees with the official success check. It was built after an audit of the earlier pipeline
(`../20261104_env_audit`). It is verified on all 480 scenes and 48 full GR00T runs (`tests/README.md`).

| file | what |
|---|---|
| `envkit.py` | `Episode`: one official episode from a frozen scene; `step`, `snapshot` / `restore`, `record`, `replay` |
| `stages.py` | `track(ep)`: which stages of the task are done now; `instructions(ep)`: one sentence per stage |
| `policy.py` | the GR00T client: `Client(port)`, `act`, `chunk_seed`, `rollout` |
| `serve_seeded.py`, `servers.sh` | the GR00T server (official `run_eval.py` server, action noise seeded per request) and its launcher |
| `scenes/` | the frozen scenes: 16 tasks x seeds 0-29 (`scenes/README.md`) |
| `tests/` | the tests that must pass before experiments (`tests/README.md`) |

## What is exact and what is not
- **Exact, bit for bit:**
  - the scene;
  - the physics and every success / stage check, given the actions;
  - a replay of recorded actions, in any process;
  - a restore, and everything after it, given the same actions;
  - the policy's chunk for the same observation and seed.
- **Not exact: camera images.** The official rendering (MSAA 4) changes a few edge pixels by +-1 at random on any
  render, even in one process. A live policy run, also after a restore, can take another path. This acts like
  extra policy noise; success is judged on the physics, so the official rendering is kept.
- **Consequences:**
  - Compare methods by success rates over scenes.
  - To reproduce a specific episode, replay its recorded actions.
  - Never compare images bit for bit; compare states with `envkit.state_digest(ep, images=False)`.

## Setup
```bash
cd <repo>/runs/envkit
PY="../../envs/groot/bin/python -I"        # the GR00T python ($GROOT_PY overrides it); always run with -I
nvidia-smi                                  # pick a free GPU (this project uses GPU1)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 OMP_NUM_THREADS=1
setsid nohup ./servers.sh 1 5581 2 > /dev/null 2>&1 < /dev/null &   # 2 GR00T servers on GPU1, ports 5581-5582
ss -ltn | grep -E ":558[12] "                                       # ready when both ports listen (about 1 min)
```
- Each server needs about 6 GB of GPU memory. Logs go to `logs/server_<port>.log`.
- The frozen scenes (`scenes/*.pkl.gz`) are not in git. In a fresh checkout, make them with `tests/campaign.sh`,
  which also checks them.
- In Python, put this folder on the path:
  ```python
  import sys; sys.path.insert(0, "<repo>/runs/envkit")
  import envkit, stages, policy
  ```

## The API in one screen
```python
ep = envkit.Episode("KettleBoiling", 3)  # fresh env in the frozen scene (task, seed); about 30 s
ep.instruction                           # the task's language instruction
ep.obs                                   # the policy's observation: video.robot0_agentview_left / _right /
                                         #   eye_in_hand (1, 256, 256, 3) uint8, state.* (1, d), the instruction
client = policy.Client(5581)
chunk = policy.act(client, ep.obs, policy.chunk_seed("myexp", 0, len(ep.chunks)))  # {action key: (16, d)}
obs, chunk_success, done = ep.step(chunk)  # 16 env steps (fewer at the horizon), as the official eval
ep.t, ep.success, ep.success_step, ep.done  # steps so far; success at any chunk end so far; episode over
ep.chunks[-1]                            # record of that chunk: t, n, actions, step_success, success, obs hashes
stages.track(ep)                         # {t, stage, done: [bool per stage], official, ever, info}
S = ep.snapshot(); ep.restore(S)         # whole state, about 0.05 s each; restore drops the later chunk records
policy.rollout(ep, client, "myexp", policy_seed=0, on_chunk=None, max_chunks=None)  # policy until done
rec = ep.record()                        # plain record; envkit.replay(task, seed, rec["chunks"]) re-runs it exactly
ep.close()                               # always: frees the env and its render context
```
- **Success** is the official one: `_check_success()` after the last step of a chunk, OR over chunk ends
  (`ep.success`).
- **The horizon** is RoboCasa v1.0.1's `get_task_horizon(task)`. Episodes run to the horizon, as the official eval
  does; stop earlier only if your experiment says so.
- **Stages** (`stages.py`): the last stage is the official success. The others are read from the state now; they hold
  only while their result holds. Where the official success is a latch (WashLettuce, RinseSinkBasin, StirVegetables),
  every stage counts as done once it holds. `info` carries the official counters (e.g. `washed_time`).

## Recipes
**A plain run with the stage trace**
```python
ep = envkit.Episode(task, seed)
trace = [stages.track(ep)]
policy.rollout(ep, client, tag=f"plain/{task}/{seed}", on_chunk=lambda e: trace.append(stages.track(e)))
result = dict(task=task, seed=seed, success=ep.success, success_step=ep.success_step,
              first_stage_times={k: next(tr["t"] for tr in trace if tr["done"][k]) for k in range(len(trace[0]["done"]))
                                 if any(tr["done"][k] for tr in trace)})
ep.close()
```

**Branching at a chunk (resample / back-off / retry)**
```python
S = ep.snapshot()
outcomes = []
for k in range(8):                          # 8 candidate chunks for the same observation
    ep.restore(S)
    ep.step(policy.act(client, ep.obs, policy.chunk_seed(tag, k, len(ep.chunks))))
    policy.rollout(ep, client, f"{tag}/cand{k}", max_chunks=15)          # look ahead
    outcomes.append((k, stages.track(ep)["stage"], ep.success))
ep.restore(S)                               # back at the decision point; ep.chunks, ep.t, ep.success restored too
```
- A snapshot belongs to its `Episode` (it writes back into that env's objects). To carry a state to another process,
  record the actions and replay them.
- The candidates are exact: the observation is part of the snapshot. Look-aheads run the live policy, so they carry the
  render noise.

**Recording and replaying (exact); frames for a VLM**
```python
import gzip, pickle
with gzip.open(path, "wb") as f:
    pickle.dump(ep.record(), f)             # actions and per-step success of every chunk, observation hashes, machine
rec = pickle.load(gzip.open(path))
ep = envkit.Episode(rec["task"], rec["seed"], rec["horizon"])
frames, labels = [], []
for c in rec["chunks"]:
    ep.step(c["actions"])                   # the recorded run, bit for bit
    frames.append({k: v[0] for k, v in ep.obs.items() if k.startswith("video.")})  # (256, 256, 3) per camera
    labels.append(stages.track(ep))
ep.close()
```
Replayed frames can differ from the live run's by render noise (+-1 on a few pixels). `envkit.replay(task, seed,
chunks)` does the same in one call.

**Many episodes in parallel.** Use one episode at a time per process, many processes, spread over the server ports,
and one json line per episode. The pattern is in `tests/policy_tests.sh`:
```bash
export PY CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID OMP_NUM_THREADS   # from Setup
for s in $(seq 0 29); do for t in $TASKS; do echo "$t $s"; done; done | \
  xargs -P 8 -L 1 bash -c 'POLICY_PORT=$((5581 + RANDOM % 2)) $PY my_run.py $0 $1 >> results/runs.jsonl'
```
- Start long jobs detached (`setsid nohup ... &`), so a closed session does not kill them.
- A live episode takes 3-10 min (most of it physics and policy calls). Snapshot / restore cost almost nothing.

## Rules (each one broke an earlier experiment)
1. **Only read the env.** Never call `sim.forward()`, `set_state`, `_check_success()`, `update_state()` or anything
   that writes. Reasons:
   - one extra `forward()` changes the whole later trajectory;
   - `_check_success()` has side effects (RinseSinkBasin `washed_loc`, GetToastedBread `toaster_on`).

   Read official counters where they live.
2. **One `Episode` per episode, then `close()`.** Do not reuse an env across scenes. `Episode.official` is only for
   freezing scenes.
3. **Seed every policy call** with `policy.chunk_seed(tag, policy_seed, chunk_index)`. The same tag and seed give the
   same action noise, which pairs conditions.
4. **Record what you run** (`ep.record()`). It is the only way to reproduce a specific episode.
5. **Success and stages:**
   - success is `ep.chunks[-1]["success"]` / `ep.success` (the official check, loopholes included);
   - stage progress is `stages.track(ep)`, never a new success check.
6. **After any change to `envkit.py`, `stages.py` or the policy server, run the tests** in `tests/` before
   experiments: `check` and `stages` on the affected tasks, `policy` on a few seeds.
7. **Shared machine:**
   - check `nvidia-smi` before using a GPU;
   - never touch other users' processes;
   - keep at least 50 GB free on /home;
   - big binaries (states, videos, records) stay out of git;
   - to wait for a job, wait on its PID or its output file. `pgrep -f "<pattern>"` in a waiting shell also matches
     that shell's own command line and never ends.

## Where an experiment goes
Make `runs/YYYYMMDD_<name>/` with:
- a README: the question, the method, the results, how to re-run;
- its code and launchers;
- its results as jsonl.

Import envkit from `../envkit`. Large binary outputs go in `.gitignore`.

## Internals (only to change envkit)
- **Frozen scene:** the official reset's MjData and the whole Python object graph of the env, stored by attribute path
  and written back in place. RoboCasa's reset needed two fixes to be reproducible; envkit applies both at import:
  - render contexts freed by the garbage collector corrupted later images;
  - counter regions were ordered by memory address.
- **Snapshot:** the same capture in memory. A restore writes it back without `set_state` / `forward`.
- **Recorded last chunk:** cut at the horizon; `step` pads it, since the official wrapper indexes 16 rows.
- `../20261104_env_audit/README.md` lists every problem found and how it is checked.
