# Data for the VLM critic (started 2026-10-26)

The goal is a critic that picks, among the frozen policy's candidate chunks, the one that will finish the current stage.
The labels come from the simulator.

- `make_jobs.py` picks the decision points.
- `collect.py` collects them; its docstring describes everything it saves, and the file format.
- `run.sh` starts 4 seeded GR00T servers and the workers on one GPU.
- `start_when_free.sh` starts a task's collection on the first free GPU.

## Setup
For each task, the decision points come from the 30 seed-0 episodes of `runs/20261025_seen480`:
- KettleBoiling: every 80 steps, 250 points.
- SteamInMicrowave: every 160 steps, 262 points.

At each point the policy proposes 8 candidate first chunks. Each candidate is continued twice: its first chunk is
executed, then the plain policy runs with fresh noise until the stage finishes or L steps pass. That is 16 rollouts
per point.

Saved once per point:
- the sim state;
- RGB and depth from the 3 cameras at 480x480, plus each camera's intrinsics and extrinsics;
- wrist depth with the robot hidden;
- the exact policy request.

Saved per rollout:
- every chunk;
- the sim state after every step;
- a small video;
- labels: done, prev_ok, steps, success, and progress.

Everything else is derived offline from the saved states: gripper paths, contacts, progress curves, events, overlays,
and GR00T features.

## Runs
| dir | what |
|---|---|
| `run/` | KettleBoiling on GPU1, started 12:31; 2 of its points were collected earlier on GPU2. Done when `run.out` shows `done:`. |
| `run_SteamInMicrowave/` | SteamInMicrowave on GPU1. `start_when_free.sh` started it at 17:23, once the KettleBoiling run was done. It finished at 01:56 with 261/262; s27 t960 failed to rebuild its scene (`could not initialize task`) and was rerun alone at 05:00. Total 262/262, 3.4 GB. The failed row stays in the jsonl. |
| `run_v2/s0..s4/` | KettleBoiling re-collected with the fix on nano4 (2026-10-27, 250/250). Use this instead of `run/`. |
| `run_SteamInMicrowave_v2/` | SteamInMicrowave re-collected with the fix on GPU1, 14 workers. Started 08:40 on 2026-10-27. **Paused at 12:01 for a machine shutdown with 124/262 points, 0 errors** (about 0.6 points/min while the machine's load was ~138). Resume command below. Use this instead of `run_SteamInMicrowave/`. |
| `run_PackIdenticalLunches/` | PackIdenticalLunches (with the fix): every 320 steps, 256 points. **Not started (0/256):** GPU0 and GPU3 stayed fully used by another user from 08:39 until the watcher was stopped at 12:01. `start_when_free.sh` starts it on GPU0 or GPU3 when one is free, with workers = min(12, idle CPUs - 4); it waits while fewer than 6 workers would fit. Log: `run_PackIdenticalLunches.out`. Smoke test (k = 3, `k_replay` = 3, shared chunk identical) in `smoke_PackIdenticalLunches/`. |
| `run_aborted_oom/` | First GPU2 try. 12 workers plus a second GL context per worker did not fit in 24 GB. |
| `smoke/`, `smoke_SteamInMicrowave/` | Smoke tests (N = 2, L = 48). |

GPU2 was tried with 10 workers. It ran correctly, but its memory peaked at the 24 GB limit, so it was stopped and the
collection moved to GPU1.

## Notes
- The env-only wrist depth is rendered through the env's own render context, with the robot geoms and sites made
  invisible. The old `mujoco.Renderer` path also drew collision geoms, so the kettle outline and counter edges were
  off.
- Some state lives outside the sim state, for example `microwave._turned_on` and the stage checks' memory S. It is
  rebuilt by replaying the episode's saved states up to the decision point. The replayed stage `k_replay` matched the
  logged stage in the SteamInMicrowave smoke test (k = 3).
- At 70 of 250 KettleBoiling points:
  - 33 have mixed candidates (some finish the stage, some do not);
  - 26 have no candidate finishing;
  - 11 have every candidate finishing.

## Bug (found 2026-10-27): rollouts inherited the previous rollout's controller state; run/ and run_SteamInMicrowave/ are affected
`set_state()` restores the sim state but not the robot controllers' internal state:
- the gripper command ramp (`PandaGripper.current_action`);
- the arm goal accumulated in base mode;
- the controllers' cached kinematics;
- MuJoCo's warm start.

All 16 rollouts of a decision point ran one after another in the same env, so each rollout started with whatever the
previous one left behind.

Evidence:
- The two continuations of a candidate share the start state and the first 16 actions, yet they diverged at the first
  step in 100 % of pairs.
- In `check_restore.py`, the same chunk run after a different preceding rollout ended 11.7 mm apart at the fingers
  after 16 steps.
- How wide continuation 1 opened its gripper in its first steps correlates (+0.5) with the gripper state continuation 0
  ended in.
- Some bins have negative within-candidate covariance, which is impossible without such a dependence.

Fix: `../scripts/hidden_state.py` adds `capture`, `restore` and `settle`, and `rollout(..., hidden=)` uses them.
`collect.py` now settles the controllers at each decision point and restores that state before every rollout.

| `check_restore.py` (KettleBoiling s0, step 160) | max abs state diff | after 16 steps |
|---|---|---|
| `set_state` only (as collected) | 2.2 | arm 0.25 deg, fingers 11.7 mm |
| restore the captured hidden state | 0 | 0 |
| restore the settled hidden state | 0 | 0 |
| continuous run vs exact restore | 0 | 0 |
| continuous run vs settle (decision-point approximation) | 9e-5 | arm 0.001 deg, fingers 0 mm |

What this invalidates:
- The "Finding" below.
- The per-phase analysis.
- Any use of run/ and run_SteamInMicrowave/ that compares the rollouts of a decision point.

The rollouts are still physically valid trajectories, but each one starts with a perturbation that depends on the
previous rollout. These runs must be re-collected.

## Clean re-collection: run_v2/ (KettleBoiling, nano4, 2026-10-27; `analyze_phase.py`)
How it was run:
- With the fix: `hidden="settled"`, and the shared first chunk is bit-identical across both continuations
  (480 pairs checked).
- 250/250 decision points, in 5 nano4 jobs, each with 1 H200 and 12 cores. Total 3.4 GPU-hours, including the failed
  attempts.
- Renders match this machine. 20 decision points were re-rendered here, including the 12 that stalled longest; they
  differ by at most 1 intensity level.
- 6 points first failed `get_real_depth_map`'s range check after GPU stalls. Their shards were resubmitted and those
  points succeeded on the rerun.

Results:

| decision points | same-candidate vs different-candidate agreement | between-candidate variance of P(done), 95 % CI | perfect best-of-8 picker |
|---|---|---|---|
| all 250 | 0.89 vs 0.87 | +0.010 [+0.003, +0.017] | about +14 points [8, 19] |
| S0, steps 0-80 of the stage | 0.73 vs 0.68 | +0.027 [-0.004, +0.058] | about +23 points [0, 34] |
| S0, steps 80-160 | 0.85 vs 0.78 | +0.033 [+0.005, +0.064] | about +26 points [10, 36] |
| S0, steps 160-240 | 0.94 vs 0.92 | +0.007 [-0.004, +0.020] | about +12 points [0, 20] |
| S0, steps 240-400 | 0.94 vs 0.94 | +0.002 [-0.008, +0.012] | about +6 points [0, 16] |
| S1 (turning the knob), all bins | — | small n, CIs include 0; one bin is negative | — |

- **The candidate chunk does change the outcome.** The earlier "no effect" came from the bug.
- **The effect is concentrated early in S0**, during the approach to the kettle and the grasp. Late in the stage the
  outcome is mostly already decided: most points have all 16 rollouts finish, or none.
- Only 99 of 250 points have mixed outcomes.

## (Invalidated by the bug above) Finding: the first chunk does not decide the stage outcome (`analyze_labels.py`, 2026-10-26 23:15)
Within a decision point, the two continuations of the same candidate share the first chunk, while continuations of
two different candidates share nothing. The covariance of the two continuations across candidates is an unbiased
estimate of how much the stage outcome varies with the first chunk.

| run | points | between-candidate variance of P(done), 95 % CI | gain of a perfect best-of-8 chunk picker |
|---|---|---|---|
| KettleBoiling | 250 | −0.0005 [−0.006, +0.005] | about 0, at most 10 points |
| SteamInMicrowave | 262 | −0.007 [−0.015, +0.001] | about 0, at most 5 points |

- Agreement is the same within and across candidates:
  - KettleBoiling: 0.84 for two continuations of the same candidate vs 0.85 for two different candidates.
  - SteamInMicrowave: 0.73 vs 0.76.
- Picking the candidates whose continuation 0 finished does not raise the success rate of continuation 1:
  - KettleBoiling: 0.72 vs 0.72.
  - SteamInMicrowave: 0.58 vs 0.61.
- The only hint of an effect is at points where the candidates disagree on closing the gripper within the first chunk
  (17 and 11 points). There the CI reaches +0.03, but closing early does not predict the outcome.
- So at these uniformly sampled points, the outcome is set by the noise of the later chunks, not by the first one. A
  critic that only chooses the first chunk cannot do better than random, however good it is.
- The search ceiling worked because it chose among whole stage-length rollouts.

## Check progress
```
tail -2 run_SteamInMicrowave_v2.out run_PackIdenticalLunches.out
cat run_SteamInMicrowave_v2/results_w*.jsonl | grep -vc '"error"'   # points done (of 262)
cat run_PackIdenticalLunches/results_w*.jsonl | grep -vc '"error"'  # (of 256)
pgrep -fc "^/bin/bash ./start_when_free.sh"     # 1 while the watcher still waits
```

## Resume (after the 2026-10-27 shutdown)
`collect.py` skips every point that already has an error-free row in the output dir, whatever the worker count, so
rerunning the same command continues the run. The points that were in flight when it was stopped have partial files in
`snap/` and `roll/`; they are recomputed and overwritten. Check that GPU1 is free first.
```
cd runs/20261026_critic_data
setsid nohup ./run.sh run_SteamInMicrowave_v2 jobs_SteamInMicrowave.json 14 4 >> run_SteamInMicrowave_v2.out 2>&1 < /dev/null &
setsid nohup ./start_when_free.sh PackIdenticalLunches >> run_PackIdenticalLunches.out 2>&1 < /dev/null &
```
Then analyse with `python analyze_phase.py "run_SteamInMicrowave_v2/results_w*.jsonl"`.

## Next
1. Label statistics.
2. How much of the outcome the first chunk explains, using the variance between the two continuations of each
   candidate.
3. Offline renders of the overlay variants.
4. Zero-shot evaluation of Qwen3.8-27B-FP8 on GPU1.
