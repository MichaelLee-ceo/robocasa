# Critic data: candidate chunks at decision points, with simulator labels

Each decision point is a state on the plain policy's own trajectory (the seed-0 "main runs" of
`../20261025_seen480/run`).
- **Candidates.** The frozen GR00T N1.5 (composite-seen checkpoint-60000) samples N candidate first chunks, one action
  noise seed each.
- **Continuations.** Each candidate is executed, then the plain policy continues with fresh noise until the current
  stage's official condition holds or the stage budget L runs out. Each candidate is continued M times.
- **V.** The stage-finish rate of these rollouts is V = P(stage finished | state, plain policy) at that point.
- **Use.** This is the data for locating the critical chunks (`../20261028_critical`) and for training or evaluating a
  critic.

## Layout
```
data/<task>/<dataset>/results.jsonl   one row per decision point (labels per rollout, see below)
                      roll/           <tag>_t<t>_c<i>_m<m>.npz (sim state at every step, every chunk's actions) and .mp4
                      snap/           <tag>_t<t>.npz (3 cameras: RGB, depth, intrinsics, extrinsics; the exact policy request)
jobs/<task>.json                      job templates of the coarse points: scene, stage k, budget L (make_jobs.py)
jobs/dense_<task>.json                the dense points of the current jump intervals (../20261028_critical/make_dense_jobs.py)
logs/                                 quality checks: qc.jsonl, flags.jsonl, flags_reviewed.md (every flag investigated)
```
Load the data with `../20261028_critical/data.py`. It provides `rows(task, dataset)`, `roll(row)`, `snap(row)` and
`outcomes(row)`.

| dataset | what | N x M | KettleBoiling | SteamInMicrowave | PackIdenticalLunches |
|---|---|---|---|---|---|
| coarse | uniform decision points along the main runs (every 80 / 160 / 320 steps) | 8 x 2 | 250 | 262 | 256 |
| dense | V points inside the intervals where the coarse V jumps, refined to single chunks | 8 x 1 | 53 | 92 | 300 |
| confirm | the located critical chunks, salted seeds | 8 x 4 | 5 | 6 | 21 |
| control | random coarse points, same protocol as confirm | 8 x 4 | 5 | 6 | 21 |

**Row fields:**
- `tag`, `t` (step), `k` (stage), `L`, `N`, `M`;
- `hidden`, the restore version (see History);
- `restore` (settled-v5): the gripper ramp it started from, the finger gap and the contacts;
- `code`: hashes of the restore code;
- `py_state`, `k_replay`, `instr`;
- `src_dir`: the dir the row was collected into before the 2026-10-29 reorganization;
- `cands[i].rollouts[m]`:
  - `done`: stage k finished;
  - `prev_ok`: the earlier stages still hold;
  - `steps`, `success`, `reached`, `progress`;
  - `obs_fp`;
  - settled-v5 only: `bounds`, per chunk boundary [t, bitmask of the stages whose check holds, success steps in the
    chunk, success at the boundary]. Other definitions of "finished" can be computed from these.

## How a decision point is collected (`collect.py`)
1. `load_scene` restores the frozen scene.
2. Replaying the main run's saved states up to t rebuilds the python fixture state and the stage memory.
3. The robot controllers' hidden state is set with `hidden_state.settle` (`../scripts/hidden_state.py`). At t = 0 the
   state right after `load_scene` is exact. It is captured once and restored before every rollout, together with the
   observation clocks.
4. Seeds:
   - candidate i's first chunk: sha256(`tag|critic|t|ci|m0|0`);
   - continuation chunks: add m and the job's salt.
5. Live asserts: every rollout starts at the main run's state at t, and the continuations of a candidate share their
   first chunk bit for bit.

Running a collection:
- `servers.sh <gpu> <port0>` keeps 4 policy servers up.
- `pool.sh <queue dir> <jobs.json> <workers>`, with `PORTS="..."`, starts workers on a shared queue. A job's `out`
  field names its dataset dir.
- `watchdog.sh <jobs.json>` runs detached:
  - it restarts dead servers;
  - it runs `qc.py` on every new row;
  - it retries failed jobs, but blocks live-assert failures as bugs;
  - it writes DONE when finished.

## History (details in ../20261029_review)
**Restore bugs.** Restoring a saved state misses the controllers' hidden state. Five bugs were found and fixed:
1. no controller state at all;
2. the base-mode arm goal set in the world frame;
3. the base start pose overwritten;
4. a wide grasp read as opening;
5. the gripper ramp restored as +-1. At t = 0 its true value is 0, and during grasp or release starts it is in between.

Bug 5 was fixed together with the observation clock and the toaster state. The fix version is settled-v5:
- the gripper ramp is the servo target, 1 - (gap + 2 x gap change) / 40 mm, with contact rules;
- validation: `../20261029_review/fix/README.md`.

**Which rows came from which restore.** Every affected point was re-collected. A row from an earlier restore remains
only where that restore gives the same hidden state as settled-v5: the same gripper ramp, a base turned < 2 degrees,
and t > 0. See `hidden`.

**2026-10-29 reorganization.**
- The layered dirs (`run_*`, `_fix`, `_fix3`, `_fix5`, `dense2`, `confirm5`, ...) became `data/<task>/<dataset>`.
- Every superseded row and the old confirmations were deleted.
- `data.py` before and after returned identical rows, and the same files (hard links), at all 1,227 points.
- Key results computed on the old layout (critical chunks, confirm / control gains) are unchanged when recomputed.
