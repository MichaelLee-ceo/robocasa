# Data-integrity review of the critic data (review-integrity, 2026-10-29)

Written by the review-integrity agent and saved here by the main session, because the agent could not write .md
files.

**Scope.**
- Every error-free row of:
  - the current coarse rows, in the override order of `effect_by_event.points()`;
  - `run_dense_<task>` (valid);
  - `run_confirm_KettleBoiling`;
  - the two `_bug3` dirs (positive control).
- Also scanned: the superseded coarse rows; `run_dense2_*` (complete at 03:52); 30 points each of the bug-1 dirs
  `run/` and `run_SteamInMicrowave/` (positive control for check 1).
- No simulation was run and no GPU was used. Every rollout npz was read in full, so each file's CRC was checked.
- Scripts and outputs are in this folder; the commands are at the end.

## Verdict in short

- **Bookkeeping, file health and the restore are clean.** Across 1,800 points, 24,336 rollout npz, 10,224
  continuation pairs and 50,400 candidate pairs, checks 1-6 fail only in the positive controls and on the items below.
- **New bug A: `settle()` sets a wrong gripper ramp at t = 0 (91 points).** Every rollout there first closes the
  gripper from 41 to 24 mm, while the main run opens it at once. The true ramp at t = 0 is known to be 0.
- **New bug B: the v4 gripper rule reads a gripper that is still opening near fully open (77-78.5 mm) as closing.**
  - In every rollout of 6 of the 46 dense2 points, the fingers close from 78 to 34 mm and reopen.
  - It is live in the running confirmation: SteamInMicrowave s29 t976 is already collected this way (32 of 32
    rollouts). SteamInMicrowave s7 t960 and PackIdenticalLunches s14 t16 are queued.
- **Bug 4 left 2 valid dense points behind:** SteamInMicrowave s4 t192 and t224. They are harmless.
- **Latent bug C: py_state misses the Toaster's dict state**, which leaks from one rollout into the next. The one
  case is in a `_bug3` dir; no valid start state is affected.
- **Noise N1:** the first-request obs_fp differs between rollouts of the same point at 1,128 of 1,800 points.
  - The cause is RGB render noise of +-1-2 intensity levels. The proprio state and the sim state are bit-identical.
  - Side effect: re-collected points do **not** reproduce the old candidates. Only 23-77 % of first chunks are
    identical, against the README's "same first chunks".

## 1. What was checked, with counts

Per dir: points, rollouts, continuation pairs (in brackets: pairs where both run past step 16), and violations.

| group | points | rollouts | cont. pairs | first-chunk / states[0..16] mismatch | cand. pairs / identical first chunk | states[0] != main[t] | obs_fp not unique | row-vs-npz violations |
|---|---|---|---|---|---|---|---|---|
| current run_v2/s0..s4 (KettleBoiling) | 249 | 3984 | 1992 (1935) | 0 | 6972 / 0 | 0 | 0 | 0 |
| current run_KettleBoiling_fix | 1 | 16 | 8 (8) | 0 | 28 / 0 | 0 | 0 | 0 |
| current run_SteamInMicrowave_v2 | 105 | 1680 | 840 (828) | 0 | 2940 / 0 | 0 | 5 | 0 |
| current run_SteamInMicrowave_fix | 3 | 48 | 24 (24) | 0 | 84 / 0 | 0 | 0 | 0 |
| current run_SteamInMicrowave_fix3 | 154 | 2464 | 1232 (1164) | 0 | 4312 / 0 | 0 | 123 | 0 |
| current run_PackIdenticalLunches | 110 | 1760 | 880 (856) | 0 | 3080 / 0 | 0 | 107 | 0 |
| current run_PackIdenticalLunches_fix | 20 | 320 | 160 (160) | 0 | 560 / 0 | 0 | 19 | 0 |
| current run_PackIdenticalLunches_fix3 | 126 | 2016 | 1008 (984) | 0 | 3528 / 0 | 0 | 105 | 0 |
| dense KettleBoiling / SteamInMicrowave / PackIdenticalLunches (M = 1) | 45 / 66 / 246 | 360 / 528 / 1968 | n/a | n/a | 1260 / 1848 / 6888, 0 identical | 0 | 20 / 44 / 234 | 0 |
| confirm KettleBoiling (M = 4) | 12 | 384 | 576 (576) | 0 | 336 / 0 | 0 | 5 | 0 |
| dense2 SteamInMicrowave / PackIdenticalLunches | 18 / 28 | 144 / 224 | n/a | n/a | 504 / 784, 0 identical | 0 | 10 / 26 | 0 |
| superseded (KettleBoiling 1, SteamInMicrowave 186, PackIdenticalLunches 191) | 378 | 6048 | 3024 (2932) | 0 | 10584 / 0 | 0 | 276 | 0 |
| **_bug3 SteamInMicrowave / PackIdenticalLunches (positive control)** | 62 / 117 | 496 / 936 | n/a | n/a | 0 identical | 0 / **1** | 21 / 109 | 0 |
| **bug-1 run/, run_SteamInMicrowave/ (positive control)** | 30 / 30 | 480 / 480 | 240 / 240 | **240 / 240** | 0 identical | 0 | 14 / 10 | 0 (\*) |

(\*) `run/` rows predate `k_replay` and `prev_ok`. The 30 "k_replay != k" hits there mean "field missing", not a
violation.

**Check 1, shared first chunk.**
- Every continuation pair has bit-identical first-chunk actions (all 5 keys) and bit-identical states[0..16]: 6,720
  valid pairs plus 3,024 superseded pairs.
- Every pair that runs past step 16 first differs at row 17, exactly when chunk 1 starts.
- The positive control works: in the bug-1 dirs, 480 of 480 pairs already differ at row 1 (example:
  `run/KettleBoiling_s27_single_t160_c0` differs by up to 1.29 at step 1).

**Check 2, start state.**
- states[0] equals the main run's state at t bit-for-bit in 24,335 of 24,336 rollouts. The exception is the Toaster
  leak (C).
- The snap state equals the main run's state at t at all 1,800 points.
- The first-request obs_fp is not identical within a point at 1,128 of 1,800 points (see N1).

**Check 3, seeds.** No two candidates of a point share a first chunk: 0 of 50,400 pairs, compared on all keys and on
the end-effector position alone. No continuation pair shares chunk 1: 0 of the 9,930 pairs where both run past
step 16.

**Check 4, row vs npz.** The exact rule, derived from `rollout()` and `MultiStepWrapper`:
- The robocasa env runs with `ignore_done=True`, and truncation fires at `len(reward) >= L`, where every L is a
  multiple of 16. So every chunk executes all 16 steps.
- The loop ends after the chunk in which the stage finished, success happened, or L was reached.
- Hence steps = 16 x chunks and len(states) = steps + 1. If done, steps <= L; if not done, steps = L and reached = k.

Results:
- **0 violations** of this rule in any group. All 24,336 rollouts have a step count divisible by 16.
- Time column: every rollout advances exactly 0.05 s per row (largest deviation 2.4e-13). All values are finite.
- Every rollout has all 5 action keys with shape (C, 16, d) as float32, and states as float64 of width 1 + nq + nv.
- For every current, dense, confirm and dense2 row: k_replay = k, py_state is present and non-empty, and N, M, L, k
  and episode_success match both the job and the main run's stage history.
- The hidden tag is consistent everywhere, and the physical signatures in section 3 agree with it:
  - coarse level-0 dirs: settled;
  - fix: settled-basefix;
  - fix3: settled-v3, plus settled-v4 for s4 t160;
  - dense KettleBoiling and _bug3: settled-basefix;
  - dense SteamInMicrowave, dense PackIdenticalLunches, confirm KettleBoiling: settled-v3;
  - dense2: settled-v4.
- Two harmless nits:
  - `progress = -0.001` in 10 SteamInMicrowave k=2 rollouts: a rounding artifact; the value is not clipped to [0, 1].
  - In SteamInMicrowave, 13 % of k=1 and 24 % of k=2 rollouts with `done=True` have `prev_ok=False`. Analyses that
    use only `done` count them as finishes. This is a labels question, not integrity.

**Check 5, coverage.**
- Every job has exactly one error-free current row: coarse 250 / 262 / 256, dense 45 / 66 / 246, confirm
  KettleBoiling 12, dense2 18 / 28.
- Duplicate rows (same dir, same override level, or across the KettleBoiling shards): 0. Unparseable lines: 0. Rows
  that no jobs file lists: 0.
- All 20 error rows have an error-free replacement in the same dir.
- Every point of `jobs_fix*` is current from its fix dir. 29 of SteamInMicrowave's 32 fix points are current from
  fix3, correctly, since they are also in `jobs_fix3`; s4 t160 is current from fix3 as settled-v4.
- Superseded rows: run_v2/s3 1; run_SteamInMicrowave_v2 157; run_SteamInMicrowave_fix 29; run_PackIdenticalLunches
  146; run_PackIdenticalLunches_fix 45.

**Check 6, file health.**
- 0 npz that fail to load, 0 CRC errors, 0 missing npz/mp4/snap files.
- All 1,800 snaps pass `testzip`, and every mp4 has a moov atom.
- In every point, the roll-file modification times increase in rollout order and lie between (snap time - wall_s)
  and the snap time. Nothing was overwritten later.
- Orphans exist only in the `_bug3` dirs: 8 + 13 partial points (36 + 62 files), and 11 + 18 claims without rows, left
  by the killed workers. Those dirs are invalid anyway; keep glob-based tools away from them.

## 2. Anomalies

### A. BUG: `settle()` overrides the known initial gripper ramp at t = 0 (91 points, all rollouts)
- At the episode start the ramp is exactly 0: `robot.reset()` sets `gripper.current_action = zeros`.
- `decision_point` still calls `settle(kenv, None)`. It reads the 41.1 mm init gap as "closing" and sets the ramp to
  +1. Every candidate commands open at t = 0, so every rollout closes first, then opens:

  | step | 0 | 1 | 2 | 3 | 4 | 7 | 16 |
  |---|---|---|---|---|---|---|---|
  | settled rollouts, gap (mm) | 41.1 | 30.4 | 24.3 | 23.6 | 25.8 | 43.4 | 77.6 |
  | main run, gap (mm) | 41.1 | 42.8 | 46.9 | 52.7 | 59.5 | 74.4 | 78.2 |

- The bug-1 run's rollout c0m0 got no settle and so kept the fresh, true state. It reproduces the main run's profile
  exactly (41.1, 42.8, 46.9, 52.8, 59.6, ...).
- **Scope:**
  - all 90 coarse t = 0 points (30 per task, in run_v2/s*, run_SteamInMicrowave_v2, run_PackIdenticalLunches,
    run_PackIdenticalLunches_fix);
  - confirm KettleBoiling s29 t0;
  - 16 of the 58 dense intervals in `critical_chunks.json`, which start from these V(0) values;
  - 4 located critical chunks that start at t = 0: KettleBoiling s29 0-16 (a confirm point), SteamInMicrowave s4 0-32
    and s12 0-16, PackIdenticalLunches s27 0-16.
- **Impact on V: unknown.** The robot is usually far from objects at t = 0, but the artifact shifts the gripper by up
  to 30 mm for about 8 steps, in early S0, where the README finds the chunk matters most.
- **Reproduce:** `$P show.py gripper run_v2/s3 KettleBoiling_s29_single 0` (any t = 0 point gives the same result).

### B. BUG (a regression from the bug-4 fix): the v4 rule reads a gripper still opening near fully open as closing (live now)
- v4 counts a gap as "opening" only if it is at least 78.5 mm or widened by more than 0.5 mm in the last step. Near
  the limit the fingers open by only 0.1-0.3 mm per step, so a gripper at 77-78.5 mm that is still opening is read as
  closing.
- Every candidate commands open. Yet the fingers close from 78 to 34 mm by step 4, then reopen to 77.6 mm by step 16.
  The old 75.2 mm rule got all of these cases right.
- **Exposure** (all rollouts at each point show it):
  - dense2 SteamInMicrowave s12 t16, s29 t976, s8 t1072, s9 t816; dense2 PackIdenticalLunches s14 t16, s27 t16;
  - confirm SteamInMicrowave s29 t976, already collected: all 32 rollouts close 78.3 -> 34.5 mm. Candidates c3-c7
    finish in 16 steps (4 of 4); c0-c2 never finish (0 of 4);
  - queued: confirm SteamInMicrowave s7 t960 (gap 77.9 mm, up 1.3 mm over the last 4 steps) and PackIdenticalLunches
    s14 t16.
- These sit exactly at critical chunks, and the jumps there mix rule versions. SteamInMicrowave s29 976->992 (jump
  -0.62) and PackIdenticalLunches s14 16->32 (jump -0.50) each go from an artifact point (v4) to a clean point (v3).
- Overall, 9 of the 58 located chunks have an endpoint with this artifact; 3 of those are among the 32 jumps >= 0.4.
- **Reproduce:** `$P show.py gripper run_dense2_SteamInMicrowave SteamInMicrowave_s12_single 16`;
  `$P gripper_artifacts.py` lists all affected points.

### B2. Bug 4 left 2 valid dense points behind (harmless)
- In `run_dense_SteamInMicrowave` (settled-v3), s4 t192 and t224 hold the vegetable at 76.0 and 75.5 mm, gap steady.
  The old rule read them as opening.
- All 8 close-commanding rollouts open about 1.3 mm at step 1, then close again. Nothing falls, and done is 8 of 8 at
  both points.
- **Reproduce:** `$P show.py gripper run_dense_SteamInMicrowave SteamInMicrowave_s4_single 192`

### B3. Suspect: a +-1 ramp at gaps held steady mid-range (11 points, SteamInMicrowave k = 2, "close the door")
- fix3 s28 t960, s5 t960, s9 t800, s8 t800; dense s8 t832/864/896/928 and s21 t864/896/928.
- The main run's fingers hold steady at 39-73 mm while touching the microwave. `settle` sets the ramp to +-1, and
  every candidate that commands open first closes by more than 5 mm.
- At s28 t960 and s21 t896 the gap is 40.8 mm, the target of a ramp of about 0 (alternating commands). The truth there
  is most likely "neutral", which a +-1 ramp cannot represent.
- The main run's gripper commands are not saved, so this cannot be settled from the data. Treat these points as
  suspect.

### C. LATENT BUG: python fixture state with non-scalar attributes leaks between rollouts
- `imagine.py_state` copies only the scalar attributes of each fixture. `Toaster.update_state` keeps `_turned_on`,
  `_num_steps_on`, `_cooldown` and `_state` in dicts, and writes qpos (`set_lever`).
- Case: `run_dense_PackIdenticalLunches_bug3`, PackIdenticalLunches s15 t64. c4m0 pushes the right lever down at step
  294. c5m0 then starts with the lever forced down (qpos[55] = 0.0609, against -0.0004 in the main run), so its
  states[0] differs from the main run's state at t. The lever pops up after the toaster's 500-step timer, and c6m0
  starts clean.
- No case appears in the valid data: every start state matches. But any rollout that turns a toaster on would corrupt
  the next one.
- **Reproduce:** `$P show.py joint run_dense_PackIdenticalLunches_bug3 PackIdenticalLunches_s15_single 64 toaster_main_group_sideR_lever_joint`

### N1. NOISE: render nondeterminism makes obs_fp differ within points
- Within a point all states are bit-identical, yet obs_fp differs at 1,128 of 1,800 points. Only c0m0's request is
  saved (in the snap), so the request components were compared across 655 pairs of snaps of the same (tag, t):
  - the proprio state, the instruction, K, T, depth, depth_env and the sim state are bit-identical in all 655;
  - RGB differs in 21-60 % of pairs, by at most 2 levels, in a median of 5-13 pixels (at most 466 of 230k);
  - example: `$P show.py snapdiff run_SteamInMicrowave_v2 run_SteamInMicrowave_fix3 SteamInMicrowave_s13_single 640`
    shows 1 and 15 differing pixels, each off by 1.
- The variation tracks the camera view: SteamInMicrowave v2 points whose base had turned >= 5 deg vary in 102 of 153,
  those below 5 deg in 5 of 109, and KettleBoiling run_v2 (collected on nano4) in 0 of 249.
- **Judgement: noise.** It is harmless for continuations m >= 1, which replay chunk 0 rather than query the policy.
- **Side effect:** the same observation and seed does not give the same first chunk.
  - Even with identical obs_fp, the first chunk differs in 388 of 2,036 cross-dir candidate pairs (the policy server is
    nondeterministic).
  - Re-collected candidates that match the superseded ones: v2 -> fix 198 of 256 (SteamInMicrowave) and 142 of 520
    (PackIdenticalLunches); v2 -> fix3 279 of 1232 (SteamInMicrowave) and 242 of 1008 (PackIdenticalLunches).
  - Never pair candidates across dirs by index.
- Testing render determinism within a point from the mp4 frames (`render_noise.py`) was inconclusive, because H.264
  lookahead makes early frames depend on later ones.

## 3. Positive controls and physical signatures of bugs 1-4 (`signatures.py`, `drops.py`, `gripper_rules.py`)

| bug | data-level signature | superseded / invalid rows | current, dense, dense2, confirm rows |
|---|---|---|---|
| 1 | continuations diverge before step 17 | 480 / 480 pairs (bug-1 dirs) | 0 / 6,720 pairs (and 0 / 3,024 superseded) |
| 2 | arm joint speed in the first 16 steps, rollouts starting in base mode | settled rows: median 10 rad/s, > 3 rad/s in 97-100 % (688 + 420 + 4 rollouts) | max 1.27 rad/s over 1,618 base-start rollouts |
| 3 | realized base velocity turned by -dyaw relative to the command, tested at \|dyaw\| >= 45 deg | bug-3-like at 62 of 72 points (3 correct-like, 7 ambiguous) | correct-like at 57 of 57 points |
| 2/4 | at least half of a point's rollouts drop a held object >= 10 cm within 16 steps | 35 points (e.g. s4 t160 16/16, bowls at t480/t640) | 0 points |

- KettleBoiling: the base turns at most 3.8 deg (0 of 307 points at >= 5 deg), so bug 3 is negligible there, as
  believed.
- Residual bug-3 approximation: in the current settled and basefix PackIdenticalLunches rows, 41 points have 2-5 deg of
  yaw and base motion. Their base commands are turned by 2-5 deg, a few cm of drift per metre driven.

## 4. Recommended actions (no fixes applied)
1. **Fix the gripper part of `hidden_state.settle`**, then re-check with `gripper_rules.py`.
   - (a) At t = 0, do not settle. Capture the state right after `load_scene` / `replay`: it is the exact true hidden
     state (ramp 0).
   - (b) Infer the ramp *value*, not just its side:
     - opening, if the gap rose by more than 0.2 mm over the last 4 steps or the finger velocity at t is positive;
     - closing, if the gap fell;
     - steady with nothing between the fingers: ramp ~ 1 - gap / 40 mm, rounded to the nearest 0.2;
     - steady around an object: +1.

     This fixes A, B and B3, and keeps s4 t160 (steady at 76.8 mm around an object) at closing.
   - (c) Log the inferred ramp and gap in each row, so affected points can be found later.
2. **The running confirmation:** patch the code before SteamInMicrowave s7 t960 and PackIdenticalLunches s14 t16 run,
   or mark them. Re-collect SteamInMicrowave s29 t976 after the fix, then recompute those confirm rows and their
   critical chunks.
3. **Re-collect after the fix:**
   - the 90 coarse t = 0 points and confirm KettleBoiling s29 t0;
   - the 6 dense2 points of B and the 11 points of B3;
   - optionally dense SteamInMicrowave s4 t192/t224 (B2).

   Then rerun `dense_analysis.py` for the 16 intervals that start at t = 0 and the 9 chunks with an artifact endpoint.
4. **Fix `py_state` / `set_py_state`** to deep-copy every attribute of the fixtures that override `update_state`, not
   only the scalars (C).
5. **Add two cheap live assertions to `collect.py`:** states[0] equals the main run's state at t, and states[:17] is
   identical across the continuations of a candidate. They would have caught bug 1 and C at collection time.
6. **Analysis hygiene:**
   - do not use obs_fp as a determinism check, and do not pair candidates across dirs (N1);
   - keep the `_bug3` orphan files out of globs;
   - decide whether `done && !prev_ok` counts as a finish;
   - optionally re-collect the 41 PackIdenticalLunches points with 2-5 deg of yaw and base motion.

## Reproduce
```
cd /home/michaellee/mclee/robocasa/runs/20261029_review/integrity; P="/home/michaellee/mclee/robocasa/envs/groot/bin/python -I"
$P layout.py 3          # joint layout per scene, by compiling each scene XML -> layout.json
$P scan.py 3            # full per-point scan, about 4 min -> scan.jsonl
$P coverage.py          # check 5 and the orphan files -> coverage.json
$P analyze.py           # checks 1-4 and 6 tables, anomaly lists -> analyze.json
$P signatures.py; $P drops.py; $P gripper_rules.py; $P gripper_artifacts.py; $P crossdir.py 3
$P show.py gripper|joint|snapdiff ...   # single findings, see the sections above
```
