# V swings and labels of the critic data (review-labels, 2026-10-29, updated 05:05)

Written by the review-labels agent and saved here by the main session, because the agent could not write .md files.

**Environment for every command below:**
```
cd /home/michaellee/mclee/robocasa/runs/20261029_review/labels
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 MUJOCO_GL=egl
PY="/home/michaellee/mclee/robocasa/envs/groot/bin/python -I"
```

**Rows used** (override order as in `effect_by_event.points()`):

| | coarse | dense | dense2 | total |
|---|---|---|---|---|
| KettleBoiling | 250 | 45 | none (0 jobs) | 295 |
| SteamInMicrowave | 262 | 66 | 17 at the time of the V run | 345 |
| PackIdenticalLunches | 256 | 246 | 28 | 530 |

- dense2 is now complete (18/18 and 28/28). Re-run `vseries.py`, `vswing_test.py` and `calibration.py` to include
  the last row.
- Code versions in the current rows:
  - KettleBoiling coarse rows: `settled` (v1); KettleBoiling dense: `basefix`;
  - SteamInMicrowave and PackIdenticalLunches: v1 / `basefix` / v3 coarse; v3 dense;
  - dense2 and SteamInMicrowave s4 t160: v4.
- The `git diff` of `hidden_state.py` shows that v1 differs from the current code only in the three fixes (base start
  pose, arm goal frame, gripper rule).

## 1. V swings

### What was checked
`vseries.py` builds V = mean(done) per (episode, stage), sorted by t, and flags:
- **DIP**: V >= 0.8, then a run of any length at <= 0.2, then >= 0.8;
- **SPIKE**: the mirror of DIP;
- **JR**: |dV| >= 0.75 followed by a reversal of >= 0.5.

Binomial null: (a) a constant true rate within each triple; (b) a parametric bootstrap with a monotone truth per
series (isotonic fit), 2000 replicates.

### Results

| task | series | points | observed DIP / SPIKE / JR | expected from noise, (a) / (b) |
|---|---|---|---|---|
| KettleBoiling | 53 | 295 | 0 / 0 / 0 | 0.001 / 0.001 / 0.009-0.017 |
| SteamInMicrowave | 97 | 345 | 0 / 0 / 0 | 0.001-0.005 / 0.001-0.002 / 0.03-0.04 |
| PackIdenticalLunches | 81 | 530 | 0 / 0 / 0 | 0.005-0.008 / 0.005-0.007 / 0.11-0.17 |

- The same flags on V_ok (done and prev_ok) also give 0.
- **Positive control** (`positive_control.py`, superseded rows):
  - SteamInMicrowave: 11 DIP + 11 JR (s0, s2, s4 t160, s5, s8 x2, s13, s14, s15, s17, s18, s19, s23, s24, s25), with
    worst-case p from 1e-7 to 3e-15;
  - PackIdenticalLunches: 1 JR (s5 S0 0:14/16 -> 320:0/16 -> 640:10/16);
  - KettleBoiling: 0.

  The detector therefore catches swings of the size bugs 2-4 produced.
- **Weaker swings.** A one-sided Fisher intersection-union test over every local extremum (179) gives 1 at p < 0.01
  and 4 at p < 0.05, against at most 1.8 and 9 expected under noise. Windowed with Bonferroni (`vswing_test.py`), it
  gives 1 flag against 0.07 expected: PackIdenticalLunches s16 S0 320:0/16 -> 640:6/16 -> 960:0/16. Those are coarse
  points 320 steps apart, and the main run's only events are open at 591 and close at 786. Probably genuine.
- **Calibration against the main run** (`calibration.py`).
  - From the same state, the main run's continuation is one more draw of the rollout process. Y = the main run finished
    stage k within L steps; censored if the episode ended first.
  - 980 points: mean V 0.554 vs main-run rate 0.566, difference +0.012 [-0.029, +0.055] (cluster bootstrap over
    (episode, stage)).
  - By V bin: 0.03 vs 0.07; 0.33 vs 0.39; 0.62 vs 0.61; 0.96 vs 0.96.
  - Coarse rows by task: KettleBoiling +0.038 [+0.002, +0.077], SteamInMicrowave +0.020 [-0.02, +0.06],
    PackIdenticalLunches +0.001 [-0.04, +0.04].
  - KettleBoiling `settled` +0.039 [+0.004, +0.076] is weak evidence: it is one of about 24 subgroup intervals. If it is
    real, it could come from the residual small base yaw (the 0.5-5 deg bins are +0.06 / +0.07).
  - Dense points are selected on V jumps and on main-run outcomes, so they are not a clean test.
- **Near stage completion.** At the 72 points where the main run finished stage k within <= 48 steps, mean V is
  0.97-1.00. The one exception is KettleBoiling s24 t480 (V 0.25): the 32 confirm rollouts agree (8/32), and it is the
  known release chunk.

### Moderate patterns, judged from main-run events and the confirm runs
- **KettleBoiling s17 S0**, 96:1/8 -> 112:6/8 (confirm 28/32) -> 128:0/8. The main run reaches the handle at 112, holds
  it at 118 and slips at 126. Genuine.
- **KettleBoiling s2 S0**, 64:0/8 (confirm 5/32) -> 80:9/16 (confirm 15/32) -> 96:1/8. Genuine approach dynamics.
- **CPU replays queued** (`compare_main.py`, output in `run_after.out` / `run_after2.out`):
  - PackIdenticalLunches s16 S0;
  - SteamInMicrowave s8 S2: V oscillates 0.38-1.0 while the main run opens and closes the gripper at the door 6 times
    between 926 and 1038. Its t1072 point is v4-affected (see below);
  - PackIdenticalLunches s18 S3: [1504, 1520) jumps +0.88. Both gripper rules agree there, so there is no version
    effect;
  - PackIdenticalLunches s5 S0: the located rise [352, 368) of +0.62 (0/16 at 320, 1/8 at 352, 6/8 at 368, 7/8 at 384)
    comes while the main run shows no object or gripper event between 116 and 403. Check this one first.

### Finding A (bug, current code): the `settled-v4` gripper rule
**Mechanism.** The ramp `PandaGripper.current_action` moves 0.2 per step, and through the GRIP controller it acts as
a finger position target. v4 says closing if gap < 80 - 1.5 mm and the gap did not widen by > 0.5 mm in the last
step. A free gripper in its last, slow phase of opening (+0.02 to +0.3 mm/step at 77-78.5 mm) is therefore restored
with the ramp at "closed".
- In all rollouts at SteamInMicrowave s9 t816, s12 t16, s29 t976, s8 t1072 and PackIdenticalLunches s14 t16, s27 t16:
  - `gripper_close` = 0 (open) for the whole first chunk;
  - yet the gap still goes 78 -> 63 -> 45 -> 36 -> 34 mm, then 51.7 at step 8 and 66.5 at step 10;
  - the trajectory is identical in every scene, so it is controller dynamics, not the policy;
  - over the same steps the main run keeps opening: 78.2 -> 78.7 mm.
- SteamInMicrowave s8 t880 is ambiguous: there the main run's fingers were closing at -1.7 mm/step.
- **Reproduce:** `np.load('.../run_dense2_SteamInMicrowave/roll/SteamInMicrowave_s9_single_t816_c0_m0.npz')`; the gap is
  `|states[:,12] - states[:,13]|`.

**Scope** (`grip_rule.py`, plus a classification over all 1171 points): 42 points lie in the window where the old
and the v4 rule disagree.

| state at t | points | right rule |
|---|---|---|
| opening slowly | 26 | old right, v4 wrong |
| actively closing (< -0.5 mm/step) | 7 | v4 right, old wrong: pre-v4 rows KettleBoiling s5 t80, s12 t112, SteamInMicrowave s23 t480, s12 t128, s20 t96, s21 t832; v4 row SteamInMicrowave s8 t880 |
| static, touching an object | 5 | v4 right for SteamInMicrowave s4 t144/160/176/224 (holding the vegetable); old right for KettleBoiling s26 t112 |
| static, free | 4 | ambiguous |

**KettleBoiling s26 t112.** The fingers had just touched the handle while open. Rollouts c0 and c4, with a close
command, reproduce the main run's closure step for step (78.4 -> 76.8 -> 72.8 -> 67.1). So the located critical chunk
[96, 112) of s26 stands.

**V impact so far: no visible anomaly.** SteamInMicrowave s12 t16 is 1/8, the same as the v3 point at t32 (1/8).
PackIdenticalLunches s27 t16 is 1/8, the same as at t32 (1/8). The located drop SteamInMicrowave s29 S2 [976, 992)
(0.62 -> 0) has its t976 endpoint on an affected point.

**Fix.** When the fingers are free, set the ramp to the value whose target equals the current finger opening. Use +1
only when they are blocked on an object (contact). Alternatively, use the finger qvel in the saved state with a small
threshold. Validate any rule on the 42 window points against the main run's continuation, then re-collect the 7
dense2 points.

### Finding B: renders are not deterministic on the local GPUs
`obs_fp` (the fingerprint of each rollout's first request) should be identical within a point. Points with more than
one distinct `obs_fp`:
- KettleBoiling run_v2 (nano4): 0/249;
- SteamInMicrowave_v2: 5/105; SteamInMicrowave fix3: 123/153;
- PackIdenticalLunches coarse: 107/110; PackIdenticalLunches fix3: 105/126;
- dense SteamInMicrowave / PackIdenticalLunches / KettleBoiling: 44/66, 234/246, 20/45;
- dense2 SteamInMicrowave / PackIdenticalLunches: 10/18, 26/28.

The physics and the restore are deterministic: start states differ by 0 across rollouts, and the states over the
shared first chunk (m0 vs m1) are bit-identical (6 fix3 points checked). So the difference is in the images.
- Same state and seed in two runs: the first chunk is identical for 41-149 of 200 candidates (pool runs), 35/40 for
  dense vs confirm, and 0/56 for nano4 vs this machine.
- The median max |delta action| is 0.004-0.008, sometimes a binary gripper or mode flip.

This does not bias V, but the README's "the candidates' first chunks are the same as before" does not hold for most
candidates. Corrupted frames are not yet ruled out; `frame_scan.py` and `render_check.py` are queued.

## 2. Labels
- **k:** k == k_replay == sum(tk <= t over the main run's history) for all 1170 current rows. Superseded rows also
  have k == k_replay.
- **Re-evaluation** (`label_check.py`, state replay without rendering, evaluated as `rollout()` does):
  - `load_scene` (fingerprint checked), `collect.replay` for S and k_replay, `set_py_state` with the row's py_state;
  - `_check_success` at every step; at chunk boundaries the stage loop and the stop rule; labels as `collect.py`
    computes them;
  - sample (`sample.json`, `make_sample.py`): 229 rollouts at 210 points, task x stage x done with 10 each, half coarse
    and half dense, plus 29 edge cases (transient success, success at a non-last stage, reached = k+2, prev_ok False).
- **Update at 05:05:** 151/229 rollouts at 139 points. KettleBoiling and PackIdenticalLunches are complete; for
  SteamInMicrowave, S0 is done and S1-S3 are pending.
  - 148 match on every label; 79 of the 85 done rollouts hold the condition at the recorded boundary, and 7 are done
    through a transient success.
  - 3 PackIdenticalLunches rollouts differ, each on a contact predicate right at its threshold. Online, the checks see
    contacts and kinematics from before the last substep's integration (lite_physics); a replay of the saved state
    evaluates them one 2 ms substep later. Not code bugs, but the third shows that "success at any step" can hinge on a
    single substep:

    | rollout | field | stored | replay | probable reason |
    |---|---|---|---|---|
    | s0 t2240 S2 c2m1 (fix3) | progress | 0.512 | 0.272 | one object's contact-based "held" flag flips |
    | s9 t1280 S3 c7m1 (v1) | reached, progress | 3, 0.834 | 4, 1.0 | `_check_success` true at the boundary in replay, not online; done, success and steps still match |
    | s6 t1152 S3 c2m0 (dense) | done, success | False, ran 496 steps | success at the single step 357, stops at 368 | same as above |

  - `resim.py` re-simulates from the saved actions and evaluates exactly as online would, e.g.
    `$PY resim.py run_dense_PackIdenticalLunches PackIdenticalLunches_s6_single 1152 2 0`.
  - PackIdenticalLunches impossible-assignment points: 33 of 298 (`pil_assign.json`, now finished; episodes s1, s2, s6,
    s17, s19, s20, s29).
- **Results at the first handback:** 94/229 rollouts at 87 points. Covered: all KettleBoiling cells, PackIdenticalLunches S0 and
  S1, 3 of PackIdenticalLunches S2, and the edge cases among them. SteamInMicrowave is pending.
  - 0 mismatches in done, prev_ok, steps, success, reached, progress (+-0.0015) and stopping step.
  - The start state equals the main run's state at t in every case. The py_state rebuilt by the replay and k_replay are
    identical.
  - Of the 54 done rollouts, 50 hold the stage condition at the recorded boundary; the other 4 are transient successes.
  - No not-done rollout has the condition at any boundary, or even between boundaries.
  - Finish with `$PY summarize_labels.py`.
- **prev_ok at the end state, all 309 SteamInMicrowave prev_ok=False rollouts** (`prev_ok_causes.py`): 304 are
  explained. Five evaluate True at the saved end state:
  - S1: s13 t320 c4m1, s22 t1280 c4m1, s23 t480 c0m0;
  - S2: s14 t1280 c1m0 and c4m0.
- **Likely cause of the five:** with `lite_physics` the online checks see contacts and kinematics from before the last
  substep's integration, while `set_state` + forward evaluates one 2 ms substep later. These are threshold flips that
  only touch prev_ok; they are not a code bug.
  - `resim.py` (queued for s14 t1280 c1m0) re-simulates from the saved actions with the row's settle version and
    evaluates exactly as online.
  - The same one-substep difference applies to `collect.replay`, but no k mismatch resulted.

## 3. Edge semantics
1. **Off-by-one at chunk boundaries: none.** None of the 1170 points has stage k's order-free condition true at t, or
   in t-15..t. 27 points complete within their first chunk, with V 1.00.
2. **The last stage uses success at any step**, from the gym reward `_check_success` after each step. The other stages
   need the condition at a chunk boundary. 39 current rollouts are done only through a transient success:
   KettleBoiling S1 20, SteamInMicrowave S3 16, PackIdenticalLunches S3 3.
   - **KettleBoiling s13 t320 (14/16).** The gripper turns the knob at 0.250-0.252 m from the kettle. `gripper_obj_far`
     (> 0.25 m) holds for steps 5-7 while the burner turns on; at the boundary the distance is 0.242-0.249 m. The main
     run's own "success" at 336 is the same kind: its history has no S1 entry.
   - **SteamInMicrowave s5 t1280 (6/16).** The fingers touch the start and stop buttons together. `_turned_on` is on
     for one step and then off; the labels are done=True, success=True, progress=0.0.
   - **Reproduce:** `$PY transient.py run_v2/s3 KettleBoiling_s13_single 320 0 0 1 0` and
     `$PY transient.py run_SteamInMicrowave_fix3 SteamInMicrowave_s5_single 1280`.
   - For the last stage, reached > k always comes with success (616/616, 238/238, 133/133).
   - Success at a non-last stage: 56 KettleBoiling S0 rollouts reached 2 because the burner was already on (e.g. s12,
     where the main run finished both stages at 384), plus 6 in SteamInMicrowave S2. Both are consistent with the
     tracker.
3. **prev_ok** only matters for SteamInMicrowave: KettleBoiling S1 is the last stage, and in PackIdenticalLunches the
   earlier checks follow from the count.
   - S1: 1287 done, 168 with prev_ok False. 91 had the vegetable already out at t in the main run (13 points), 74 lost
     it during the rollout (23 points), and 3 are unexplained.
   - S2: 589 done, 141 with prev_ok False. 123 fail only on the S1 check's transient "gripper > 0.25 m from the bowl"
     (32 points), 16 had the vegetable already out (s28 t960), and 2 are unexplained.
4. **Earlier stages are never re-checked.**
   - **SteamInMicrowave:** at 28 points the vegetable is already out of the bowl at t (S1 20, S2 2, S3 6).
     - S1 V counts carrying an empty bowl: s6 t480 and t640 have V 1.00 against V(done & prev_ok) 0.00; s28 t800 1.00
       vs 0.00; s16 t1280 0.81 vs 0.00.
     - S3 V is 0 by construction (s6 t800-1280, s28 t1120-1280).
   - **KettleBoiling:** 6 points, s24 t560-960 S1. The kettle is 0.162 m from the nearest burner (the limit is 0.15),
     and V is 0-0.12.
   - **PackIdenticalLunches** (`pil_assign.py`; 272 of 298 points done). The S0-S2 checks count any object in any
     tupperware, while S3 needs one vegetable and one meat in each.
     - 28 points can no longer succeed without moving an object already placed: s1 t1280-2560, s2 t640-2560, s17
       t1920, s19 t1600-2560, s20 t1280-2560, s29 t960-2560.
     - At the S2 points among them, V is high anyway: s2 t640 1.00, s20 t1280 0.94, s29 t960 0.75, s29 t1280 1.00.
       All their S3 points have V = 0.
   - The microwave `_turned_on` is True in py_state at SteamInMicrowave s20 t1280, s25 t960/1120 and s28 t1280. The
     button was pressed with the door less than 90 % open, which RoboCasa allows. V is 0 there for other reasons.

## Recommended actions
1. Fix the v4 gripper ramp rule as in Finding A before collecting more data. Re-collect SteamInMicrowave s8 t880, s8
   t1072, s9 t816, s12 t16, s29 t976 and PackIdenticalLunches s14 t16, s27 t16.
2. Decide what "done" means for the last stage: check it at chunk boundaries like the other stages, or store a
   `done_by` field (boundary or success step).
3. Compute prev_ok from the lasting conditions only: vegetable in the bowl, bowl inside the microwave, kettle on the
   burner. Do not use done & prev_ok for SteamInMicrowave S2 as it is now.
4. Mark the points where the main run already broke an earlier stage, or where the PackIdenticalLunches assignment is
   impossible (lists above). Either exclude them from the critical-chunk search, or define V as P(stage k done and the
   earlier stages still hold).
5. Store per-key fingerprints, or the first request, for every rollout. Render one state twice on each GPU to measure
   the image difference, and finish the frame scan.
6. Optional: re-collect about 20 random KettleBoiling points with the current settle to test the weak calibration
   offset.

## Files and jobs
- **Scripts:** `common.py`, `vseries.py`, `positive_control.py`, `vswing_test.py`, `calibration.py`, `grip_rule.py`,
  `grip_against.py`, `transient.py`, `prev_ok_causes.py`, `prev_ok_mismatch.py`, `pil_assign.py`, `make_sample.py`,
  `label_check.py`, `summarize_labels.py`, `resim.py`, `compare_main.py`, `frame_scan.py`, `render_check.py`,
  `run_after.sh`, `run_after2.sh`.
- **Outputs:** `vseries.json`, `flags.json`, `swing_test.json`, `calibration.json`, `grip_rule.json`,
  `prev_ok_causes.json`, `sample.json`, `sample_why.json`, `label_check.jsonl` (still growing), `pil_assign.out`.
- **Still running at handback** (detached, one CPU slot each): `label_check.py` (2 workers) -> `label_check.jsonl`;
  `pil_assign.py` -> `pil_assign.json`; `run_after.sh` -> `prev_ok_mismatch`, `grip_against`, `resim` x3, `compare_main`
  x4, `frame_scan` (output in `run_after.out`); `run_after2.sh` -> `compare_main` for PackIdenticalLunches s5 and
  `render_check` (output in `run_after2.out`).
- Afterwards, re-run `vseries.py`, `vswing_test.py`, `calibration.py` and `summarize_labels.py`.
