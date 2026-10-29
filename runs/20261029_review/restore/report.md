# Restore-fidelity review (review-restore, 2026-10-29)

Written by the review-restore agent and saved here by the main session, because the agent could not write .md files.
Scripts and raw outputs are in this folder (`out/`); nothing outside it was touched.

## Summary
**What was checked:**
- **Exact replays:** 15 cases over all 3 tasks and all 4 settle versions, each run twice. All are bit-identical,
  including a row collected on nano4.
- **Continuous replays:** 41 of 41 match their saved states exactly. Restoring the captured true hidden state gives 0.0
  in 47 of 47 cases, so capture/restore itself is exact.
- **Continuous vs set_state + settle:** 32 boundaries chosen from a scan of 77.5k. The robot state is off by more than
  1e-3 at 23 of them: 14 from the gripper side, 9 from the base-mode goal lag.
- **Attribute audit** at those boundaries.
- **Row checks:** 1463 rows (k_replay, instruction, `microwave._turned_on`); all correct.
- **Gripper-rule accuracy** against the exact ramp. That ramp was verified against `current_action` at 266 boundaries.

**Main findings:**
1. **Current-code bug: v4's 78.5 mm threshold reads the slow end of every opening (77-78.5 mm) as closing.**
   - It gets the side wrong at 79 % of those boundaries: 4.5 % of all boundaries, against 1.6 % for v3.
   - The restored fingers close by up to 44 mm. Effects seen: a lunch item moved 48 mm, the fingers closed on the bowl,
     and the kettle fell 16 steps early.
   - All 47 `settled-v4` rows are affected; 7 points are flagged.
2. **v3 is wrong in two situations.**
   - An open gripper squeezed against a fixture is read as closing: microwave door 79 %, also the fridge.
   - A held object that pushes the fingers apart is read as opening: the bowl slipped 23 mm.
3. **t = 0.** The true ramp is exactly 0 (confirmed). Settle sets +1, so every rollout at all 92 t = 0 points first
   closes its fingers 41 -> 24 mm. They are 6 mm off after 1 step.
4. **New: the observation clock.** The observables' sampling clock advances 2 ms per `rollout()` call and is never reset,
   so rollout r's later observations lag rollout 0's by r substeps. Measured: base_position +1.47 mm per rollout index
   (22 mm at r = 15). Replays are unaffected, and no outcome trend shows in the data.
5. **The base-mode goal lag (known) is larger than the README says:** arm 10-60 mrad after 16 steps, up to 0.27 rad/s
   at step 1, held items 0.3-2.6 cm off. It also shows in KettleBoiling chunks that start in base mode.
6. **Rendering is nondeterministic** (1 intensity level). First requests differ between rollouts at 964 locally
   collected points.

**Best gripper rule:** `settle_candidate.py` "g_final".
- Held object: closing iff gap < 78.5 mm. Fingers on a fixture at 5-75 mm: the contact side decides. Otherwise v3.
- Wrong side 0.6 % (KettleBoiling 0.6, SteamInMicrowave 1.7, PackIdenticalLunches 0.1), against 4.5 % now.
- Mean finger error over 16 steps: 4.6 mm, against 15.2 for v4, 14.9 for the trend rule (c) and 15.2 for the value rule.
- The in-env implementation agrees with the offline rule at 200/200 boundaries.
- Rules (c) and (d) are worse because they drop held objects (21 cm in one case).
- Mid-ramp states (4.1 % of boundaries) cannot be fixed with +-1.

**Fixes (not applied):**
- At t = 0, capture without settle.
- Switch to g_final, then re-collect the dense2 rows and the ~30 flagged points.
- Reset the observable clocks at the start of each rollout.
- Save `capture()` and the actions in future main runs.

## 0. Findings

| # | finding | verdict | affects |
|---|---|---|---|
| F1 | **v4 gripper rule (current `hidden_state.settle`).** An opening gripper creeps 77 -> 78.5 mm over about 10 steps, at < 0.5 mm/step, and v4 reads that creep as closing. Wrong at 79 % of boundaries in that band; 4.5 % of 74k saturated boundaries overall, vs 1.6 % for v3. | **bug**, introduced by the bug-4 fix | all `settled-v4` rows (47 points: dense2 + fix4) and all future collection; 7 points flagged |
| F2 | **v3 rule.** (a) Open fingers squeezed by a fixture read as closing: microwave door in SteamInMicrowave stage 2 (79 %), the PackIdenticalLunches fridge. (b) A held object that widens the gap by > 0.5 mm in one step read as opening. | **bug** | about 24 flagged points |
| F3 | **t = 0 ramp.** The ramp after `load_scene` is exactly 0. `settle(None)` sets +1, and every t = 0 rollout closes its fingers 41 -> 24 mm while the main run opens them. | **bug** | first chunk of 92 points |
| F4 | **Observation clock.** The observables' `_time_since_last_sample` gains 2 ms per `rollout()` and is never reset, so rollout r gets its chunk >= 2 observations r substeps earlier. Even rollout 0's forced first request is 1-2 substeps fresher than what the main run saw (0.7 mm eef, 2.5 mm base). | **bug, minor:** policy inputs only, but correlated with candidate index | all rows |
| F5 | **Base-mode goal lag** (known approximation). | approximation, but large | base-mode points |
| F6 | **GPU rendering nondeterminism** (also seen in `20261025_determinism`). | noise | first chunks |
| F7 | **Stale OSC origin.** In the continuous run the OSC origin is 1 substep stale. While the base moves, the first arm-mode goal is about 1 mm off. | small approximation | |

## 1. Exact replay
`exact_replay.py`; settle v1-v4 are emulated in `harness.settle`. All 15 cases are bit-identical in two runs each.

**KettleBoiling** (`run_v2` was collected on nano4 and still replays exactly here):
- run_v2/s2 s0 t160 (v1)
- run_v2/s0 s0 t0 (v1)
- dense s12 t112 (v2)
- confirm s24 t480 (v3)

**SteamInMicrowave:**
- fix3 s13 t640 (v3, base-mode start)
- fix3 s4 t160 (v4)
- fix3 s27 t800 (v3)
- dense s11 t832 (v3, base turned 171 deg)
- dense2 s12 t16 (v4)
- dense2 s14 t1200 (v4, base turned 99 deg)

**PackIdenticalLunches:**
- fix3 s4 t320 (v3)
- fix3 s15 t2240 (v3, 577 states)
- dense s0 t352 (v3)
- dense2 s27 t16 (v4)
- run_PIL s0 t0 (v1)

**Code drift.** The only drift since collection is bug 4's threshold. v3 rows whose start gap is 75.17-78.5 mm diverge
from step 1 when replayed with the current settle: SteamInMicrowave s27 t800 by 0.196, PackIdenticalLunches s4 t320 by
0.195.

**Policy request vs `snap/req`.** Proprio is identical; images differ by at most 1 level.

## 2. Continuous vs set_state + settle
Source: `cont_vs_restore.py`, results in `out/cvr_table.md`.

Method:
- 32 boundaries, chosen from the scan (`scan_boundaries.py`).
- At each, the same saved actions ran 32 steps under three restores:
  - an exact restore of the captured true hidden state (always 0 difference);
  - v4 settle;
  - v3 settle.
- For t = 0, the comparison is the hidden state right after `load_scene` against the settled one.

Max |delta| after 1 / 16 steps.

**Gripper side wrong, or t = 0 (14 cases):**
- t = 0 on all 3 tasks: fingers 6.2 mm at step 1, 17 mm at step 4.
- Creep band, KettleBoiling s0 b16, s2 t400 b208 and SteamInMicrowave s20 t1280 b208: fingers 8 mm, back near 0 by
  step 16 (v3 gives 0).
- KettleBoiling s5 t0 b304 (creep band, at the kettle): arm 0.069; the kettle falls 16 steps early (0.72 m deviation).
- SteamInMicrowave s16 t800 b352: arm 0.023; the fingers close on the bowl.
- PackIdenticalLunches s14 t32 b48: arm 0.012; the item moves 48 mm.
- Closed but read as opening:
  - KettleBoiling s9 t320 b48 (knob): arm 0.003;
  - SteamInMicrowave s1 t544 b96: the bowl slips 23 mm and ends 12 mm lower;
  - PackIdenticalLunches s21 t1792 b64: arm 0.004.
- Fridge squeezing open fingers, PackIdenticalLunches s22 t1280 b544: arm 0.003.
- KettleBoiling s5 b192 (mid-ramp): fingers 4 mm.

**Base-mode goal lag (9 cases):**
- SteamInMicrowave s27 t480 b32 (yaw 72 deg): arm velocity 0.12 rad/s, arm position 0.042 rad, bowl 6 mm.
- SteamInMicrowave s1 b656 (yaw 174 deg): 0.19 rad/s, 0.062 rad.
- PackIdenticalLunches s15 t128 b112: 0.27 rad/s, 0.060 rad, item 26 mm.
- Also 0.010-0.024 rad: SteamInMicrowave s26 b32 (yaw 180 deg), s27 b96 and b144, KettleBoiling s29 t160 b144 (3 base
  steps), PackIdenticalLunches s21 t1408 b16 and s15 b80.

**Base and torso** stay <= 7e-4 in every case at yaw 0-180 deg, so bug 3 is fixed.

**Correct gripper side, arm mode: <= 3e-4 over 16 steps.** This covers holding the kettle, the big vegetable at
75.6 mm (v4), holding a lunch item, knob turning, the door, the button, an open free gripper and a closed empty one.

Loose objects can still diverge by up to 2 cm after 16-32 steps, because of the zeroed warm start (chaotic).

## 3. Attribute audit
All boundaries; the model physics are identical.

**Differences with an effect:**
- **Gripper `current_action`.** The true c is anywhere in [-1, 1]; settle sets +-1. `format_action` reads it. This
  explains F1-F3.
- **OSC `goal_pos` / `goal_ori`** in the base frame. In base mode the true goal lags the achieved pose, e.g. 7.8 mm and
  0.023 rad at SteamInMicrowave s27 b32. Read only in "desired" mode. This explains F5.
- **OSC `origin_pos` / `origin_ori`.** One substep stale in the continuous run, fresh after settle. `set_goal` reads them
  in "achieved" mode before `run_controller` refreshes them. Effect <= 1.3 mm (F7).
- **`qacc_warmstart`.** Known; it matters for loose objects only.
- **The observables' clock and cache.** This explains F4.

**Differences with no effect:**
- `_goal_update_mode`: reset before each `set_goal`.
- Cached kinematics, `new_update` True vs False: both resolve to values at the current state.
- Base, torso and gripper `goal_*` / `vels`, `_applied_action_dict`, `ctrl`: overwritten before use.
- `recent_*` buffers: read neither by the controllers nor by GR00T's inputs.
- `env.timestep`, `cur_time`: unused (grep). `done` stays False under `ignore_done=True`.
- Base `init_pos` / `init_ori` and OSC `initial_joint`: identical.

## 4. Other restore steps
- **Stage memory and instruction.** `k_replay == k`, and k matches the main-run history at t, in 1463/1463 rows. The
  instruction always matches. `microwave._turned_on` matches the trace, including the 8 rows where it is True.
- **MultiStepWrapper.** It reads only the last observation. Truncation at L counts from the rollout start, as intended.
- **F4, observation clock** (`obs_phase.py`, SteamInMicrowave fix3 s13 t640 c0, 18 rollouts in sequence):
  - the clock advances +0.002 per rollout, exactly;
  - the dynamics are identical in all runs;
  - the policy's `base_position` differs from rollout 0's by 1.47 mm x r, and `eef_rel` by 0.10 mm x r;
  - with 32 rollouts per point (the confirm runs) the lag reaches 48 ms at r = 23, then wraps.
- **Outcome vs rollout index** (within-point regression): no dir exceeds 2 SE, and the signs are mixed (9+ / 7-).
- **F6, rendering:** re-rendering the same state changes up to 1 level (`obs_phase` part A).

## 5. Gripper-side inference
**Ground truth.** c starts at the row's settle output and moves 0.2 per step toward each saved command. This matches
`env.robots[0].gripper["right"].current_action` exactly at 266 boundaries in 28 replays. The rules were scored on
77,533 boundaries.

**t = 0.** The true ramp is 0, confirmed in replays of all 3 tasks.

**The rules:**
- a = v3.
- b = v4.
- c = the trend over 8 steps, +-0.3 mm.
- d = the value rule.
- g_final:
  - held by both fingers: closing iff gap < 78.5 mm;
  - fixture-only contact at 5-75.2 mm: the contact side decides (pushed open means closing, pushed closed means
    opening);
  - otherwise v3.

**Wrong side at saturated boundaries:**

| situation | a | b (now) | c | d | g_final |
|---|---|---|---|---|---|
| all | 1.6 % | 4.5 % | 4.7 % | 6.1 % | **0.6 %** |
| creep band 75.2-78.5 mm | 0 | **78.7 %** | 13.4 % | 4.7 % | 0 |
| holding < 20 / 20-60 mm | 0.3 / 0.9 % | 0.3 / 0.9 % | 4.0 / 4.1 % | 3.4 / 4.3 % | 0 / 0 |
| holding >= 60 mm | **14.9 %** | 3.4 % | 11.3 % | 14.7 % | 3.1 % |
| fixture contact, mid gap | 36 % | 36 % | 39 % | 63 % | **16 %** |
| object touched, not held | 2.6 % | 9.6 % | 11.8 % | **68 %** | 2.6 % |

Per task, wrong side for b (now) -> g_final: KettleBoiling 3.9 % -> 0.6 %, SteamInMicrowave 8.6 % -> 1.7 %,
PackIdenticalLunches 2.6 % -> 0.1 %.

The steady mid gaps on the microwave are saturated ramps (573 open, 100 closed, 0 mid), not alternating commands.

**State error** (`ramp_restore.py`): 18 boundaries, restored with settle plus each rule's ramp, then run 16 steps.
Mean (max) finger-gap deviation, in mm:

| rule | mean (max) | note |
|---|---|---|
| true ramp | 0.04 | |
| a | 7.9 | |
| b | **15.2** (44) | |
| c | 14.9 | drops a held item 21 cm |
| d | 15.2 | drops a held item 21 cm |
| **g_final** | **4.6** | mid-ramp states only |

**Mid-ramp states** are 4.1 % of boundaries.
- The best +-1 choice still leaves 2.7 steps of error; the rules leave about 4.2.
- Rounding toward the commanded side costs < 1 mm; the other way costs 5-46 mm for up to 8 steps.
- Steady mid ramps are only 0.2 % of boundaries.

**Known cases:**
- KettleBoiling s26 t112 is a start-of-close mid-ramp state: a and g_final say opening; b, c and d say closing.
- PackIdenticalLunches s14 t16: only b gets it wrong.

**In-env check.** `validate_candidate.py`: the in-env `gripper_closing` and the offline rule agree at 200/200
boundaries.

## 6. Fixes (not applied) and what they invalidate
1. **t = 0: `H0 = capture()` without settle.** Re-collect the 92 t = 0 points (V(0)): 30 per task in run_v2,
   SteamInMicrowave_v2 and PIL, plus confirm KettleBoiling s29 and the fix PackIdenticalLunches t0 point.
2. **Use `settle_candidate.gripper_closing`.** Re-collect:
   - creep-band v4 points: SteamInMicrowave s12 t16, s9 t816, s29 t976, s8 t1072, s4 t176 (may be a real grasp),
     PackIdenticalLunches s14 t16, s27 t16 (or all of dense2);
   - SteamInMicrowave door points in use: fix3 s8 t800/t960, s25 t1120, s28 t960; dense s8 t832;
   - held-widen points: KettleBoiling s19 t160, SteamInMicrowave s21 t320, s1 t624, PackIdenticalLunches s20 t1920;
   - the 11 KettleBoiling knob points: check them with the contact test, as they are mostly real grasps.

   Full list: `out/points_at_risk.json`.
3. **Observation clock.** At each `rollout()` start, set every observable's `_time_since_last_sample = 0.002` and
   `_sampled = True`, or add both to `capture` / `restore`. No re-collection is needed unless candidates are compared by
   index.
4. **The base-mode lag and the residual gripper errors** cannot be fixed from the saved data. Store `capture()` and the
   actions at every chunk boundary in future main runs. Until then, flag the points that start in base mode.

## 7. Reproduce
Run from this folder after `source env.sh`, which sets GPU2 and `$PY`.

| what | command |
|---|---|
| exact replays | `$PY exact_replay.py cases_exact_SIM.json out/x.jsonl` |
| continuous vs restore | `$PY cont_vs_restore.py cases_cvr_SteamInMicrowave.json out/x.jsonl`, then `$PY cvr_summary.py out/x.jsonl --attrs` / `$PY cvr_table.py` |
| gripper rules, offline | `$PY grip_table.py && $PY grip_geo.py && $PY grip_eval_geo.py` |
| gripper rules, state error | `$PY ramp_restore.py cases_ramp_KettleBoiling.json out/x.jsonl && $PY ramp_summary.py` |
| observation clock | `$PY obs_phase.py run_SteamInMicrowave_fix3 SteamInMicrowave SteamInMicrowave_s13_single 640 0 18 out/o.json` |
| row checks / risk list | `$PY rows_check.py`, `$PY points_at_risk.py` |

The bowl slip, as a one-boundary case for `cont_vs_restore.py`:
`[{"dir":"run_dense_SteamInMicrowave","task":"SteamInMicrowave","tag":"SteamInMicrowave_s1_single","t":544,"c":6,"m":0,"boundaries":[96],"versions":["v4","v3"]}]`
