# Physics-anomaly scan of the critic data (review-anomaly, 2026-10-29)

Written by the review-anomaly agent and saved here by the main session, because the agent could not write .md files.

## Summary
- **Scope.** Every rollout of every collected point was scanned, not a sample: 90 episodes, 2,252 point instances and
  31,568 rollouts, covering all candidates and all continuations, current and superseded. Each rollout's first 32 steps
  were compared with the main run's next 32 steps (`ep_states[t:t+33]`). No rendering, 3 CPU workers, 35 min.
- **The known bugs are gone from the current rows.** All four show clearly in their positive controls (the superseded
  rows, `run_dense_*_bug3`, `run/` and `run_SteamInMicrowave/`), and none is left in the current rows. The current
  data's free-object, end-effector-speed and joint-velocity flags are at the level of the null (sibling candidates, or
  the main run's own windows).
- **New cluster ("bug 5"): the gripper command ramp.** `settle()` sets `PandaGripper.current_action` to +-1. It fails:
  - at t = 0, where the true value is 0;
  - during gripper transitions, where it is in between;
  - under the v4 rule (the bug-4 fix), which reads a fully open gripper at 77-78.5 mm as closing.
- **Size.** In 136 of 1,183 current point instances (11.5 %), the main run's finger gap lies more than 3 mm outside the
  range of all 8 candidates within 4 steps:
  - all 91 t = 0 points: the candidates close to about 24 mm and reopen, while the main run opens at once. Nothing is
    within reach, so this is transient;
  - 7 v4 points (`run_dense2_*`): the fingers snap shut by up to 44 mm while the policy commands open. This regression
    also affects the v4 collections now running;
  - 38 transitions, 13 of them touching an object within the chunk.
- **Confirmed by reproduction.** `repro.py` replays the saved actions without rendering and reproduces the saved
  rollouts with a state difference of 0. With the ramp read off the main run's own finger gap instead, the candidates'
  fingers follow the main run exactly, or to within 0.3 mm.
- **It changes labels at a confirmed critical point.** At KettleBoiling s17 t112, as collected, all 8 first chunks hold
  the kettle. With the main run's true ramp (-0.6), 2 slip within the chunk exactly as the main run did, and 2 more
  half-slip. The finding "the main run drew a rare bad chunk" at s17 t112 (`../../20261028_critical/README.md`) is
  largely a restore artifact.
- **Known approximation, now measured.** At points that start in base mode, the arm lag puts the end effector a median
  of 4-6 mm off at step 4 (90th percentile 14-26 mm). The candidates' own spread is 1.5 mm, so the main run is the
  outlier in 29 % of PackIdenticalLunches points and 50 % of SteamInMicrowave points, against 11 % expected.

## 1. What was checked
**Coverage** (`scan.py`, `analyze.py`):
- current coarse rows, in `effect_by_event.points()` order: KettleBoiling 250, SteamInMicrowave 262,
  PackIdenticalLunches 256;
- `run_dense_*`: 357; `run_dense2_*`, complete rows only (written 04:00): 46; `run_confirm_KettleBoiling`: 12;
- superseded rows: 378; `run_dense_*_bug3`: 179; the bug-1 runs `run/` and `run_SteamInMicrowave/`: 512.

The current data is 1,183 point instances and 15,896 rollouts, 9,464 of them continuation 0 (m0).

**How the states were read.** For every saved state: set the flattened state and run `mj_kinematics`. Every 4th
state, also run `mj_fwdPosition` for the gripper-object contacts. One env per task per worker, built with `replay_env`;
one `reset_to` per episode.

**Checks** (first 32 steps, vs the main run's next 32):

| check | rule |
|---|---|
| F1a, F1r, F1b, F1c (free objects) | moved > 10 cm while the main run's moved < 2 cm; fell > 10 cm while the main run's fell < 5 cm; fell > 10 cm; faster than 2 m/s |
| F2 (base direction) | both displacements > 5 cm and the angle between them > 90 deg |
| F3a, F3b (end effector) | speed relative to the base > 1 m/s; a single step > 2 cm and > 4 x the trajectory's median step |
| F4, F4e (fingers) | the gap opens > 5 mm within 8 steps while touching an object (touch from the main run's trace) and the main run's does not; F4e is the same at s0 = 0 only |
| F5 (contact explosion) | arm joint > 3 rad/s, finger > 0.5 m/s, fixture joint > 20, or object angular velocity > 50 rad/s. The main runs' own maximum over all 8,256 chunk windows is 3.8 rad/s for the arm |

**How rates are judged.**
- Relative flags are computed against the main run, and against a sibling candidate's m0 (the null). Both are draws of
  the same policy from the same observation, so with a faithful restore the two rates are equal.
- Absolute flags are compared with the main run's own windows at the same points, and at every chunk boundary of the
  90 main runs.

**Extra tests.**
- **Exchangeability rank test** (`ranks.csv.gz`). At each point the main run and the 8 m0 candidates are ranked by
  leave-one-out outlyingness, per feature group: early steps 1-2, finger gap, end effector, base, held object, arm
  joint velocity. The main run should be the most outlying in 1/9 = 11.1 % of points.
- **Restored state.** `start_diff` is 0 for 31,567 of 31,568 rollouts. The exception is in an invalid dir (section 6).
- **Shared first chunk.** Continuations m >= 1 reproduce m0's first 16 steps exactly in all 9,456 settled pairs
  (`pairs.py`). In the bug-1 runs, 4,087 of 4,096 pairs differ.
- **Base-command frame check** (`base_frame.py`). This needs none of the main run's actions: in base mode, the realized
  base displacement in the robot frame should follow the commanded `base_motion`.
- **First-chunk base direction** (`base_first8.py`), main run vs candidates over the first 8 steps.
- **Gripper-ramp diagnosis** (`ramp.py`, `classify.py`). The Panda fingers are position-controlled: target gap =
  0.04 (1 - r), with r = `current_action[1]` moved 0.2 per step.
- **Drill-down replays** (`repro.py`): the same scene, `collect.replay`, a choice of settle variants, and the saved
  actions stepped through the robosuite env.

## 2. Positive controls: the known bugs are detected

| bug | data | signal | current data |
|---|---|---|---|
| 1, controller state inherited | `run/`, `run_SteamInMicrowave/` | F4e 10.3 % vs 2.8 % null (KettleBoiling) and 18.7 vs 8.6 (SteamInMicrowave); F1r 14.6 vs 5.8; end effector > 1 m/s 8.6 % vs 0 %; m0/m1 first chunks differ in 4,087/4,096 pairs | F4e 0.1-0.4 % vs 0.2-0.7 %; 0/9,456 pairs differ |
| 2, arm lunge at a base-mode start | superseded `settled` rows starting in base mode (1,112 rollouts) | end effector > 1 m/s in 97.7 % (max 5.1 m/s); arm > 3 rad/s in 97.8 % (max 15.8); object falls (F1r) in 46.8 % | base-mode starts (1,618 rollouts): 0.0 %, 0.0 %, 0.1 %; max 0.39 m/s |
| 3, base frame turned | superseded and `run_dense_*_bug3` rows | command-vs-motion angle at turns of 90-180 deg: median 143 deg (SteamInMicrowave superseded), 163 deg (`_bug3`), 117 deg (PackIdenticalLunches superseded); F2 16.1 % vs 0.4 % in SteamInMicrowave `_bug3`; the first-chunk main-vs-candidate angle tracks the turned angle (corr -0.89) | median 8-9 deg, 0 % > 45 deg at turns of 90-180 deg; first-chunk angle 2-12 deg, at the leave-one-out null of 3-12 deg |
| 4, wide grasp read as opening | SteamInMicrowave s4 t160, v2 row | F1r in 16/16 rollouts | the v4 row: 0/16 |

`base_first8.py` shows no residue of bug 3. PackIdenticalLunches `_bug3` is weak on F2, because its turns are mostly
under 90 deg. The frame check picks it up there (36 deg, with a third of rollouts above 45 deg at turns of 30-90 deg).

## 3. Residual rates in the current data
Each cell is "vs the main run / vs the null"; the full tables are in `tables.md`.

| task | role | points | F1a | F1r | F2 | F4 | F4e | F1b rollouts / main runs | F3a | F5 |
|---|---|---|---|---|---|---|---|---|---|---|
| KettleBoiling | coarse | 250 | 0.0 / 0.1 | 0.0 / 0.0 | 0 / 0 | 0.8 / 1.1 | 0.4 / 0.7 | 3.6 / 3.6 | 0 / 0 | 0 / 0 |
| KettleBoiling | dense + confirm | 57 | 0.3 / 0.3 | 0 / 0 | 0 / 0 | 0.0 / 0.3 | 0 / 0 | 1.4-8.3 / 2.2-8.3 | 0 / 0 | 0 / 0 |
| SteamInMicrowave | coarse | 262 | 0.0 / 0.2 | 0.0 / 0.1 | 0 / 0 | 0.1 / 0.5 | 0.1 / 0.2 | 2.2 / 1.9 | 0 / 0 | 0 / 0 |
| SteamInMicrowave | dense | 66 | 0.2 / 0.2 | 0 / 0 | 0 / 0 | 0.4 / 0.2 | 0.8 / 0.6 | 1.1 / 1.5 | 0 / 0 | 0.4 / 0 |
| SteamInMicrowave | dense2 (v4) | 18 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **5.6 / 0.0** | 0 / 0 | 0 / 0 | 0 / 0 |
| PackIdenticalLunches | coarse | 256 | 0.1 / 0.1 | 0.1 / 0.2 | **1.2 / 0.5** | 0.2 / 0.4 | 0.4 / 0.2 | 6.4 / 5.9 | 0 / 0 | 0.2 / 0 |
| PackIdenticalLunches | dense | 246 | 0.5 / 0.3 | 0.3 / 0.3 | 0.9 / 1.0 | 0.2 / 0.4 | 0.1 / 0.1 | 8.4 / 7.7 | 0 / 0 | 0.5 / 0.4 |
| PackIdenticalLunches | dense2 (v4) | 28 | 0.4 / 0.4 | 0 / 0 | **3.1 / 0.4** | 0 / 0 | 0 / 0 | 6.7 / 7.1 | 0 / 0 | 0 / 0 |

Every cell is at the null except these three:
- **SteamInMicrowave dense2 F4e (s1 t624).** Part of the gripper cluster (section 4).
- **PackIdenticalLunches F2 (7 points): policy behaviour, not a restore bug.** Within the first chunk the main run's
  base moves like the candidates', and the base follows its commands at every turned angle. The main run differs only
  in its own second chunk: at s15 t1280 it drives forward 20 cm like the candidates, then reverses 42 cm in steps
  16-32; at s16 t320 it goes 16 cm right like them, then 36 cm left.
- **F1b / F1c / F5 point clusters.** They occur at the same points in the main run's own window, e.g.
  PackIdenticalLunches s25 t320 and SteamInMicrowave s28 t640, where objects are already falling at t. The arm spikes
  above 3 rad/s are wrist motions in the middle of the rollout (steps 12-31) with no contact. The one early spike, at
  PackIdenticalLunches s21 t1760, is in the main run too (3.6 rad/s).

**The exchangeability test is the one clear signal.** The main run is the most outlying in 22-26 % of current points
for the gripper groups (early and grip), against 11 % expected. The ramp classes explain it fully:

| current points | early | grip | end effector | base |
|---|---|---|---|---|
| ramp mismatch classes A-E (205) | 60-78 % | 66-78 % | 12-17 % | 9-13 % |
| ramp-consistent F and G, arm-mode start (839) | 9-14 % | 11-15 % | 8-12 % | 10-12 % |
| ramp-consistent F and G, base-mode start (139) | 23-29 % | 11-18 % | **29 % (PackIdenticalLunches), 50 % (SteamInMicrowave)** | 10-15 % |

## 4. Cluster 1 (new): the gripper ramp is restored as +-1 (bug 5)
**Mechanism.** `PandaGripper.format_action` moves r = `current_action[1]` by 0.2 per step towards the commanded side.
The finger actuators are position servos with target gap = 0.04 (1 - r): r = +1 is closed, r = -1 is 80 mm. `settle`
sets r = +-1 from the gap. The true r is in between in these cases:
- **t = 0:** r = 0, because `GripperModel` starts with `current_action = 0`. The settle rule says close (+1).
- **Transitions:** the ramp is partway through its 10-step swing.
- **Static intermediate gaps:** free fingers at 39-75 mm in SteamInMicrowave stages 2 and 3.
- **The previous-state widening rule:** at KettleBoiling s19 t160 and SteamInMicrowave s1 t624 it fires on contact
  noise while an object is held.
- **v4:** "fully open" is the joint limit minus 1.5 mm (78.5 mm), but 6.8 % of the static open grippers at the
  collected points sit at 76.6-78.5 mm. In the v4 data, 7 of 27 open-gripper points are read as closing.

**Classes of the 1,183 current point instances** (`classify.py`, `ramp_classes.csv.gz`). "Outside" means the main
run's gap lies > 3 mm outside all 8 candidates at step 2 or 4.

| class | points | main run outside | size at step 4 |
|---|---|---|---|
| A, t = 0 | 91 (31 / 30 / 30) | 100 % | 34 mm: the candidates close 41 -> 24 mm and reopen; the main run opens at once |
| B, v4 reads a fully open gripper as closing | 7 (`run_dense2_*`) | 100 % | 35-44 mm: the gripper snaps shut while commanded open |
| C, gripper moving while touching | 19 | 37 % | up to 26 mm |
| D, gripper moving, free | 58 | 29 % | up to 44 mm |
| E, static intermediate gap, free | 30 | 43 % | up to 45 mm, mostly SteamInMicrowave k = 2-3 at the door or button |
| F and G, consistent | 978 | 0.1 % | |

Excluding t = 0, the rate is 45 of 1,092 points: KettleBoiling 0-9 %, PackIdenticalLunches 0-15 %, SteamInMicrowave
stage 2 8-38 % and stage 3 0-33 %, dense2 1-3 points per stage. Per-point list: `corrupted_points.csv`.

**Reproduction** (`repro.py`: it first reproduces the saved rollouts exactly, then changes only the ramp and replays
the same saved actions; outputs in `repro_out/`):

| point (dir) | settled (as collected) | ramp from the main run | main run |
|---|---|---|---|
| KettleBoiling s0 t0 (`run_v2/s0`) | +1: gap 41 -> 30 -> 24 -> 24 -> 26 | 0: 41 -> 43 -> 47 -> 53 -> 60 | 41 -> 43 -> 47 -> 53 -> 60, identical |
| SteamInMicrowave s12 t16 (dense2, v4) | +1: 78 -> 63 -> 46 -> 36 -> 35 | v3 rule, -1: 78.2 -> 78.3 -> ... -> 79.1 | identical |
| KettleBoiling s19 t160 (`run_v2/s0`) | -1: c1 24 -> 39 -> 54 -> 64 -> 70 | +1: 24.1 -> 24.4 -> 24.6 -> 25.0 -> 27.3 -> 37.2 -> 51.0 | 24.1 -> 24.4 -> 24.6 -> 25.1 -> 27.6 -> 37.3 -> 51.1 |
| SteamInMicrowave s8 t800 (fix3) | +1: 60 -> 49 -> 37 -> 31 -> 31 | -0.6: 59.0 -> 59.1 -> 59.3 -> 59.5 -> 59.7 | 59.7 -> 59.7 -> 59.8 -> 59.8 -> 59.8 |
| KettleBoiling s17 t112 (confirm) | +1: 73 -> 58 -> 39 -> 28 -> 26 | -0.6: 73.4 -> 67.4 -> 60.5 -> 53.2 -> 45.5 | identical |

**Effect on labels.** The policy was not re-run; the effect is judged within each candidate's own first chunk, by
replaying its saved actions.
- **KettleBoiling s17 t112**, a confirmed critical point (`run_confirm_KettleBoiling`, `run_dense_KettleBoiling`,
  `repro_out/KB_s17_t112_confirm_all.txt`).
  - Under the settled ramp, all 8 candidates hold and lift the kettle by step 16.
  - Under the true ramp of -0.6:
    - c2 and c6 slip, as the main run did: gap 3-4 mm and kettle z 0.995 at step 16, against the main run's 5.8 mm and
      0.995;
    - c0 and c7 end half-closed at 14-16 mm;
    - c1, c3, c4 and c5 hold.
  - So V(t112) = 0.88 is biased upwards. The main run's slip is what 2-4 of 8 candidates do from its true state, not a
    rare chunk.
- **KettleBoiling s19 t160.** The main run was still holding the kettle (r = +1) and released on its own command.
  Settle said opening, so every candidate released at once, including those whose chunk commands "hold" (c0, c2, c5).
  All 16 are labelled done, because the release was already over the burner. With r = +1, the replayed c0 and c2 keep
  holding (kettle z 1.009-1.05).
- **SteamInMicrowave s1 t624** (dense2, v4). The bowl is held. Settle's -1 opens all 8 candidates to 62-68 mm while
  they command close, against 58-62 mm with -0.6 to -0.4. The bowl is not dropped, but it is held looser; V = 0.25.
- **PackIdenticalLunches s20 t1920.** All candidates release meat0 within 2 steps, against the main run's 8. Stage 3 is
  stuck either way, so no label changes.
- **No change within the chunk** at KettleBoiling s19 t96 (all hold) and PackIdenticalLunches s11 t512 and s21 t96 (all
  miss either way); only the timing of the grasp shifts, by 2-4 steps.

**Judgement: a bug,** a fifth restore error of the same family as bug 4. It is benign at t = 0 (no contact within the
chunk) and in the 25 free transitions. But it can change labels at grasp and release onsets, the very moments the
critical-chunk analysis targets. In the current v4 collections it is a regression introduced by the bug-4 fix.

## 5. Cluster 2 (known approximation): the arm lags in base mode
`settle` puts the OSC goal at the achieved pose. In the main run the arm still converges to its accumulated goal
("desired" mode), so its end effector continues where the candidates' stops (`eef_lag.py`, `eef_lag.csv.gz`).

| point start | main run vs candidate median, step 1 / 4 / 16 | candidates' own spread at step 4 |
|---|---|---|
| base mode, PackIdenticalLunches (105 points) | 0.9 / 3.9 / 9.8 mm (90th percentile 3.1 / 14.1 / 44) | 1.4 mm |
| base mode, SteamInMicrowave (34 points) | 1.3 / 6.3 / 8.9 mm (90th percentile 5.0 / 25.7 / 43) | 1.5 mm |
| arm mode, all tasks | 0.7 / 4.3-4.7 / 14-17 mm | 3.6-4.2 mm |

This is the approximation the README describes ("millimetres to centimetres, not in a wrong direction"), now measured.
It is not a new bug, but it could matter when the robot carries an object close to an obstacle.

## 6. Other observations
- PackIdenticalLunches F2: policy behaviour in the main run's second chunk (section 3).
- A start-state mismatch in an invalid dir: in `run_dense_PackIdenticalLunches_bug3`, s15 t64 c5 m0 starts with
  `toaster_main_group_sideR_lever_joint` 0.061 away from the main run, while the other 31,567 rollouts start exactly.
  Python fixture state (the toaster) can leak between rollouts and move a joint inside `set_state` -> `update_state`.
  No current rollout is affected.
- Bug-4-like "wide grasp read as opening" in the current old-rule rows: 25 class F points, e.g. SteamInMicrowave s4
  t192 and t224 in dense. In none of them does the main run lie outside the candidates' range, and no object falls.
  They are benign.

## 7. Points believed still perturbed (full list: `corrupted_points.csv`, 136 instances)
**Label may change.** The gripper is in transition and touches an object within the first chunk: 13 instances at 12
points.
- KettleBoiling: s17 t112 (dense and confirm, the critical point above); s19 t160 (run_v2); s19 t96 (dense); s14 t160
  (run_v2); s12 t112 (dense).
- SteamInMicrowave: s1 t624 (dense2); s20 t96 (dense).
- PackIdenticalLunches: s20 t1920 (coarse); s11 t512, s21 t96, s14 t96 (dense); s7 t1120 (dense, marginal at 3.1 mm).

**v4 regression:** SteamInMicrowave s12 t16, s29 t976, s8 t880, s8 t1072, s9 t816; PackIdenticalLunches s14 t16, s27
t16 (all dense2).

**Transient, likely benign:**
- all 91 t = 0 points;
- 25 free transitions, 15 of them in SteamInMicrowave stages 2-3: s8 t800-t960, s21 t864-t928, s28 t960, s3 t960, s5
  t960, s9 t800.

## 8. Recommended actions (no fix applied)
1. **Hold the v4 collections** (`run_dense2_*` and the SteamInMicrowave and PackIdenticalLunches confirm dirs) until the
   gripper rule is fixed. v4 reads 7 of 27 open grippers as closing.
2. **Fix the ramp in `settle`:**
   - at t = 0, set r = 0 (exact);
   - otherwise use the servo relation: with free fingers, r = 1 - gap/0.04, rounded to the 0.2 grid (validated above);
   - while touching an object, use +1 unless the gap has widened steadily for 2 steps or more;
   - never decide "open" from a gap threshold alone; use contact.

   Check the fix with `repro.py ramp=` variants and with the rank test: grip and early should fall to about 11 %.
3. **Re-collect, in this order:** the 13 label-relevant instances; the 7 dense2 points; then, cheaply in the same pass,
   the 91 t = 0 points and the 25 free transitions. Then re-check the s17 t112 conclusion in
   `../../20261028_critical/README.md` and the confirm V of 0.88. Also re-check the "losers close in the air" reading
   at s19 t80, which is near a gripper transition (KettleBoiling s19 t96 is class C).
4. **For future main runs, save each step's hidden state:** the gripper `current_action`, the OSC goal and the base
   start pose, together with the actions. A restore could then use `capture`/`restore` exactly, which also removes the
   base-mode arm-lag approximation of section 5.
5. **Keep this scan as a gate after every collection.** It takes about 35 min on 3 CPU workers and covers start-state
   equality, m0/m1 first-chunk equality, the base command-frame check, the rank test per group, and the end-effector,
   joint-velocity and object flags.

## 9. Reproduce
Run from `runs/20261029_review/anomaly`, with `ROBOCASA_ROOT=/home/michaellee/mclee/robocasa PIP_CONFIG_FILE=/dev/null
NO_ALBUMENTATIONS_UPDATE=1 MUJOCO_GL=egl` and `PY=/home/michaellee/mclee/robocasa/envs/groot/bin/python`.
```
$PY -I scan.py 3 4          # feat/<tag>.pkl, 35 min with 3 workers
$PY -I analyze.py 3         # rollouts.csv.gz, mainwin.csv.gz, ranks.csv.gz
$PY -I tables.py > tables.md
$PY -I ramp.py && $PY -I classify.py      # ramp.csv.gz, ramp_classes.csv.gz, corrupted_points.csv
$PY -I base_frame.py; $PY -I base_first8.py 8; $PY -I eef_lag.py; $PY -I pairs.py
# drill-downs (no rendering; outputs in repro_out/): ./run_repro.sh; ./run_repro2.sh, e.g.
OBJS=obj NSTEPS=24 $PY -I repro.py run_confirm_KettleBoiling KettleBoiling_s17_single 112 0,1,2,3,4,5,6,7 0 v3 ramp=-0.6
OBJS=obj $PY -I repro.py run_v2/s0 KettleBoiling_s19_single 160 0,1,2,4 0 settled ramp=1
OBJS=vegetable NSTEPS=16 $PY -I repro.py run_dense2_SteamInMicrowave SteamInMicrowave_s12_single 16 0,1 0 v4 v3
OBJS=obj NSTEPS=16 $PY -I repro.py run_v2/s0 KettleBoiling_s0_single 0 0,1 0 settled ramp=0
```

**Files.**
- Scripts: `scan.py`, `metrics.py`, `analyze.py`, `tables.py`, `ramp.py`, `classify.py`, `base_frame.py`,
  `base_first8.py`, `eef_lag.py`, `pairs.py`, `repro.py`, `run_repro*.sh`, and `explore.py` (the qpos layout and timing).
- Data: `feat/` (440 MB, per-episode features), plus the CSVs listed above.
