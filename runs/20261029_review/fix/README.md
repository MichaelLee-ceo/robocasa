# The settled-v5 restore: fixes from the 2026-10-29 review, their validation, and the re-collection

> **Paths below are from before the 2026-10-29 reorganization.** The data now lives in
> `../../20261026_critic_data/data/<task>/<dataset>`. The superseded rows these checks compared against were deleted
> after the comparison. The scripts here that read `run_*` dirs document what was run; they do not re-run on the new
> layout.

The five review reports (`../*/report.md`) found these problems in the restore. All are fixed in code; collected rows
carry `hidden="settled-v5"`.

1. **Gripper ramp restored as +-1** (bug 5): restore F1-F3, integrity A/B/B3, anomaly cluster 1, labels finding A.
   - The true `PandaGripper.current_action` ramp lies on a 0.2 grid. At t = 0 it is exactly 0, and during grasp and
     release starts it takes values in between.
   - The v4 threshold read the slow end of an opening (77-78.5 mm) as closing.
   - v3 misread fingers squeezed against a fixture, and held objects that push the fingers apart.
2. **Observation clock** (restore F4): every `rollout()` start advanced the observables' sampling clock by 2 ms, so
   rollout r sampled r substeps early.
3. **Toaster dict state** (integrity C): `py_state` copied scalar attributes only.
4. **Seeds** (analysis B1): the continuation seeds were the same in every run.

## Code changes
- **`runs/scripts/hidden_state.py`**
  - `gripper_ramp()`:
    - held object: +1 unless within 1.5 mm of fully open;
    - an object and a fixture, not held: the old +-1 rule;
    - fixture only, fingers still: the contact side;
    - otherwise (free fingers, one object touched, or fingers moving): the servo target,
      `c = 1 - (gap + 2 * dgap) / 40 mm`, rounded to the 0.2 grid.
  - `finger_contacts()` was moved here from `../restore/settle_candidate.py`.
  - `capture` / `restore` now include each observable's clock (`_time_since_last_sample`, `_sampled`,
    `_current_delay`).
- **`runs/20261026_critic_data/collect.py`**
  - At t = 0 there is no settle: the hidden state right after `load_scene` is the main run's.
  - New row fields:
    - `restore` (ramp, gap, dgap, contacts at t) and `code` (hashes of the restore code);
    - `salt`, appended to the continuation seeds (ci >= 1);
    - per rollout: `bounds` (per chunk boundary: t, a bitmask of the stages whose check holds, the success steps in
      the chunk, success at the boundary step) and `obs_fp_keys`.
  - Live asserts: every rollout must start at the main run's state at t, and the continuations of a candidate must
    share their first chunk's states bit for bit.
  - A job's `out` field overrides `--out`, and the env is rebuilt when the task changes, so one queue can fill
    several dirs.
- **`runs/scripts/atomic_eval.py`**: `rollout(..., boundary_hook=None)`, called at each chunk boundary before the stage
  checks.
- **`runs/20261026_imagine/imagine.py`**: `py_state` deep-copies plain containers. `set_py_state` restores dict-key
  types after a JSON round trip.
- **`../restore/harness.py`**: emulates the v4 gripper rule for `settled-v4` rows and adds v5, so the review tools
  still replay old rows exactly.

## Validation (this folder; `run_validate.sh`, outputs in `out/`)
| check | result |
|---|---|
| Rule vs true ramp, offline: 77.5k boundaries of saved rollouts (`../restore/out/grip_table.npz`) | Mean error 0.09 ramp steps vs 0.62 for v4; mid-ramp 0.8 vs 4.6; wrong side at saturated boundaries 0.5 % vs 4.5 %. |
| In env vs offline rule (`validate_v5.py`, 712 stratified boundaries) | Same value at 708. The 4 others are contact-classification edge cases. |
| Exact replay of old rows (`../restore/exact_replay.py`) | Bit-identical, 11 cases x 2 runs, v1-v4, all 3 tasks. One SteamInMicrowave scene failed to initialise, a known flaky `_load_model` error. |
| Continuous vs restore, 29 boundaries (`cont_vs_restore.py`, `cvr_compare.py`) | Worst group after 16 steps: median 1.3 vs 4.3 for v4, max 70 vs 567. See the notes below. |
| Observation clock, 6 rollouts in sequence (`obs_clock_check.py`) | Phase (0.004, sampled) at every boundary of every rollout, equal to the main run's. Requests identical. |
| Pilot through `pool.sh` / `collect.py` (`check_pilot.py`, `../../20261026_critic_data/pilot_v5/`) | No assert fired, and `bounds` are logged. KettleBoiling s17 t112 restored at -0.6: the main run's fingers are inside the candidates' range at every step, against 10-22 mm outside before. SteamInMicrowave s12 t16 (the v4 band): 44 mm outside -> 0. PackIdenticalLunches s15 t64 (the toaster case): no leak. |

Continuous vs restore, details:
- **Ramp differs from v4 (10 boundaries):** v5 is better at 9 and exact at 4. At KettleBoiling s5 t0 b304 the object
  error is 1.3 mm against 567 mm with v4.
- **Same ramp (19 boundaries):** the errors are the base-mode arm lag (10-60 mrad), identical for v4 and v5.

Two findings explain residual differences. Neither changes the collection.
- **Warm start in the harness.** `settle` zeroes `qacc_warmstart`, but the controllers' `update()` then calls
  `sim.forward()`, which refills it from the leftover `ctrl`. In the harness several captures follow each other at one
  point, so their warm starts differ and loose objects diverge chaotically (`hdiff.py`: of 59 differing fields only
  `warmstart` changes the trajectory). In `collect.py` the single capture per point comes right after `load_scene`, so
  it is deterministic and shared by all rollouts, and the live assert checks that.
- **Policy, not restore.** When the main run lies outside the candidates' range, it can be the policy rather than the
  restore. At PackIdenticalLunches s0 t352 all candidates command open from ramp 0, as the main run's fingers imply, but
  the main run opened more slowly: its own commands differed.

**Live QC finding (07:50): fingers in contact with a fixture and moving.** This branch of `gripper_ramp` extrapolates
the servo from the past motion, but contact slows the fingers. At PackIdenticalLunches s2 t2240 the restore gives +0.2,
and the main run's later closing implies about +0.5.
- On the 827 ground-truth boundaries of this kind (`fixture_future.py`) no gap-based estimator is good:
  - past motion (v5): mean error 2.0 ramp steps, >= 2 steps at 54 %;
  - future motion: 2.0 / 54 %;
  - the old +-1 rule: 4.9 steps, wrong side at 50 %.
- So v5 is still the best of these.
- Ten of the 1,587 rows use this branch:
  - KettleBoiling s0 t400, s21 t400, s24 t640, s8 t960;
  - SteamInMicrowave s20 t96, s21 t832, s8 t880 / t992 / t1024;
  - PackIdenticalLunches s2 t2240 (V = 0 there, old and new).
- They are marked in each row (`restore.fixture` and `|restore.dgap_mm| > 0.5`). After phase 1 they get a sensitivity
  check: collect again with the future-motion ramp and see whether V moves.

Known residuals that no restore from saved states can fix:
- the arm's goal lag when a chunk starts in base mode;
- an opening ramp that starts while an object is held (it does not show in past states);
- the warm start.

Future main runs should save `capture()` and the actions at every chunk boundary.

## Phase-1 result and the gate (2026-10-29 06:30-09:55)
- **Collection:** 244 of 244 points, 0 errors, no retries (`watchdog_v5.sh`).
- **Live QC** (`qc_v5.py`): 8 flags, each investigated in `../../20261026_critic_data/run_v5_queue/flags_reviewed.md`:
  - 4 are policy or contact cases;
  - 3 are door-stage points where the old row was the bug-5 clamp (V 1.0 -> 0.5 / 0.38 / 0.75);
  - 1 is a fixture-moving row whose outcome is unaffected (V 0 both ways).
- **Paired old vs new** (`paired_v5.py`):
  - Main run's fingers > 3 mm outside the candidates' range within 4 steps: KettleBoiling 51 % -> 1 %,
    PackIdenticalLunches 43 % -> 2 %, SteamInMicrowave 77 % -> 4 %.
  - Mean |dV| 0.05-0.08, mean dV about 0. 16 points moved by >= 0.25: the door stage, t = 0 points both ways, and
    KettleBoiling s17 t112 0.75 -> 0.50.
- **Gate** (`../gate_v5/`: `scan.py`, `ramp.py`, `analyze.py`, `tables.py` -> `tables.md`; the anomaly scan with the
  `_fix5` dirs):
  - **Fingers:** 6 of the 244 v5 points (2.5 %) are > 3 mm outside within 4 steps, the 6 already investigated. Of the
    old rows still in use, only PackIdenticalLunches s7 t1120 is, marginally (3.1 mm at step 4, 0 at step 1).
  - **Exchangeability:** the main run is the most outlying of 9 at the expected 11 %. KettleBoiling coarse 10-12 %;
    SteamInMicrowave / PackIdenticalLunches coarse 9-13 %, eef / obj 15-17 % (base-mode lag). Replaced rows: 26-100 %
    (the positive control).
  - **Elevated old dense rows still in use** (KettleBoiling grip 24 %, SteamInMicrowave early 19 %): every such point
    has the main run's fingers inside the candidates' range at steps 1-4. These are degenerate ranks: all 8 candidates
    identical, fingers fully open or closed, so any sub-mm difference ranks the main run 9th. They are not restore
    errors.
  - **Absolute physics flags** (drops, speeds, base direction): the current rows are at the main run's own rate.

## Re-collection (phase 1)
`affected.py` (all rows in use) -> `select_v5.py` -> `jobs_v5.json`: 244 points.

Selection rules:
- every t = 0 point;
- every point whose restored ramp changes;
- `settled` / `settled-basefix` rows whose base had turned >= 2 degrees since the episode start (residual of bug 3).

| task | coarse | dense | dense2 |
|---|---|---|---|
| KettleBoiling | 70 / 250 | 4 / 45 | - |
| SteamInMicrowave | 50 / 262 | 14 / 66 | 6 / 18 |
| PackIdenticalLunches | 89 / 256 | 9 / 246 | 2 / 28 |

- Output: `run_<task>_fix5`, `run_dense_<task>_fix5` and `run_dense2_<task>_fix5`, overriding the rows of the same
  (tag, t). Seeds are the original ones (salt ""), so old and new rows pair up.
- Phase 2 covers the 7 affected confirm points and new confirmations with a salt. It runs after the coarse and dense
  data are re-analysed.
