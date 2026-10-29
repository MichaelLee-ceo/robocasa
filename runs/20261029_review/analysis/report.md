# Review of the analysis and orchestration code (2026-10-29)

Written by the review-analysis agent and saved here by the main session, because the agent could not write .md files.
Scope: `runs/20261028_critical/*.py`, `runs/20261026_critic_data/{analyze_phase,make_jobs,collect}.py`, the `*.sh`
chains, and `runs/scripts/atomic_eval.py:rollout()`.
- Every check below is a read-only script in this folder, with its output beside it as `<name>.out`.
- Data as of 03:50-04:10 on 2026-10-29.

Severity:
- **A**: a reported or about-to-be-reported number is wrong.
- **B**: a number is biased now, or will be wrong on the README's required re-run.
- **C**: latent, or negligible today.

## A1. `analyze_phase.py` ignores the re-collected rows; the "clean" SteamInMicrowave/PackIdenticalLunches tables are not clean, and the natural glob picks the buggy rows
**The problem.**
- `analyze_phase.py` takes one glob, and later files win in `sorted()` order. It has no notion of `_fix` or `_fix3`.
- The README tables "Clean SteamInMicrowave results (run_SteamInMicrowave_v2)" and "Clean PackIdenticalLunches
  results (run_PackIdenticalLunches)" were computed from those dirs alone. So they include the rows that bugs 2 and 3
  invalidated: 32 + 153 in SteamInMicrowave, 65 + 126 in PackIdenticalLunches.
- A re-run with `"run_SteamInMicrowave*/results_w*.jsonl"` uses the v2 row for all 262 points, because `_fix` <
  `_fix3` < `_v2`. That glob also pulls in the bug-1 dir `run_SteamInMicrowave/`.
- For PackIdenticalLunches the sort order happens to be right.

Evidence: `check_analyze_phase.py`, and `analyze_phase_current.py` (the same estimator on `effect_by_event.points()`).

| | README | rows in use |
|---|---|---|
| SteamInMicrowave, all | +0.011, 15 pts [10, 19] | +0.0084 [+0.003, +0.014], 13 [8, 17] |
| SteamInMicrowave, S1 middle third | P(done) **0.25**, +0.041, 29 pts | P(done) **0.92**, +0.021 [+0.003, +0.044], 20 [7, 30] |
| SteamInMicrowave, S1 160-240 | 0.23, +0.036, 27 | 0.88, +0.018 [+0.001, +0.037], 19 [4, 27] |
| PackIdenticalLunches, all | +0.012, 15 [11, 19] | +0.0048 [+0.000, +0.010], **10 [0-2, 14]** |
| PackIdenticalLunches, S1 240-400 | +0.054, 33 [18, 44] | +0.016 [-0.006, +0.044], 18 [0, 30] |

What this changes:
- "The effect sits in mid-S1, where the main run was struggling (P(done) 0.25)" came from the base-turn bug.
- "About +14 to +15 in all three tasks" is now 14 / 13 / 10.
- The 20261028_critical tables are also pre-fix. Re-running `effect_by_event` moves bins by 3-11 pts
  (`effect_by_event_current.out`). For example, grasp "8+ chunks later" goes 16 -> 10 [0, 15], and PackIdenticalLunches
  "grasp within 48" goes 21 -> 10.

Fix:
- Load through `points()`, or an explicit ordered list RUNS -> FIX -> FIX3, never a sorted glob.
- Assert that every `jobs_fix*` job resolves to its dir (`check_override.py`; all pass today).
- Re-run every table before quoting it.

## A2. The pooled "best-of-8 gain" 142*sqrt(mean var_b) overstates the per-pick gain
The expected gain at a random point is mean_j(1.42 sigma_j), which is at most 1.42 sqrt(mean sigma_j^2) by Jensen's
inequality. var_b sits in a minority of points, so the pooled number is too high. The constant itself is right:
E[max of 8 N(0,1)] = 1.4246, and both variance estimators are unbiased in simulation. Evidence: `check_estimators.out`.

| task | pooled (as reported) | mean over stage x elapsed bins | mean over episodes | one-rollout critic, cross-validated |
|---|---|---|---|---|
| KettleBoiling | 14.2 | 12.6 | 11.6 | +1.5 |
| SteamInMicrowave | 13.0 | 10.6 | 10.6 | +1.2 |
| PackIdenticalLunches | 9.9 | 7.1 | 7.6 | +1.1 |
| all | 12.5 | 10.1 | 9.9 | +1.3 |

- The binned columns are still biased upwards, because Jensen applies within each bin too.
- KettleBoiling confirmation: 36 pooled vs 30 per point. The honest cross-validated value is 21.
- For bounded p the factor 1.42 depends on the shape. For p_i in {0, 1} it is 0.99 sigma at q = 0.5 and 1.57 sigma at
  q = 0.1.

Fix:
- Report the cross-validated pick gain. With M = 2, pick on continuation 0 and score on 1, then the reverse.
- If a perfect-picker number is wanted, compute it per point (e.g. Beta moment-match, then E[max of 8] - p) and average.
- At minimum, relabel the pooled number as "the gain at an average-variance point".

## A3. The gripper-ramp rule is wrong in the 75.2-78.5 mm band, in both versions (restore code, found while checking version mixing)
The two rules (`check_grip_band.out` compares each against the main run's gap trend over t-4..t):
- Old: closing iff the gap is below 75.2 mm.
- v4: closing iff the gap is below 78.5 mm, unless it widened by more than 0.5 mm in the last step.

**The old rule read closing fingers as "opening", at 9 points.**
- Coarse points: KettleBoiling s5 t80; SteamInMicrowave s23 t480, s20 t1280, s2 t800.
- Dense points: KettleBoiling s26 t112, s12 t112; SteamInMicrowave s20 t96, s12 t128, s21 t832.
- Example, KettleBoiling s26 t112: the gap goes 79.9 -> 78.4 -> 76.8 -> ... -> 8.6 mm in 12 steps, so the fingers are
  closing, but the restored ramp is fully open.
- That dense point has V = 0.00. It is the end of the headline chunk "s26 | 96-112 | 0.62 -> 0.00", which weakens "the
  critical chunk comes before the visible failure". The confirmation at t96 is fine, because both rules agree there.
- `grip_check.py` cannot see this: it only checks touching points, and only flags objects that fell >= 10 cm by step 8.

**v4 reads slowly opening fingers as "closing".**
- This happens at every episode's t16 (the gap creeps 77.4 -> 79.3 mm between t10 and t38), and after releases.
- Affected points:
  - dense2 SteamInMicrowave s29 t976, s9 t816, s12 t16;
  - dense2 PackIdenticalLunches s27 t16, s14 t16;
  - the confirmation jobs now running: SteamInMicrowave s7 t960, s29 t976, PackIdenticalLunches s14 t16.
- The ramp starts at full close, so the fingers are commanded closed for about 5 steps (it moves 0.2 per step).
- The located chunks SteamInMicrowave s29 [976, 992), s12 [0, 16), s9 [816, 832) and PackIdenticalLunches s14 [16, 32),
  s27 [0, 16) each have such a point at one end.
- In total, 15 coarse and 15 dense rows were collected with a ramp that the current rule would set differently
  (`check_grip_rule.out`).

Fix:
- Infer the side from the finger joint velocity sign in the saved state, or from a multi-step trend. Use the threshold
  only when the fingers are still.
- Re-collect the 3 confirmation points and the affected points listed above.
- Make `grip_check` judge "ramp vs main-run trend" at all points, not "fell".

## B1. Confirmation runs reuse the seeds of the rollouts that located them
- `seed = sha256(f"{tag}|critic|t{t}|c{i}|m{m if ci else 0}|{ci}")` depends neither on N or M nor on the run. So a
  confirmation at a dense point re-runs that point's continuation 0, and at a coarse point continuations 0 and 1.
- KettleBoiling dense-ta points: the first chunk is identical in 35 of 40, and the outcome is the same in 37 of 40
  (`check_seed_reuse.out`).
- SteamInMicrowave/PackIdenticalLunches: all 26 confirmation points reuse seeds, 22 at dense points and 4 at coarse
  points (`predict_confirm.out`). v3 and v4 differ only in the gripper band, so these are likely exact replays.
- Effect: V(ta) from a confirmation is not independent of the selection. Example, KettleBoiling s26: continuation 0
  gives V = 0.62, the new continuations give 0.46. var_b is slightly deflated, if anything. `cv_gain` is unaffected,
  because it scores on continuations 2-3.

Fix: keep the first-chunk seed (it is the candidate's identity) and salt the continuation seeds per run (e.g. m + 100),
or drop the reused columns from V.

## B2. `split_points.py`, `overlay_critical.py` and `drop_check.py` would pair new labels with superseded rollouts on re-run
- Labels come from `points()`, but the roll dir and `py_state` come from the RUNS rows only. `drop_check` globs only the
  RUNS roll dirs.
- The saved outputs predate the fix dirs, so they are consistent today, but they were computed on pre-fix data.
- On re-run (`check_latent_rolls.out`):
  - 44 of 122 split points (32 SteamInMicrowave, 12 PackIdenticalLunches) would use old rollouts, with 47 winner <->
    loser swaps;
  - 23 of 53 overlay points would be affected, with 29 swaps.

Fix: use `roll_dir()` and the `py_state` of the row in use.

## B3. `dense_intervals.json` is stale after the s4 t160 re-collection
- `dense_analysis` never recomputes the intervals. The two SteamInMicrowave s4 S0 intervals, [0, 160] and [160, 272],
  existed only because of the bug-4 row (V = 0). They now sit in `critical_chunks.json` (03:52) with jump 0 and share 0,
  against stale v0/v1.
- `after_refine.out` prints "18/18, median share 0.75"; the truth is 16 intervals and 0.80. 2 refinement points were
  spent on them.
- `share` is above 1 in 12 of 58 rows, because the series are non-monotone.
- The located jump is within 2 SE of noise in 18 of 58: KettleBoiling 5/12, SteamInMicrowave 7/18, PackIdenticalLunches
  6/28 (`check_jump_noise.out`). The coarse selection itself is fine: about 3 of the 50 jumps are expected from noise.
- 46 series mix v3 grid points with v4 refinement points. This matters only in the A3 band.

Fix: recompute the intervals with `make_dense_jobs.intervals()` and drop those whose ends no longer qualify. Compute
share from the series' own ends. Use a noise-aware jump test.

## B4. Confirmation thresholds differ across tasks
KettleBoiling confirmed all 12 intervals (|jump| >= 0.25), while `after_refine.sh` uses 0.4 for SteamInMicrowave (6
points) and PackIdenticalLunches (20). KettleBoiling's mean var_b over its 6 points at >= 0.4 is 0.108, against 0.064
over all 12. The "6 x the uniform points" claim depends on this.

## C. Low or latent
1. **Event windows are off by one.** The candidate chunk causes events at f in (t, t+16], but the code uses [t, t+16).
   The same applies to [t0, t1) in `value_curve` and `dense_analysis`. 10 points have an event at dt = 0 and 7 at
   dt = 16; bins move by at most 2 pts (`check_event_window.out`).
2. **`after_refine.sh` line 14** checks grep's exit status, so a `dense_analysis` crash would still let
   `make_confirm_jobs` run on the stale `critical_chunks.json`. The 02:56 version of that file contained the s4
   artefacts with |jump| = 1. Fix: `set -o pipefail`.
3. **Orchestration: no damage found.** No duplicate (tag, t) rows, partial lines or W collisions. No orphan claims
   outside `_bug3` and the running confirmations. All fix jobs resolved. The filesystem is local ext4, so O_EXCL and
   O_APPEND are atomic.
4. **Orchestration: latent hazards.**
   - A dead worker's claim is never cleaned if its PID is reused, so the point is silently skipped.
   - `pool.sh` exits on a busy port and leaves the servers it already started running.
   - A pool that owns its servers kills them even when other pools attached via PORTS (`after_sim.sh`).
   - `after_fix.sh` ran `make_dense_jobs` before its retry finished.
   - The loaders silently skip bad or error rows and fall back to older dirs.
   - The `hidden=` tag is a literal in `collect.py`, edited separately from `hidden_state.py`; store hashes of both
     instead.
   - `confirm_analysis` has no override mechanism.
5. **Split null.** The plug-in null gives 96.5 expected splits, the permutation null 99.4 (127 observed). Negligible.
6. **KettleBoiling confirmation, continuation 0 vs the rest.** m = 0 is higher at 8 of 9 points (p = 0.04). All M = 2
   dirs are exchangeable, so this is likely chance; recheck on the SteamInMicrowave/PackIdenticalLunches confirmations.
7. **"When to sample" table.** No saved script exists for it.
8. **`rise_to_done`** assumes V(tc) = 1. That is plausible but unverified.

## Verified correct
- The M = 2 covariance (ddof 1) and the M = 4 correction p(1-p)/(M-1) are both unbiased in simulation.
- `cv_gain`: ties are averaged, and there is no winner's curse.
- `points()`:
  - the override order is correct;
  - every point turned >= 5 degrees is served from FIX3;
  - `roll_dir` matches the row in use at all 768 points;
  - the `_bug3`, dense and confirm rows never enter it.
- t indexes states (states[0] is the start), and the history is recorded at chunk boundaries.
- `make_jobs` k equals `k_replay` in every row.
- Episode-level bootstrap CIs are about the same as point-level ones.
- The 'complete' flag is correct on the current data. It is only fragile in principle, because refinement points count
  towards completeness.
