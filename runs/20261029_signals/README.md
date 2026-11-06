# Question 2: can we tell, without simulating, where steering helps? (2026-10-29)

> **Record only (2026-11-05).** This experiment ran on the earlier pipeline, which had known problems (`../20261104_env_audit`). Its code was removed from the working tree; it is in git at commit `84e894a`. Its large outputs (git-ignored) were moved to `~/mclee/robocasa_old_data/runs/<this folder>/`. This folder keeps the README and the results. New work uses `../../envkit`.

**Short answer: no cheap signal does it.**
- None of the deployable signals separates the critical or steerable chunks from the others in a way that holds on both
  data sets. This covers the 8 candidates' disagreement and its shape, the gripper, contacts, robot motion, past events
  and the stage.
- The few hits on the 64 confirm / control points fall apart in one of three ways: they flip sign between tasks, they
  depend on the label definition, or they do not replicate on the 768 coarse points.
- A ridge regression on all 26 signals at once, trained on coarse, also fails.
- So the trigger has to judge what the candidates will do in the scene, i.e. a learned / VLM critic (see
  "Recommendation").

## Data and labels
- **Discovery set.** 32 critical chunks (`confirm`) and 32 random chunks (`control`), 8 candidates x 4 continuations
  each.
- **Replication set.** `coarse`, 768 uniform points with 8 x 2 continuations.
- **Same states.** Every point is a state on the main run; its snapshot state equals the main run's state at t (checked
  at every row), so the main run's states, trace and events describe the robot at the point.
- **Steerable** = the 8 candidates differ beyond luck: a permutation test of the candidates' rates against shuffling
  the 32 outcomes among candidates, p < 0.05.
  - The test is calibrated at every V. A fixed cut such as var_b >= 0.05 has a 1-12 % false-positive rate depending on
    V (simulated).
  - All 15 points it labels have var_b >= 0.056, i.e. a perfect best-of-8 picker gains >= 22 pts.

  | group | n | cross-validated pick gain | mean var_b | decided (V <= 0.1 or >= 0.9) |
  |---|---|---|---|---|
  | steerable (all critical) | 15 | **+46 pts** | 0.133 | 0 |
  | critical, not steerable | 17 | +4 | 0.012 | 2 |
  | random | 32 | +0.5 | -0.001 | 23 |

  - The cross-validated gain here averages the 6 ways to pick on 2 continuations and score on the other 2.
  - **Sensitivity check:** the effect-size label var_b >= 0.05 (18 points), last column of the tables.
  - **Continuous target:** var_b.
- **AUROC** compares positives and negatives of the **same task** only, so task scale and base rate cannot fake a
  separation.
  - 0.5 means no separation; below 0.5 means the signal is lower at the positives.
  - CIs: 2000 bootstrap resamples within each (task, class) stratum.
  - With 26 signals x 3 targets, about 4 CIs exclude 0.5 by chance.

## Result 1: discovery set (64 points)
AUROC [95 % CI]; * = the CI excludes 0.5. Selected rows; all signals are in `analyze.out`.

| signal | critical vs random | steerable vs not | steerable among critical (n = 32) | rho with var_b |
|---|---|---|---|---|
| var (all dims; candidate disagreement) | 0.43 [0.28, 0.59] | 0.52 [0.32, 0.73] | 0.55 [0.31, 0.79] | +0.25 [+0.01, +0.48]* |
| var_arm | **0.27 [0.16, 0.40]*** | **0.29 [0.17, 0.43]*** | 0.40 [0.20, 0.64] | -0.14 |
| var_grip | 0.53 | 0.63 [0.49, 0.79] | 0.64 | +0.27* |
| var_base / var_mode | 0.50 / 0.50 | 0.57 / 0.48 | 0.60 / 0.47 | +0.26* / -0.04 |
| bimodal (two options) / pc1 | 0.47 / 0.49 | 0.58 / 0.60 [0.39, 0.80] | 0.64 / 0.62 | -0.01 / -0.00 |
| growth (late-chunk divergence) | 0.49 | **0.72 [0.54, 0.88]*** | **0.76 [0.56, 0.92]*** | +0.18 |
| close_frac (candidates close the gripper) | 0.47 | 0.66 [0.50, 0.81] | **0.75 [0.56, 0.93]*** | +0.18 |
| gripper gap | 0.63 [0.47, 0.79] | 0.58 | 0.51 | +0.10 |
| arm_moved (last 16 steps) | **0.67 [0.52, 0.81]*** | 0.63 | 0.52 | +0.20 |
| arm_speed / base_speed | 0.57 / 0.51 | 0.55 / 0.46 | 0.50 / 0.44 | +0.16 / -0.05 |
| touch (gripper on a task object / fixture) | 0.52 | **0.67 [0.52, 0.83]*** | **0.72 [0.56, 0.88]*** | +0.21 |
| grasp | 0.53 | 0.63 | 0.66 [0.50, 0.82]* | +0.21 |
| since_event (steps since the last contact event) | 0.49 | 0.55 | 0.61 | +0.10 |
| stage index k | 0.42 | 0.46 | 0.51 | -0.01 |
| stage prior (the stage's mean var_b on coarse) | 0.56 [0.42, 0.72] | 0.54 [0.37, 0.71] | | |
| *ref: time in stage (not deployable)* | 0.38 [0.24, 0.53] | 0.54 | 0.71 [0.48, 0.89] | +0.00 |
| *ref: steps to the next event (not deployable)* | 0.39 | 0.33 [0.18, 0.50]* | 0.33 | -0.13 |

Why the hits do not count:
- **var_arm** (critical chunks have *lower* arm disagreement) flips between tasks.
  - Critical vs random: KettleBoiling 0.76, SteamInMicrowave 0.81, PackIdenticalLunches 0.20.
  - The pooled number is PackIdenticalLunches alone.
  - With the var_b >= 0.05 label it is 0.36 [0.22, 0.52].
- **arm_moved** (the robot moved in the last chunk) is consistent across tasks (0.68 / 0.58 / 0.68). It mostly says
  that random chunks are often stuck: 23 of 32 have V <= 0.1 or >= 0.9. It does not replicate on coarse.
- **touch, grasp, close_frac, growth** form one picture: at 7 of the 15 steerable points the gripper touches the object.
  At 5 of them it holds it: all candidates keep the gripper closed and their arm commands diverge late in the chunk
  (where the object goes).
  - The other 8 have the gripper open and touch nothing: approach moments (KettleBoiling s17, s19, s26;
    PackIdenticalLunches s11, s13, s19, s21) and SteamInMicrowave s29 in the door stage.
  - Holding points exist among the random chunks too, but there V is 0 or 1. So contact is not a gate.
  - None of the four replicates on coarse.
  - With the var_b >= 0.05 label none of their CIs excludes 0.5.
- **Disagreement itself** (var, bimodal, pc1, outlier, minority, pos_rel, grip_end, var_mode) does not separate critical
  from random (AUROC 0.40-0.59).
  - The candidates disagree about as much at random chunks as at the critical ones. Where the choice matters is not
    where the policy is uncertain in action space.
- **The stage** does not help: the index gives 0.42-0.51, the coarse-learned stage prior 0.54-0.56.

## Result 2: coarse replication (768 points, 8 x 2)
- **Setup.** Each signal becomes its within-task percentile. We take the quarter of points it ranks highest (in the
  direction seen on the 64) and compare their mean var_b and cross-validated gain with all of coarse.
  - All of coarse: var_b +0.0065, cv gain +1.1 pts. A useless signal's top quarter gets the same.
- **Rank correlation.** Every deployable signal's rank correlation with var_b is within [-0.03, +0.06], with CIs
  covering 0 (`analyze.out`).
- **Best top quarters.**
  - bimodal: var_b +0.0135 [+0.0076, +0.0201], cv +2.4 [+1.2, +3.8] pts.
  - pc1: +0.0132 [+0.0070, +0.0203], cv +2.3 [+1.0, +3.7].
  - That is about 2x the average, just outside the CI. It is one signal (the two are near duplicates), and it was not
    significant on the 64 (steerable 0.58 / 0.60). At most a weak hint.
  - In absolute terms, +2.4 pts at a quarter of chunks against +46 at the steerable critical chunks.
- **The discovery hits** (var_arm, growth, touch, close_frac, arm_moved): their top quarters get var_b 0.0035-0.0086,
  the same as average.
- **Power caveat.** var_b from 2 continuations is very noisy. This set can only see an enrichment of about 1.5-2x, i.e.
  a signal of AUROC ~0.7 or better for the rare steerable points. Weaker effects cannot be excluded, but they would be
  useless as a trigger.
- **What does show on coarse is "decided or not".** For "0 < V < 1" vs decided, the best deployable AUROCs are:
  - gripper open (gap) 0.61;
  - candidates not closing the gripper 0.65;
  - base still 0.60.

  Time in stage, a reference, gives 0.68. Steerable points are always undecided, but most undecided points are not
  steerable.

## Result 3: all 26 deployable signals combined
- **Method.** A ridge regression on coarse var_b (within-task percentiles of every signal).
- **Out of fold on coarse** (5 folds by episode): rho -0.10. The top quarter's var_b is +0.0043, below the average.
- **Fit on coarse, scored on the 64:** AUROC 0.46 [0.31, 0.62] for critical and 0.56 [0.36, 0.77] for steerable.
- So the cheap signals carry no usable information jointly either, at least linearly.

## Recommendation for the online detector
- **No cheap trigger.** Neither the policy's own candidate spread nor gripper, contact or motion state marks the
  chunks where the choice decides the stage.
- **Make the critic the trigger.** At each chunk, sample the 8 candidates. Have the critic (learned Q(obs, chunk), or
  the VLM) score them, and steer only where the predicted spread across candidates is large.
  - "When to steer" and "which chunk" are the same judgement. Both need a model of what each candidate will do in this
    scene.
  - The critic must also say "no difference" at the random-like chunks. Picking there gains nothing (control: about 0),
    and noise could cost.
- **Cost control.** Gate the critic with "stage still open" cues (gripper open, candidates not closing, base still;
  AUROC ~0.6 for undecided). A cheaper option is to call the critic every k chunks. Do not expect either to find the
  critical chunks by itself.
- **Training data.** The collected points already label candidates with outcomes: coarse 768 x 8, dense, and confirm /
  control 64 x 8 x 4. These are a training / validation set for a learned Q(obs, chunk) before trying the VLM. The
  confirm / control 64 are the cleanest test set.

## Files
- `features.py`: labels and signals per point (confirm, control, coarse) -> `features.jsonl`.
  - Reuses `../20261028_critical/{data,distribution,disagreement,effect_by_event}.py` and the main runs' states, traces
    and events.
  - Run as `python -I features.py 10`: 12 s with 10 workers.
- `analyze.py`: the AUROC / Spearman tables, the per-task check, the stage prior, the ridge combination and the coarse
  replication -> `analyze.out`.
  - Run as `python -I analyze.py > analyze.out`: about 3 min.
- Signal definitions are in the docstring of `features.py`. The robot joints are qpos 0-12 in every scene, checked in a
  KettleBoiling, a SteamInMicrowave and a PackIdenticalLunches scene.
