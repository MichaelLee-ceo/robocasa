# Does DiVeR's decision-criticality score find our critical chunks? (2026-10-31)

**Question.** DiVeR (arXiv 2610.04933, NeurIPS 2026) scores how decision-critical a state is by the dispersion of K
sampled candidate chunks in the policy's own feature space: u_t = mean over coordinates of Var_k[z^(k)], z = the action
expert's final-layer action tokens at the last denoising step, averaged over tokens, K = 32. It validates u_t against a
rollout oracle (max - mean candidate Q) on 500 RoboCasa *atomic* states: Spearman 0.49. Our signals study
(`../20261029_signals`) found that the variance of the *raw* candidate actions does not mark the chunks where choosing
helps (steerable vs not: AUROC 0.52). Does the feature-space version do better on our composite tasks, at the critical
chunks that the oracle located?

**Data** (`../20261028_critical/data.py`, all settled-v5 or equivalent):
- confirm: the 32 located critical chunks (KettleBoiling 5, SteamInMicrowave 6, PackIdenticalLunches 21), 8 x 4;
- control: 32 random chunks, same protocol;
- dense: the chunk-by-chunk points inside the intervals where the coarse V jumps (53 / 92 / 300), 8 x 1;
- coarse: 768 uniform points along the main runs (250 / 262 / 256), 8 x 2: the replication set of the signals study,
  and the points that give DiVeR's per-trajectory standardization.
Labels as in the signals study: critical (confirm) vs random (control); steerable (permutation test, 15 points);
var_b (continuous).

**Method.**
1. `../scripts/serve_features.py`: serve_seeded.py plus a forward hook on the action head's DiT; each response carries
   the DiT output at the 16 action positions of the last of the 4 denoising steps (`_feat`, 16 x 1024). z = its mean
   over the 16 tokens (DiVeR's choice; the full tokens are kept).
2. `extract.py` (launched by `server.sh` x 4 and `run.sh`): at every point, the snapshot's request with the 8 labelled
   candidates' seeds (k < 8) and, in confirm / control, 24 new seeds (K = 32). Reproduction check against the recorded
   first chunks. Pilot (KettleBoiling s17 t112): 32 calls in 9 s; all 8 labelled chunks identical to the recorded ones.
3. `analyze_u.py`: u8 (the 8 labelled candidates) and u32; DiVeR's per-trajectory standardization over the coarse
   points of the same main run; the signals tables (within-task AUROC with bootstrap CIs, Spearman with var_b), with
   the raw-action variance as the reference row; along dense, whether u peaks at the critical chunk.

**Decided before running.**
- Usable as a trigger: steerable vs not AUROC >= 0.7 with the CI excluding 0.5 (the bar of the signals study).
- Consistent with DiVeR: a clear positive Spearman with var_b.
- With 64 points the CIs are about +-0.2, so only a strong signal can show.

**Reproduction caveat.** The snapshot holds candidate 0's first request. At many points the other candidates' first
requests differ from it in some camera images (GPU render noise; state and instruction are equal): pairs of candidate
vs candidate 0 with equal requests are 43 / 70 (KettleBoiling), 41 / 84 (SteamInMicrowave), 72 / 294
(PackIdenticalLunches) in confirm + control. A labelled candidate is reproduced bit for bit only where its request
equals candidate 0's; elsewhere `extract.py` logs how far the chunk moves. DiVeR's u_t is defined at one observation
with only the noise varying, which is what the snapshot's request gives.

## Results (2026-10-31; `analyze_u.out`)
**Short answer: no.** DiVeR's u_t does not mark the critical chunks, the steerable ones, or even the undecided points.
No variant reaches the bar; on the 768 coarse points its correlation with var_b is about 0, against DiVeR's 0.49 on
atomic tasks with pi0.5.

Discovery, the 64 confirm / control points (within-task AUROC [95 % CI]; * would mark a CI that excludes 0.5):

| signal | critical vs random | steerable vs not | steerable among critical | rho(var_b) |
|---|---|---|---|---|
| u8 (DiVeR's u, 8 labelled candidates) | 0.47 [0.31, 0.63] | 0.55 [0.38, 0.73] | 0.62 [0.40, 0.84] | +0.06 [-0.18, +0.29] |
| u32 (DiVeR's K = 32) | 0.48 [0.33, 0.64] | 0.60 [0.44, 0.76] | 0.68 [0.47, 0.88] | +0.13 [-0.09, +0.36] |
| utok (per-token dispersion) | 0.43 [0.27, 0.58] | 0.47 [0.29, 0.65] | 0.50 [0.26, 0.74] | -0.05 [-0.29, +0.16] |
| uz (standardized within the main run) | 0.44 [0.28, 0.61] | 0.48 [0.33, 0.64] | 0.52 [0.28, 0.74] | +0.03 [-0.19, +0.25] |
| *ref: raw-action variance (signals study)* | 0.43 [0.28, 0.59] | 0.52 [0.32, 0.73] | 0.55 [0.31, 0.79] | +0.25 [+0.04, +0.45] |

- u8 and u32 rank the points alike (Spearman +0.91), so K = 8 is not what hides a signal.
- u32's 0.68 among the critical chunks is the best number, but on 32 points its CI reaches 0.47.

Replication, the 768 coarse points (298 undecided, 0 < V < 1):

| signal | rho(var_b) all | rho(var_b) undecided | undecided vs decided (AUROC) |
|---|---|---|---|
| u8 | +0.03 [-0.03, +0.10] | +0.06 [-0.05, +0.18] | 0.51 [0.46, 0.55] |
| uz | +0.03 [-0.04, +0.09] | +0.05 [-0.07, +0.15] | 0.50 [0.46, 0.54] |
| *ref: raw-action variance* | +0.02 [-0.04, +0.09] | +0.05 [-0.06, +0.16] | 0.48 [0.44, 0.53] |

Temporal, the 32 critical chunks against the dense points of the same main run within +-64 steps:
- mean percentile of u at the critical chunk 0.60 [0.50, 0.71] (0.5 = no preference);
- the largest u sits at the critical chunk in 7 / 32 (chance 7.5) and one chunk after it in 9 / 32 (chance 7.2).

**Reading.**
- On these composite tasks with GR00T N1.5, how much the policy's own samples disagree, in action space or in its
  action-head features, says nothing about whether the choice matters there. This agrees with the signals study and
  extends it to DiVeR's feature space and per-trajectory standardization.
- What this does not test: DiVeR uses u_t only to reweight verifier training, never as a test-time trigger. A weak
  weight can still help training a little; it cannot serve as our gate.
- Possible reasons for the difference from DiVeR's 0.49 (not separated here): policy (GR00T's 4-step flow DiT vs pi0.5's
  action expert), task length (composite vs atomic), and target (DiVeR's max - mean Q over all states vs our var_b).

**Do the features carry outcome information at all?** (`probe.py`, `probe_last.out`) u_t only measures spread. Three probes
on the same z, using the 10,040 of 10,216 labelled candidates whose re-sampled chunk is within 0.05 of the executed one:
1. **Pipeline.** u8 moves with the raw-action variance on coarse: Spearman +0.13 (KettleBoiling), +0.39
   (SteamInMicrowave), +0.53 (PackIdenticalLunches). The features follow the candidates' actions.
2. **State value.** The candidates' mean z predicts the point's V, cross-validated by main run: rho +0.50 / +0.36 /
   +0.41 and AUROC (V > 0.5) 0.74 / 0.72 / 0.68. Stage index and time alone already give 0.73 / 0.70 / 0.65, so the
   features add little beyond "which stage, how late".
3. **Which candidate.** A ridge model of the candidate's z relative to its siblings, predicting its success relative to
   theirs, trained on the other main runs, picks a candidate at the held-out points:

   | | features | shuffled labels |
   |---|---|---|
   | critical chunks (32) | +0.7 pts [-11.3, +12.9] | -1.6 |
   | random chunks (31) | +3.6 pts [-0.7, +8.8] | -1.2 |

   The in-sample best of 8 is +42 at the critical chunks (biased upward; the cross-validated oracle is about +25).
   So a linear read-out of these features finds nothing of the action-dependent part.
- **Limits of probe 3.** It is linear with fixed settings (32 PCs). Its training signal is sparse: most points have all
  candidates with the same outcome, and coarse and dense have 1-2 continuations per candidate. The features come from
  the last denoising step, whose input is still a noisy action (t = 0.75), and are averaged over the 16 tokens.

**Is the feature position the reason? No.** The clean pass (`serve_features.py`) runs the DiT once more on the finished
chunk at timestep bucket 999 (t = 0.999), the policy's own representation of the chunk that is executed. It is
re-extracted at every point with K = 8 into `feats_clean/` (`run_clean.sh`). The pilot showed the actions and `_feat`
bit-identical to the first extraction. `probe.py --feat <variant>` gives `probe_<variant>.out`:

| feature | V > 0.5, AUROC (KB / SIM / PIL) | pick gain at critical chunks | shuffled labels | at random chunks |
|---|---|---|---|---|
| last (DiVeR's position) | 0.74 / 0.72 / 0.68 | +0.7 [-11.3, +12.9] | -1.6 | +3.6 |
| steps (the 4 steps concatenated) | 0.75 / 0.72 / 0.70 | -0.1 [-12.0, +12.1] | -7.9 | +2.8 |
| clean (the finished chunk, token mean) | 0.64 / 0.61 / 0.58 | -4.0 [-14.4, +5.5] | +4.6 | -1.2 |
| clean_first (its first token) | 0.60 / 0.67 / 0.55 | -2.4 [-13.0, +8.8] | -1.6 | +0.4 |
| raw (the chunk's action values) | 0.58 / 0.63 / 0.59 | -3.2 [-12.7, +6.0] | +1.5 | +2.0 |
| *stage index + time* | 0.73 / 0.70 / 0.65 | | | |

- No variant picks better than its shuffled-label control at the critical chunks.
- The clean pass carries *less* about the state's V than the noisy last step. At t near 1 the DiT only has to output a
  near-zero velocity.
- Power: with 32 critical chunks the CIs are about +-12 points. A picker with half the cross-validated oracle (+25)
  would only just show.

**Reproduction** (`feats/log_w*.jsonl`, per labelled candidate `exact` and `maxdiff`):
- settled-v5 rows, candidate request equal to the snapshot's: 1,444 / 1,444 identical.
- Older rows (an earlier server setup), equal request: 33 % identical, median difference 0.004, 0.7 % flip a discrete
  action. The first run stopped on these (strict check applied to all rows); now only
  settled-v5 rows are held to identity (`../20261028_critical/data.py` documents the distinction).
- Candidates whose first request differs from the snapshot's only by GPU render noise, settled-v5 rows: a chunk
  differing by > 0.5 (a discrete action flipped) in KettleBoiling 0 / 168, SteamInMicrowave 1.0 % of 192,
  PackIdenticalLunches 2.2 % of 912. Render noise alone can change the policy's discrete decisions; a verifier trained
  on per-candidate features must use only candidates with `exact`.
