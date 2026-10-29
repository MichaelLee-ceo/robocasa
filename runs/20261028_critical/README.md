# Critical chunks: where does the choice of chunk decide the outcome? (started 2026-10-28)

This is question 1 of the research goal (`../README.md`): which chunks decide success or failure? Question 2 asks
whether they can be found without simulating; it is studied in `../20261029_signals`.

**Definition (the user's, 2026-10-28).** The critical chunk is the chunk where the stage's outcome gets decided. Along
the plain policy's trajectory it is where V = P(finish the stage | state, plain policy) jumps:
- a drop, e.g. a missed grasp;
- a rise, e.g. a grasp that holds.

If a difference only shows in later chunks, the critical chunk is that later chunk. It often comes 1-2 chunks before
the failure becomes visible: the approach decides whether the grasp will hold.

## Data
`data.py` is the one loader for every script here. It reads `../20261026_critic_data/data/<task>/<dataset>`
(coarse / dense / confirm / control; see that README). Outputs go to `$CRIT_OUT` (default: this folder).

## Method
1. `value_curve.py`: V along the main runs from the coarse points. Consecutive pairs with |dV| >= 0.4 are jumps.
2. `make_dense_jobs.py`, then `dense_analysis.py`:
   - `make_dense_jobs.py` puts a V point at every chunk boundary inside each jump interval: every 16 steps for
     KettleBoiling, every 32 for the others, refined to 16 around the jump.
   - `dense_analysis.py` locates the largest one-chunk jump in each interval, with its binomial SE, and writes
     `critical_chunks.json`.
3. **Critical chunk:** one chunk wide, |dV| >= 0.4, above 2 SE (one threshold for all tasks).
4. Confirmation (`confirm_analysis.py`, `CONFIRM_DS=confirm|control`):
   - at each critical chunk's start, 8 candidates x 4 continuations with salted seeds;
   - control: as many random coarse points, with the same protocol.

## Result (settled-v5 data, 2026-10-29)
**Located critical chunks.** The list is in `critical_chunks.json`.
- KettleBoiling, 5: s24 480-496, s26 96-112, s19 80-96, s17 112-128, s2 80-96.
- SteamInMicrowave, 6: s14 704, s23 688, s28 32, s29 976, s1 624, s7 960.
- PackIdenticalLunches, 21. The largest are s17 1376, s18 1504, s3 592 and s7 1200, each |dV| 0.88.

**Does choosing a chunk there matter?** The table shows the stage-completion gain in points, with bootstrap CIs over
points.

| task | critical: cross-validated pick gain | perfect best-of-8 | random: cross-validated | perfect |
|---|---|---|---|---|
| KettleBoiling (5 / 5) | **+34** [+5, +60] | 42 [26, 57] | +3 [-2, +11] | 8 [0, 18] |
| SteamInMicrowave (6 / 6) | **+20** [+0, +46] | 29 [19, 37] | -7 [-22, +0] | 0 [0, 0] |
| PackIdenticalLunches (21 / 21) | **+25** [+14, +37] | 25 [17, 34] | -2 [-10, +4] | 3 [0, 7] |
| all (32 / 32) | **+25.3** [+15.3, +35.8] | 28.3 [21.5, 35.2] | -2.3 [-8.2, +2.3] | 3.1 [0.4, 6.1] |

How the two gains are computed:
- **Cross-validated pick gain.** Pick the candidate with the best rate over 2 continuations and score it on the
  other 2. This removes the luck of picking a candidate that just happened to succeed.
- **Perfect best-of-8.** The expected gain of a picker that knows each candidate's true rate. It is computed from the
  between-candidate variance after the binomial noise is taken out.

Reading the table:
- **Critical vs random.** The difference is +27.6 [+16.6, +39.6].
  - At the critical chunks often only 1-3 of the 8 candidates finish the stage, e.g. KettleBoiling s24 t480 and
    PackIdenticalLunches s18 t1504.
  - At random chunks, 62 % of the points are already decided (V near 0 = stuck, or near 1), and the rest have similar
    candidates.
- **Not every critical chunk can be steered.** About a fifth show no gain: V jumped in the main run, but the 8 sampled
  candidates do about equally well.
- **Earlier numbers are superseded.** Pooled "best-of-8" gains of +14 / +13 / +10 at uniform points came from a
  pooled formula that overstates the gain, on data with restore bugs. The earlier confirmations (+21 / +10 / +24) used
  bug-affected restores and reused seeds. Neither should be quoted.

## History
- **Old restores.** Until 2026-10-29 the data had restore bugs (`../20261029_review`). Two old findings did not survive
  the fix:
  - the SteamInMicrowave door-stage critical chunks (s8) were bug-5 artefacts;
  - "the main run drew a rare bad chunk" at KettleBoiling s17 t112 was too. With the true restore, 4 of 8 candidates
    finish and 4 fail.
- **Signal analyses removed.** The earlier analyses of physical events, candidate disagreement, distribution shape,
  winners vs losers, overlays and time in stage ran on those data. Their outputs were removed; question 2 redoes them
  on the clean data.
- **Kept scripts.** `events.py` (the main runs' physical events, `events.jsonl`), `effect_by_event.py`,
  `disagreement.py`, `distribution.py`, `candidates.py`, `split_points.py`, `split_analysis.py`, `candidate_rank.py`,
  `overlay_critical.py`, `drop_check.py`, `grip_check.py`, `timeline.py` and `repro_drop.py` (exact replay of a saved
  rollout; `DATASET=...`). They are kept for reuse and read the data through `data.py`.
