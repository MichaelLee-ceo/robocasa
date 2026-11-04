# WashLettuce: stage-level diagnosis (GR00T N1.5, 30 episodes, 19 succeeded, 11 failed)

Task: turn on the sink faucet (stage 0), then get the lettuce under the running stream for 25 env steps in total (stage 1, the env's `washed_time` counter). Horizon 1100 steps.

The lettuce always starts in a colander on the counter, left or right of the sink. In every episode the policy grasps the colander (rim, loop handle or wire handle) and carries the colander with the lettuce under the spout. It never picks up the lettuce itself.

- Per-episode files: `WashLettuce_s<N>_single.json`.
- Contact sheets: `frames/`.
- Running notes: `_notes.md`.

## Counts

| root-cause stage | episodes |
|---|---|
| 0 (turn on the faucet) | 3: s4, s8, s10. All are layout 8, the paddle-lever sink. |
| 1 (lettuce under the stream) | 8: s2, s3, s6, s14, s18, s19, s26, s28 |

In every failure the root-cause stage is the first incomplete stage. No failure was caused by running out of time: every stage-1 failure had at least 550 steps left after the water came on.

| primary category | episodes |
|---|---|
| other: held_beside_stream (the colander bumped the swivel spout, so the stream ended up at the colander rim) | 4: s2, s3, s18, s28 |
| actuation_insufficient (lever left at 0.28-0.29 rad, below the 0.40 threshold) | 2: s4, s10 (stage 0) |
| missed_grasp (colander slipped from the fingers) | 2: s14, s19 |
| missed_grasp_continued_empty | 2: s6 (empty carry to the stream, stage 1), s8 (closed beside the lever, then a "turning" motion with the gripper closed on nothing, stage 0) |
| stall_pre_grasp | 1: s26. It hovered over the colander for 380 steps and never closed the fingers, after a slow stage 0 that ended at 550. |

Failure modes the coordinator asked about that did **not** occur:
- the water turned off by accident (0 cases);
- the lettuce put in the basin away from the stream: only s3, which set the colander down beside the stream at the very end.

## Recurring failure mechanisms

### 1. The colander pushes the swivel spout aside, so the stream lands on the rim (4 failures: s2, s3, s18, s28)

The water was on and the colander was carried to the faucet at a normal time (~520-540 after water at ~140-270). Then:
- The colander (or the forearm) pressed against the spout head and the spout swivelled about 90 degrees: s2 at 548-700, s3 at 540-580, s18 at 440-560, s28 at 540-580. In the agent view the gooseneck changes from a thin vertical line to an arch seen in profile (`s2_spout.png`, `s3_spout.png`, `s18_spout.png`, `s28_spout.png`).
- The stream moved with the spout and ended up falling on the colander's rim or just beside it (`s2_sinkzoom.png`, `s3_sinkzoom.png`, `s28_sinkzoom.png`).
- The lettuce sits about one colander radius away from the rim, so `washed_time` stayed at **0** for the whole episode.
- The policy then held or adjusted the colander next to the stream for 500-700 steps. It never followed the stream to its new position, and the wrist camera, filled by the colander, mostly cannot see the stream.
- s3 finally set the colander down beside the stream (~1000) and withdrew. s18 lifted it to about the height of the spout head.

Layout dependence:
- Layout 9 (pull-down faucet, double basin): 3 of 5 episodes failed this way (s2, s3, s18). A fourth, s26, failed by stalling. The only layout-9 success, s13, slid the colander under the spout outlet without pushing the spout (`spout_swing_compare.png`).
- Layout 7: s28 is the only failure of 5 episodes. Its colander has a long wire handle and came in from the left. In s0, s5, s12 and s29 the spout stayed forward (`L7_spout_compare.png`).

The packet cannot show this mechanism, for two reasons:
- It records only the sink handle and `water_on`, not the spout swivel joint.
- `gripper_touch_intervals` count only finger contacts, so the colander or arm pushing the spout is not recorded.

### 2. The layout-8 paddle lever is not turned far enough (all 3 stage-0 failures: s4, s8, s10)

All three are layout 8 / style 8. In this layout the lever is a tall paddle at the right of the spout base. The fourth layout-8 episode, s22, pushed it from 0 to 0.50 rad within ~50 steps of contact (169-235).

| episode | what the gripper did at the lever | end state |
|---|---|---|
| s4 | rested on the lever 116-594 (~480 steps); the lever crept to 0.14, then reached 0.29 as the arm withdrew | 0.29 rad |
| s10 | closed on the lever at 240 and held it until 571 with the angle at ~0; the base drifted; the lever reached 0.28 at release | 0.28 rad |
| s8 | closed on air beside the lever (166-186), then rotated the closed gripper around the spout for 220-640 (lever 0); then went to the colander with the water off and hovered there until timeout | 0 rad |

None of the three returned to the lever.

Stage 0 is also fragile outside layout 8. Long lever stalls that the policy eventually recovered from:
- s1: ~480 steps; water at 722, success with only 52 steps to spare;
- s15: ~260 steps;
- s16: left the lever exactly at the 0.40 threshold for 130 steps (no stream), then nudged it on;
- s26: ~335 steps (this one then failed in stage 1).

### 3. The colander slips at the grasp, and the policy does not retry properly (3 failures: s6, s14, s19)

| episode | what happened after the slip |
|---|---|
| s6 | closed at the rim (406) and slipped; carried on to the stream empty-handed (424-540); went back and pushed the colander with open fingers until timeout, never re-grasping |
| s14 | slipped at 389/409; retried once at 659 and slipped again; then wandered high above the sink (690-1100) |
| s19 | slipped once at ~420-444; kept the fingers **closed** for the remaining 660 steps while hovering over the colander |

In each case the fingers closed on the curved bowl wall next to the lettuce or rim and went to width 0.002-0.009.

Scene pattern: all three are **Colander008** (perforated, with a loop handle). That is 3 of the 9 episodes with this colander; none of the other 21 episodes had a grasp slip. In `colander008_grasp_compare.png` I could not see a clear difference in finger placement at the moment of closing between these three and the six Colander008 successes. The pattern is real, but its physical cause is unknown.

### 4. Hovering without a grasp (1 failure: s26)

In s26, after the slow lever (water at 550), the arm swung awkwardly (600-680). It then hovered over the wire colander, which was left of the sink, from 720 to 1100 without one close. The colander was on the far left of the counter (layout 9).

## What successful episodes do differently at the critical moments

- **Lever.**
  - Contact to water-on takes ~20-60 steps (e.g. s0 204→231, s21 179→204, s29 174→202). Stage 0 has a median of 253 steps, most of which is the approach.
  - In the layout-8 success s22, the paddle was pushed past 0.40 in one continuous push.
- **Colander grasp.**
  - A single close on the rim or handle holds. Width after closing is 0.02-0.07; on thin wire handles the fingers stop at ~0.06-0.07.
  - The colander is lifted and carried in 100-200 steps.
- **Placing under the stream.**
  - The colander is brought to the spout outlet from the front or side without pushing the spout head.
  - The stream runs through the colander centre onto the lettuce, and `washed_time` reaches 25 within ~1-3 chunks of arrival.
  - Stage 1 takes 237-396 steps (median 303).
- **One lucky success, s11.** The colander slipped from the fingers at ~703 and dropped into the basin. The lettuce rolled out and came to rest right under the stream, so success came at 742.

## Official-criterion and packet issues

1. **`objects` is `{}` in every WashLettuce packet.** `make_packets.py` builds object entries only for stages of kind `pnp`, and stage 1 here has kind `process`. So the packets have no lettuce position, holds or distance to the stream, and no "target = sink interior" entry, despite what the task notes say. Every judgement of where the lettuce was relative to the stream had to come from the video.
2. **The colander is not tracked.**
   - `close_events.touching` checks only the lettuce, so colander grasps show as `on_nothing: true` (s2, s9, s13, s18, ...) or as "touching lettuce" when one finger brushes it (s6, s14).
   - Slow squeezes on thin wire handles produce no close event at all (s0, s7, s25, s27, s28).
3. **The spout swivel joint is not recorded**, and arm or colander contact with the sink is not recorded. This hides failure mechanism 1 from the packet completely.
4. **Sink value 6.28 is a tooling artifact** (the angle is wrapped mod 2π, then abs is taken; the coordinator has confirmed this). Seen in s1, s8, s10, s15 and s18; the effective angle is ~0. `water_on` agrees with the video everywhere.
5. **Near the threshold, the criterion agrees with the video.** In s16 the handle sat at exactly 0.40 (water_on needs > 0.40) for 130 steps, and the video shows no stream either.
6. **The online log, replay and success agree in all 30 episodes.** Every failure has `washed_time = 0`: in none of them did the lettuce ever count as under the stream, not even briefly.
7. **The official labeler is meaningless on failures.** For the layout-8 stage-0 failures it emits "subtask_idx starts at 1" and a "place" segment at step 0. On stage-1 failures it emits "place"/"done" segments.

## Uncertainty

- **Mechanism 1** rests on reading the spout's silhouette in a 256-px agent view. I am confident the spout swivelled in s2, s3, s28 and s18 (less clear in s18, where the arm covers the faucet after 660). I am less sure that the swivel *alone* explains the miss, because without the lettuce trace I cannot measure how far the lettuce would have been from the original stream position. Confidence: medium for s2, s18 and s28, high for s3.
- **Mechanism 3:** the Colander008 link is 3/9 vs 0/21. Small n, and the visual cause is unexplained.
- **s6:** whether a lettuce leaf was caught between the fingers during the 424-540 "carry" is uncertain. The wrist view shows nothing held, and the colander stayed on the counter.
- **s4 / s10:** why ~300-480 steps of contact moved the lever so little cannot be measured from the packet. It looks as if the fingers rested on or held the paddle without pushing it sideways.
