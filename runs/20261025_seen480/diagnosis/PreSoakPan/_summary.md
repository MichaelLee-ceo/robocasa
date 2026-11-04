# PreSoakPan: stage-level diagnosis (GR00T N1.5, 30 episodes, 18 succeeded, 12 failed)

**Task:** put the pan and the sponge into the sink (either order), then turn on the water. Horizon 1600 steps.

**How the stages are counted.** The packet's stages are order-free counts, not the pan and the sponge:
- stage 0: the first of {pan, sponge} is fully in the basin and the gripper is more than 25 cm away from it.
- stage 1: both objects are in.
- stage 2: `_check_success`, i.e. water on, both objects in, and the gripper away from both.

The code is `scripts/atomic_steps.py`, `_order_step`. `_task.json` describes stage 0 as "pan" and stage 1 as "sponge", which is a simplification.

**Files.** One JSON per episode (`PreSoakPan_s<N>_single.json`), contact sheets in `frames/`, running notes in `_notes.md`.

## Counts

| root-cause stage | episodes |
|---|---|
| 2 (turn on the water) | 10: s4, s8, s12, s17, s19, s22, s23, s26, s27, s28 |
| 1 (second object in) | 2: s16, s20 |

In both stage-1 cases the pan was never picked up.

| primary category | episodes |
|---|---|
| stall_at_fixture | 5: s8, s12, s23, s26, s27 |
| actuation_insufficient | 2: s4 (lever 0.164 rad), s22 (0.214 rad); water needs > 0.40 |
| stall_approach_retreat | 2: s19, s28 |
| wander_away | 1: s17 |
| missed_grasp (pan) | 1: s16 |
| stall_pre_grasp (pan, fingers closed) | 1: s20 |

**Recurring secondary categories:**
- wrong_fixture_element (pressing the round spout head instead of the lever): s8, s17, s28
- wander_away after leaving the faucet: s4, s8, s12, s23

**No failure was caused by running out of time.** Every stage-2 failure still had at least 478 steps left after both objects were in, more than the p90 of 385. Some earlier stages were slow but are not the root cause:
- s12: stage 0 took 599 steps.
- s17: stage 1 took 622.
- s26: stage 1 took 882.
- s23: stages 0 and 1 took 449 and 412.

## Scene dependence (strong)

Success by layout:

| layout | successes | failed episodes |
|---|---|---|
| L1 | 0/3 | s12, s23, s27 |
| L8 | 1/6 | s4, s8, s17, s22, s28 |
| L6 | 1/3 | s16, s19 |
| L9 | 3/5 | s20, s26 |
| L2, L4, L5, L7, L10 | 13/13 | none |

All five L8 failures and all three L1 failures happened at the faucet. The faucet types:
- **L8:** a small paddle lever at the column base, directly under a horizontal spout that ends in a large round spray head.
- **L1:** a thin black stick lever beside a black gooseneck, seen against a dark basin.
- **L9:** a stub lever on the column side. The successes (s0, s18, s13) come down on it from above, with the wrist camera pitched up at the wall.
- **L6:** a side lever housing with a hanging stick handle.

## Recurring failure mechanisms

### 1. The lever is reached but not actuated (10 of 12 failures)

The approach is normal. The time from both objects being in to the first faucet contact is 85-316 steps in successes and 75-380 in failures. Failures differ only at the lever. In successes, the water is on 22-97 steps after the first lever contact (typically ~45). The slow successes are s10 (261 steps), s11 (144) and s14 (683). Four sub-patterns:

**a. Partial sweep, then the lever slips out of the fingers and the policy leaves.**
- s22: about 200 steps of motionless contact at the L8 lever (754-953). The sweep at 968-992 reached 0.214 rad, then the paddle slipped out of the finger gap. Two short re-approaches (1220-1256, 1508-1556) did not re-engage it; the last one pushed it back to 0.119.
- s4: about 190 steps of dwell (668-840). A wrist roll took the lever to 0.164 at 864, the lever slipped, and the arm hovered for the remaining 736 steps.
- The successes show the same pattern but recovered:
  - s10, also L8: the same ~210-step dwell (752-960), but the paddle stayed in the gap through the sweep, reaching 0.50 at 1013.
  - s14 (L2): pushed to 0.327 at 928, left, came back only at 1567 and succeeded at 1574, 26 steps before the horizon.

**b. Pressing the wrong part of the faucet.** The L8 round spray head sits just above the paddle. s8 (643-838), s28 (770-858) and s17 (1284-1328) pressed against the head. The lever stayed at 0.000, and the arm then retreated and wandered.

**c. Parked beside the lever with no or negligible contact, for hundreds of steps.**
- s27 (L1): 860-1600, about 740 steps, zero contact.
- s12 (L1): 920-1330, about 10 steps of light contact.
- s23 (L1): 180 steps of contact at the stick's base (1076-1255), maximum 0.003 rad.
- s26 (L9): pressed sideways on the column 1299-1600, 0 rad. The successes in that layout push the stub lever from above.

**d. Finger close on air at the lever housing.** s19 (L6) approached twice (903-912 and 1361-1380). Each time it closed the fingers on nothing next to the housing instead of pushing the hanging handle, and in between went back to poke the pan.

**Turning the faucet never disturbed the pan or sponge.** No object left the basin during stage 2 in any of the 30 episodes. The gripper-far clause never mattered either. In all 18 successes, the success step coincides with the water coming on while the gripper is at the lever, so the lever is always more than 25 cm from both objects. No failure had the water on with both objects in.

### 2. The pan is never grasped (2 of 12)

**s16 (L6).**
- The first attempt (120-192) came down on the very tip of the pan handle. The tip and its hanging ring were under one finger, not in the gap. The fingers half-closed (width 0.046) without lifting.
- The policy switched to the sponge and spent 464-956 placing it properly, although the criterion had already counted it as in the sink at step 0.
- One more pan attempt closed on air beside the handle (1232). The pan never moved.

**s20 (L9).**
- The sponge slipped from the fingers over the basin at about 170 and fell in. The fingers closed on nothing at 176 and stayed closed (width 0.001) until 1436.
- From 430 to 793 the closed fingertips poked the pan's handle tip: ten packet "holds", each lifting less than 2 cm, with width never above 0.021.
- The pan was then abandoned. The policy turned the water on at 1516 with the pan still on the counter, 0.34 m away (skipped_or_wrong_stage).

### 3. Pick-and-place inefficiencies that cost time but were not fatal

- **Holding the object at the target before releasing:** s12 (pan, about 125 steps), s23 (about 90), s1 (about 105, success).
- **Detours:**
  - s17 spent 420-600 with the wrist pointing at its own arm and bumped the sponge.
  - s24 (success) went to the faucet before picking the sponge (540-580), then corrected itself.
  - s26 tried to carry the sponge from the left basin to the pan in the right basin and got the hand wedged against the gooseneck faucet between the basins (about 600-880).
- **Pan used as a tray (both successes):** in s13 and s21 the policy first put the sponge onto the pan on the counter, then carried both. In s21 the pan was held nearly vertical and the sponge slid off into the basin.

## How the pan is grasped and placed

- **Grasp:** by the handle in every episode where it was lifted (28/30). Success pans were lifted 5-32 cm.
- **Placement:** always fully inside the basin, never resting on the rim. Pans with a long handle often end tilted 20-38 deg, leaning against the basin wall with the handle up (s1, s3, s6, s13, s17, s21, s23, s29). That still counts, correctly.
- **Sponge:** usually dropped into the pan from 4-20 cm. Some sponges were released 3-4 cm outside the region and fell in (s7, s9, s18, s19). It ends upside-down in s9 and on its edge in s11; that is fine, because the criterion has no upright test.

## Official-criterion issues

1. **s16 and s19: the sponge counts as "in the sink" at step 0 while lying on the counter at the basin edge** (centre 0.10 / 0.094 m from the basin region). Stage 0 is credited at step 0. The later "undo" events (s16 at 852, s19 at 579) are the robot lifting the sponge out of this zone to put it in properly.
   - **Did the outcomes depend on it? No.** s16 failed because the pan was never grasped, and s19 failed at the faucet. In both, the policy also placed the sponge properly in the basin.
   - It only relabels which stage failed: s16's pan becomes "stage 1".
   - In s25 (same layout, success) the sponge started 0.21 m away and was not pre-credited.
2. **The tolerance is larger than the nominal 5 cm.** `OU.obj_inside_of` tests `dot(u, p) >= dot(u, p0) - th` with `u = px - p0` not normalised. So the margin in metres is th / (edge length):
   - about 0.08-0.2 m horizontally for these basins;
   - about 0.25-0.5 m vertically.
   - This is my reading of the code, not measured.
   - It explains why a held sponge or pan counts as "inside" well above the basin: s26 from 564, s12 from 455. In those cases only the gripper-far clause keeps the stage from completing.
   - It affected no outcome in this task.
3. **No upright test.** It is harmless here: tilted pans and an upside-down sponge are all genuinely in the basin.

## Packet and tooling notes (packet vs video)

- **Sink value wrap.** `fixtures.stage2:sink.value` reads 6.283 when the handle joint sits a hair below 0 (seen in s0, s13, s19, s20, s29). It is `abs(qpos mod 2π)`, a wrap artefact, so "max = 6.283" means the handle is off. `water_on` itself is computed correctly: on for 0.40 < q < π.
- **Stage labels.** `_task.json` stage definitions say "stage 0 = pan, stage 1 = sponge", and `objects[*].target_of_stage` is 0 for both objects. The measured stages are the order-free first / second object. So "first_incomplete_stage = 1" in s16 and s20 means "the second object", which was the pan in both.
- **Sink contacts do not say which part was touched.** `gripper_touch_intervals` cover every sink geom: basin, spout head and lever alike. The video is needed to tell them apart: s8, s28 and s17 touched the spout head, s12 and s23 the basin while lowering the pan, and s26 619-1075 is the faucet while carrying the sponge. Per the coordinator, arm-link contacts are not recorded. In all 30 episodes every handle change happens within 24 steps of a recorded finger contact, so no movement went unexplained.
- **`holds` overstates grasps.** In s20 the ten pan "holds" (430-793) are pushes by fully closed fingers. In s16 the sponge "hold" at 534-560 lifted 1 mm.
- **Online vs replay.** The online log and the replay agree in all 30 episodes.

**No measurement disagreed with the video in a way that suggests a tooling bug,** apart from the 6.283 wrap and the stage-label mismatch above.

## Uncertainty

- s8 and s28 "spout head, not lever": read from 4-step wrist frames in which the head fills the view at finger height. Medium confidence.
- s10 vs s22, why one sweep kept the lever and the other lost it: not resolvable at 4-step video resolution. Plausibly small differences in finger placement on a small paddle.
- s26: arguably stage 1's 882 steps also contributed. I chose stage 2 as the root cause because 478 steps remained (above p90) and the 300 steps of contact at the column produced 0 rad.
- s19 / s28 finger closes at the faucet: it is unclear whether these were grasp attempts on the lever or approach artefacts.
- The layout effect is strong (L1 0/3, L8 1/6), but the faucet geometry and the visual style change together, so I cannot separate "lever shape" from "lever visibility" (e.g. black stick on black in L1).
