# SetUpCuttingStation: stage-level diagnosis (GR00T N1.5, 30 episodes, 14 succeeded, 16 failed)

Task: take the knife out of an already-open drawer and put it on the cutting board (stage 0), then move the meat from its plate onto the board (stage 1). Horizon 1600 steps.

Files:
- Per-episode results: `SetUpCuttingStation_s<N>_single.json`.
- Contact sheets: `frames/`.
- Running notes: `_notes.md`.
- Cross-episode comparison sheets:
  - `frames/cmp_knife_lift_wrist.png`: wrist view just after the knife lifts, for slipped grasps (top row) and successful ones (bottom row).
  - `frames/cmp_layout9_drawer.png`: the layout-9 drawer in six episodes.
  - `frames/cmp_flatsteak_grasp.png`: grasps on the flat steak_8 in s25 (lifted) vs s20 and s21 (slipped).

## Counts

| root-cause stage | episodes |
|---|---|
| 0 (knife from drawer to board) | 12: s7, s8, s12, s13, s15, s16, s17, s20, s21, s22, s27, s29 |
| 1 (meat to board) | 4: s0, s11, s18, s23 |

In every stage-1 failure the root cause really is stage 1:
- Stage 0 was still complete at the end in all four, so the knife never left the board.
- s18's stage 0 was slow (962 steps; p90 724), but it still had 638 steps left, more than stage 1's p90 of 478, and it spent them wandering.

| primary category | n | episodes |
|---|---|---|
| dropped_in_transit (knife lifted, then slipped out) | 4 | s7, s12, s15, s17 |
| missed_grasp | 4 | s20, s21, s22 (knife); s0 (meat) |
| stall_pre_grasp (over the knife) | 3 | s8, s13, s27 |
| placed_off_target | 2 | s29 (knife between the plate and the board); s23 (fish over the board corner) |
| wander_away (stage 1, the meat was never touched) | 2 | s11, s18 |
| skipped_or_wrong_stage (never tried for the knife, went to the meat) | 1 | s16 |

Recurring secondary categories:
- The policy moved on to the meat with the knife not on the board in 7 episodes: s16, s17, s20, s21, s27, s29 and s7. In s17 it even placed the meat correctly.
- It carried on empty-handed after a slip in 5 episodes: an empty "place" over the board or counter in s8, s15, s17 and s22, and over the board with the meat in s0.

Checklist of the failure types the task brief asked about:

| failure type | episodes |
|---|---|
| knife never grasped | 6: s13, s16, s20, s21, s22, s27 |
| knife dropped after a lifting grasp | 5: s7, s8, s12, s15, s17 |
| knife placed off the board | 1: s29; s7's last attempt also ended at the board edge |
| meat missed | 1: s0 |
| meat placed off the board | 1: s23 |
| meat placement pushed the knife off the board | **0**: in s1, s2, s4 and s28 the meat was laid on or against the knife without moving it |
| ran out of time with the right stage in progress | 0 as a root cause; it is secondary in s7, s8, s13, s15, s22 and s27 |

## Recurring failure mechanisms

### 1. Knife grasps that lift and then slip (5 episodes; 4 as primary)

The thin knife is lifted 9-24 cm and then pivots out of the fingers 20-45 steps after the grasp. The fingers then shut to about 0.002.

| episode | grasp → slip (steps) | notes |
|---|---|---|
| s7 | 162→204, 677→716 | |
| s8 | 1413→~1440 | |
| s12 | 514→~558 | |
| s15 | 1002→~1035, 1363→~1395 | |
| s17 | 260→~370 | lifted 0.24 m |

**Where the fingers held the knife** (wrist view, `cmp_knife_lift_wrist.png`):
- **Slips (s7, s8, s12, s15):** the fingers closed on the butt end of the handle, or near the bolster in s8, with the handle running diagonally across the finger gap. On lifting, the blade's weight swings the knife around the contact point and it slides out.
- **Successes:** the handle runs straight through the finger gap, perpendicular to the closing direction, and is held near its middle.

**s17 is the exception.** Its grasp looked like a successful one: centred and straight. It slipped later, while the arm swung toward the board and the wrist rotated (340-376).

**After a drop:**
- The knife often lands on its edge or flipped (s7, s17), and every re-grasp of an edge-on knife failed: s7 1119, s17 616/752/904.
- An open-finger press on the handle can also roll the knife onto its edge before any grasp. This happened in s20 (163-177, never recovered) and in success s10 (237-319, recovered at its 4th close, 618).

Scene dependence:
- Layout 10: 0/3 succeeded. s7 and s15 both had butt-end slips; s29 released off the board.
- Layout-8 drawer at (3.85, -2.6): 2/5 succeeded.

Knife models:
- knife_8 completed stage 0 in 3/8 episodes and knife_5 in 5/5.
- The samples are small. I could not separate the knife model from the layout.

### 2. Knife deep in the drawer under the counter overhang: hover without contact (s13, s22, s27; partly s8)

When the knife lies at the back of the drawer, the blade is under the countertop edge and only the end of the handle sticks out in the wrist view. The policy hovers in front of the counter edge:

| episode | layout | hover | closes | closest approach to the knife centre |
|---|---|---|---|---|
| s13 | 9 | ~1400 steps | none at all in 1600 steps | 0.145 m |
| s27 | 9 | ~1000 steps | 2, on air (1059, 1146) | 0.22 m |
| s22 | 8 | repeated approach/retreat | 5, on air (134, 376, 896, 1232) | never touched |
| s8 | 8 | ~900 steps (340-1260) | grasped at 1413, then slipped | — |

Counter-example: s2 had a similar back-of-drawer layout-9 start and grasped at 397. The layout-9 drawer at (0.5-0.6, -3.45) gave 2/6 successes; the stage-0 failures there were s13, s20 and s27.

### 3. Giving up on the knife and switching to the meat (stage-order failure; 7 episodes)

After some hundreds of steps of failed knife attempts, the policy goes to the meat. The knife is never retried.

| episode | knife phase | switched to the meat at |
|---|---|---|
| s20 | 5 failed closes | ~600 |
| s21 | hover 440-900 after the slip | ~940 |
| s27 | hover | ~1250 |
| s17 | 3 failed re-grasps | ~1100; it placed the meat on the board at 1255 |
| s16 | never approached the knife (black handle in a black drawer) | ~320 |
| s29 | released the knife beside the board and behaved as if it were done | — |
| s7 | ran out of time with the knife off the board | 1582 (touched the meat, no lift) |

Low contrast is only a weak explanation for s16: s1 had the same black knife in a black drawer and succeeded.

### 4. Stage-1 errors after a good knife placement (4 episodes)

- **Never went for the meat (s11, s18).**
  - s11: the plate was 2.6 m away behind an oven column. The robot went back to the knife drawer (680-1000, close on air at 928), turned past the plate (~1200) and ended in a corner.
  - s18: the plate was only 0.46 m from the board. The robot turned away to the kitchen island at 1160 and drifted until the timeout.
  - In successes, the first close on the meat comes 64-234 steps after the knife release (median ~95).
- **Grasp on a thin or tilted meat edge (s0).** The fingertips closed on the near edge of steak_6, which leans on the plate rim (596). It lifted 5 mm, slipped, and the policy did an empty "place" over the board, then wandered with the gripper closed (624-1440).
  - The flat T-bone steak_8 shows the same failure: 14 failed closes in s20 and s21 (max lift 9 mm, fingers closing to 0.004-0.015).
  - The one steak_8 success (s25) closed around the narrow bone end at a width of about 0.06 instead of pinching the thin edge (`cmp_flatsteak_grasp.png`).
- **Release over the board's corner (s23).** The fish was carried by the tail fin, head down, and released over the board's front-left corner (0.196 m from the centre). It fell half off the board. The policy then closed on air over the empty plate (1451).

## What successful episodes do differently

- **Knife grasp.**
  - 9/14 successes lift the knife on the first close that touches it.
  - The first lifting grasp comes at a median of ~338 steps (range 182-618, s10 slowest).
  - The handle runs straight and centred through the finger gap, and the lift goes to 0.35-0.50 m without slipping.
  - Recoveries:
    - s10 re-grasped an edge-on knife at its 4th close.
    - s3 succeeded on its 2nd close.
    - s25 succeeded on its 2nd close.
- **Knife release.** On the board, 0.013-0.135 m from the centre. The policy goes straight to the meat afterwards.
- **Meat grasp.** On a graspable part: fish head or tail, drumstick, upright shrimp, or around the narrow end of a steak. Released 0.02-0.12 m from the board centre.
- **Slow successes.**
  - Stage 0 was slow in s3 (1052) and s10 (773). s3's time went into a 2.4 m base move while holding the knife; s10's into the edge-on recovery.
  - Stage 1 was slow in s19 (801 steps: three failed fish closes 691-1199, then a tail grasp at 1303) and s1 (496: a fish miss at 558, re-grasped at 856).

## Official-criterion and packet issues

1. **One-step stage-0 drops with a stationary knife.** The undone reason "an object left the target" fires for single steps while the knife lies still on the board: s4, s9 (4 times), s14, s19 (14 times), s23 and s26. These look like contact flicker in the on-board check, not real undos. Anyone counting undos from `undone_events` will over-count.
2. **`close_events` is unreliable in both directions.**
   - `on_nothing=true` marks closes that became the successful grasp, because contact began 15-20 steps after the fingers started closing:
     - s5 close at 225 (touch from 241)
     - s3 close at 410 (touch from 431)
     - s25 close at 252, which touched the knife at 269-275
   - Wide closes are missing altogether: s25's steak grasp narrowed the gripper only from 0.080 to 0.060 and has no close event, although hold 836-1019 is recorded.
   - Open-finger pushes appear as holds with no close event: s18 669-681, s20 163-177.
   - So `on_nothing` cannot be trusted alone for "closed on air". I checked every on_nothing close in the failures against the touch intervals and the wrist view.
3. **Tilt fields are meaningless for several models.**
   - knife_9 reads 24.5 deg while lying flat (s18, s22, s27).
   - fish_9 (s1, s11) and the drumstick in s8 are "tipped_over" from step 0.
   - The knife lying on its edge on the board in s10 still counts as placed: there is no orientation test, which was harmless here.
4. **`dist_to_target_*` is horizontal.** A knife in a drawer directly below the board reads 0.08-0.30 m from it (s16 at the end, s21, s7, s16 at the start). `at_target` is correct.
5. **Object moves without a gripper touch are invisible in `gripper_touch_intervals`.**
   - s13: the knife moved 4.4 cm with no touch recorded (arm contact?).
   - s16: the knife moved 16 cm when the falling steak hit it.
6. **`official_labeler` segments are meaningless on all failures, as expected.** On s13, s21 and s22 they collapse to step 0-5.

## Uncertainty

- The butt-end/diagonal-grasp explanation for the slips comes from 4-step wrist frames. It fits s7, s8, s12 and s15, but not s17.
- For s13, s22 and s27 I cannot tell whether the counter overhang physically blocked the descent or the policy just did not commit. The packet has no drawer-opening value or knife pose relative to the drawer.
- The scene and model effects (layout 10 0/3, knife_8 3/8, steak_8) rest on 2-8 episodes each.
- Medium-confidence episodes: s13 (stall cause), s20 (how the knife was tipped), s22 (why the closes missed), s19 (light pass).
