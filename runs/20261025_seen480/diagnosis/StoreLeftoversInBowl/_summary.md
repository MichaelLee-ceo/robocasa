# StoreLeftoversInBowl: stage-level diagnosis (GR00T N1.5, 30 episodes, 11 succeeded, 19 failed)

**Task.** Put the drumstick and a vegetable from their plates into a bowl (stages 0/1, either order), then carry the bowl to the already-open fridge and leave it on a shelf (stage 2). Horizon 1700 steps.

**Files.** One `StoreLeftoversInBowl_s<N>_single.json` per episode, contact sheets in `frames/`, working notes in `_notes.md`. Useful comparison sheets:
- `frames/L9_release_compare.png`: the bowl-release moment of all 12 layout-9 episodes.
- `frames/L10_place_compare.png`, `frames/L10_final_compare.png`, `frames/L10_final_zoom.png`: the same for layout 10.

## Counts

| root-cause stage | episodes |
|---|---|
| 2 (bowl to the fridge shelf) | 18: s0 s1 s2 s3 s4 s8 s11 s12 s18 s19 s20 s21 s22 s25 s26 s27 s28 s29 |
| 0 (first food into the bowl) | 1: s10 |

Stage 2 is the root cause in 18 of 19 failures, but the packet's `first_incomplete_stage` says otherwise:
- It is 0 in 9 episodes and 1 in 6, because a bowl that falls spills the foods and undoes stages 0/1.
- Only 4 episodes show 2 (s21, s25, s26, s29).
- No failure ran out of time. In every failure the bowl reached the fridge, or was lost, with 400-900 steps left.

| primary category | episodes |
|---|---|
| placed_off_target | 7: s0, s3, s28 (released 8-11 cm outside the interior, fell); s21, s25, s26, s29 (stands on a shelf; official region check fails) |
| placed_then_fell_or_slid | 6: s2, s18, s19, s20, s22 (released or slipped at the shelf front edge, fell to the floor); s10 (lime rolled off the bowl rim) |
| undid_previous_stage | 3: s4, s8, s11 (foods slid out of the tilted bowl at the fridge, before or while it was released) |
| dropped_in_transit | 2: s1 (slipped out of the fingers during a 320-step base stall), s12 (the grasp held the drumstick, and the bowl slid off it) |
| stall_holding_at_target | 1: s27 (held the bowl above the top shelf ~270 steps while the grip loosened; foods and bowl fell) |

Secondary categories that recur:
- **Bowl carried tilted > 45 deg in the rim grasp:** 17 of 19 failures (all but s3 and s12).
  - It is also common in successes: 10 of 11 have tipped intervals during the carry (all but s17), but none releases the bowl tilted. So the tilt is not the cause by itself; releasing it tilted is.
- **official_criterion_issue:** s20, s21, s22, s25, s26, s29.
- **stall_holding_at_target:** s0 (drumstick held over the bowl ~220 steps), s4, s18.

## Recurring failure mechanisms

### 1. The bowl is released short of the shelf: at or in front of its front edge, often tilted, and falls to the floor (9 failures, plus 3 at the top shelf)

The policy brings the bowl to the fridge opening, opens the fingers (or the pinch slips) before the bowl sits flat on a shelf, and the bowl tumbles about 1 m to the floor.
- Released at the shelf front edge, or slipped there: s0 (1268), s2 (1192), s11 (784), s18 (968), s19 (756), s20 (860), s22 (816, the pinch slipped).
- Released several cm in front of the opening: s3 (968, 8 cm), s28 (756, 11 cm, low beside the door).

Saved states show that the insertion depth separates success from failure:

| layout | successes, release depth | failures, release depth |
|---|---|---|
| L9 (side-by-side, narrow right compartment; deeper = +x) | x 5.56 (s5), 5.63 (s17); s10's correct shelf placement 5.48 | x 5.38-5.46 (s3 s11 s18 s19 s20 s22), 5.24 (s27) |
| L10 (French door; deeper = -x) | end x 0.18-0.28 (s6 s13 s14 s15 s7) | s2 0.375, s0 0.457 |
| L8 | s16 0.55 | s28 0.79 |

So the failures release the bowl 5-25 cm shallower than the successes.

**Wrong height.** In L9 the release height is often wrong too. Successes set the bowl on a middle shelf (z 1.15-1.19). Several failures took it to the top shelf or above it: s18 z 1.46, s19 1.45, s8 1.39-1.53, s4 1.5-1.6, s27 1.59-1.67. There, the arm is fully extended and the bowl is held at the shelf lip.

**Top-shelf variant: the foods spill first** (`undid_previous_stage`):
- s4: drumstick out ~785, lime ~845. The empty bowl then stayed wedged at the top until 1700.
- s8: drumstick ~805; the bowl rode on the open hand out of the fridge and fell at 917.
- s11: both foods 766-778, then the bowl.
- s27: lemon and drumstick 1133-1166, then the bowl at 1171, after ~270 steps of holding at the top.

**Scene dependence.** Layout 9 fails 10/12 (8 of them by this mechanism). There the base parks in front of the closed left (freezer) door, and the arm has to reach sideways into a narrow right compartment. Layout 10 fails 5/10 (s0, s2 here; s25, s26, s29 in mechanism 2).

**What successes do differently at this moment:**
- They lower the bowl onto a middle shelf, push it in to the depths above, and level it.
- They release only once the rack contact already holds: `at_target_at_release` is true in 11/11. Shelf contact comes 8-34 steps before release (30 in s9).
- s9 is the borderline case. It held the tilted bowl at an upper shelf front for ~500 steps (1000-1489) before finally pushing it on. It succeeded at 1563, only 137 steps before the timeout.

### 2. The bowl ends upright on a shelf with both foods, but the official check rejects it (4 failures: s21, s25, s26, s29)

These look like successes on video, and the policy never touched the bowl again (840-875 steps left).
- **s25, s26, s29 (L10, lower glass shelf, same z as the successes).** The bowl sits 3-15 cm shallower than in the five L10 successes: x 0.305 / 0.316 / 0.344 vs 0.18-0.28. The shallowest success is s7 at 0.276, so the threshold is in between.
  - `fridge.check_rack_contact` requires all 8 bbox corners inside a shelf region, with 5 cm tolerance.
  - In Refrigerator066 the shelf regions are much shallower than the interior: half-depth 0.247 m vs 0.38 m for `reg_main`. So the front ~10-17 cm of the physical glass shelf is not covered.
  - s26 has the largest bowl (bowl_9). A late accidental nudge at ~1470 moved it in only 1.3 cm.
- **s21 (L4, French door Refrigerator083).** The bowl is upright on the top shelf (z 1.485), foods in, gripper away from 880.
  - `get_reset_regions` drops shelf regions whose base is outside z 0.5-1.5 m, and `check_rack_contact` drops regions where the object centre is above 90 % of the region height.
  - The top shelf therefore never counts. The distractors were sampled onto the shelf below, at z ~1.10.
  - I did not compute which of the two filters applies, because the fridge's absolute z offset is unknown without the sim.

With a "standing upright on any shelf inside the fridge" criterion, the success rate would be 15/30 instead of 11/30.

### 3. The bowl is lost during the carry (2 failures: s1, s12)

- **s1 (L7).** The rim pinch held the bowl nearly vertical. The base stopped ~1.2 m short of the fridge for ~320 steps (820-1136) with no visible reason. Then the bowl slipped out of the closed fingers: width 0.019 -> 0.003 with no opening command. Afterwards the robot drove on and reached into the fridge empty-handed.
- **s12 (L8).** The close at 582 clamped the drumstick lying across the rim, and the bowl hung from it. The bowl slid off at ~880 near the fridge, while the drumstick stayed in the hand to the end.
  - Success s15 used the same drumstick-plus-rim grasp, and its bowl did not slip.

### 4. Food placement (1 failure: s10)

- The lime was released at 143 on the bowl's near rim (0.11 m from the centre) and rolled onto the counter at 158-176.
- The policy never looked back. It carried the bowl with only the drumstick and placed it correctly on a shelf (rack contact 805-1057).
- This is the only failure in stages 0/1. Missed food grasps happened (s2 broccoli twice, s17 bowl once) but were retried successfully.

The checklist's other failure types did not occur, or only as secondary effects:
- No fridge-door collisions that mattered. s28's base parked against the open door, which blocked the view and pushed the arm low.
- No stall at the fridge as a root cause, except s27.
- No time-outs.

## Official-criterion issues found

1. **The shelf region is stricter than the physical shelf (affects 4/19 failures; see mechanism 2).**
   - The front ~10-17 cm of the shelves in Refrigerator066 (and ~0.2 m in L9's Refrigerator031) is outside every shelf region.
   - The top shelf of Refrigerator083 is filtered out by height.
   - The task instruction ("put the bowl in the fridge") does not ask for any of this.
2. **"Food in bowl" = contact + xy distance only, with no height or upright test.**
   - Foods lying on the floor under or against a fallen bowl count as "in the bowl": s0 (to ~1632), s12, s19, s20, s22, s27 (to 1700).
   - This makes `first_incomplete_stage` 1 instead of 0 in s12, s19, s20, s22 and s27, and keeps stage 0 'complete' for a long time in s0.
3. **Many stage 0/1 `undone_events` during a tilted carry are artefacts**, in successes too (20-40 per episode).
   - The foods stay in the bowl, but the tilt pushes their xy offset past 0.7 x the radius, or they touch the fingers.
   - Only drops with a large `height_drop_after_release` are real spills.

## Packet / tooling issues

- **Bowl `dist_to_target` is measured to the whole fridge interior (`get_int_sites`, including `reg_main`), not to the qualifying shelf regions.**
  - So `dist 0.0` together with an empty `at_target` (s21, s25, s26, s29, and s22 at release) means "inside the fridge but not on a counted shelf". It looks contradictory unless you read the code.
  - A field for the qualifying-shelf distance, or for which region the bowl is in, would have saved a lot of time.
- **`holds` keep running after the fingers open when the bowl rests on the open hand.** In those cases `release_step` marks the fall, not the release:
  - s8: fingers open at 816-824, release_step 917.
  - s4: release_step 1144, with the gripper fully open from 944.
  - s19: touch continues to 819.
  - s20: second hold 864-927.
- **`close_events` missed a grasp:** s15's closing at 504-528 (0.080 -> 0.047), which carried the bowl and drumstick for 290 steps. The holds show it.
- **`typical_in_successes` for stage 1 includes a ~1 m base move in layouts where the drumstick plate is far from the bowl** (L8, L10). A long drumstick hold there is partly driving (s7, s12, s16, s28), not hovering.
- **Not a measurement bug, but worth knowing:** the 15 diagnosis agents share one scratchpad directory, and another agent's `summ.py` overwrote mine early on. I moved to a private subfolder.

## Uncertainty

- **s25, s26, s29.** The front-overhang explanation is inferred from end positions vs the successes and from the region sizes in the fridge XML; I did not re-run `check_rack_contact`.
- **s21.** Which height filter excludes the top shelf is an inference.
- **s3 and s28 (medium).**
  - s3: the shelf geometry at the release point is hard to see.
  - s28: the agent view is blocked by the door, and the bowl's landing spot (z 0.49) is not visible.
- **s1.** Why the base stopped for 320 steps is unknown (no visible contact).
- **Deliberate opening vs. slip.** Whether some openings were deliberate releases or the grip going slack (s0, s8, s27) cannot be told from the width trace alone. The outcome is the same in each case.
