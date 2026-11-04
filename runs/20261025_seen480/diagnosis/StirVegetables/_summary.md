# StirVegetables: stage-level diagnosis (GR00T N1.5, 30 episodes, 9 succeeded, 21 failed)

Task: put two vegetables into the pot on a lit burner (stages 0 and 1, in either order), pick up the spatula (stage 2), and stir (stage 3). Horizon 1600 steps.

Files:
- Per-episode files: `StirVegetables_s<N>_single.json`.
- Contact sheets: `frames/`. `frames/cmp_stir_success.png` and `frames/cmp_stir_fail.png` show the stir moment of every episode that reached stage 3 with the spatula in hand.
- Running notes: `_notes.md`.

## How the stage-3 check actually works (env source, `stir_vegetables.py`)

`success_time` goes up by 1 on every step where all of these hold:
- The spatula is grasped: fingers closed and touching it.
- The pot touches the stove and is within 0.15 m of the task burner, which is lit at reset.
- Both vegetables are "in the pot": touching the pot, and within 0.7 × the pot radius horizontally.
- The spatula touches **veg1**.
- **Both** vegetables moved more than 0.5 mm horizontally since the previous grasped step.

Success needs 5 such steps in total; they do not have to be consecutive. The contact flag is not reset per object, so veg2 never has to touch the spatula; it only has to move. Pushing only veg2 never counts.

## Counts

The packet's `first_incomplete_stage` is 2 in eight episodes only because PickUpSpatula is evaluated as "held and lifted now", not latched. By the online log, those episodes completed stage 2 and failed in stage 3 (`failed_stage` = 3 in their JSONs).

| root-cause stage | episodes |
|---|---|
| 1 (second vegetable) | 1: s10 |
| 2 (spatula) | 4: s2, s6, s20, s9 |
| 3 (stir) | 16: s0, s1, s3, s8, s12, s13, s14, s15, s16, s17, s19, s22, s24, s25, s26, s28 |

| primary category | episodes |
|---|---|
| other: stir misdirected (the blade stays beside the pot or on its rim and pushes the pot, never sweeping the vegetables) | 6: s1, s3, s8, s16, s22, s25 |
| other: the spatula slipped out of the fingers while the blade was pushed into the pot | 5: s12, s13, s15, s17, s24 |
| other: the hand or spatula caught the pot and lifted or tipped it off the burner | 4: s0, s14, s26, s28 |
| missed_grasp | 4: s2, s6, s20 (spatula); s10 (broccoli) |
| skipped_or_wrong_stage | 1: s9 (took the spatula before the second vegetable and put it into the pot) |
| other: blade pointed up while hovering for 590 steps; the stir started late, reached 3 of 5 counted steps, then the pot was dragged off the burner | 1: s19 |

Near misses on the stir counter: s19 (3/5), s15 (2/5), s17 (1/5). In all the other stage-3 failures the counter stayed at 0.

Time was not the limit. Every stage-3 failure held the spatula for 250-1140 steps, against a median stir of 315. s14 is the closest case: a slow stage 0 left about 440 steps.

## Recurring failure mechanisms

### 1. Holding the spatula without getting the blade among the vegetables (10 episodes: the 6 misdirected stirs plus 4 pot lifts)

**Long hover before the descent.** After lifting the spatula, the policy often holds it with the blade pointing sideways or up, at the wall or above the hand, for hundreds of steps:
- s1: 640-1020
- s12: 680-900
- s19: 560-1150
- s26: 880-1040
- s28: 880-1080

**The descent then lands in the wrong place.** When the arm finally comes down, it lands on the pot's near (right) side:
- Blade outside the pot wall:
  - s1: 1040-1600
  - s22: 550-1300, beside the pot over the exposed flame
  - s25: 740-860
- Blade on the right rim of a tall or deep pot:
  - s3: 1060-1600
  - s8: 1100-1600
  - s16: 840-1600
- In every one of these the pot is shoved off the burner centre; the flame or glowing element becomes visible beside it.

**The pot gets lifted or tipped.** In four episodes the hand or the wedged blade lifts or tips the pot:
- s0: the pot is carried ~10 cm above the burner, 748-890.
- s14: lifted 1320-1600.
- s26: tipped and raised 1060-1520.
- s28: pushed back and tilted 1100-1600.

**Successes look different at this moment.** The blade goes nearly vertical and straight down into the middle of the pot:
- s5: 600-620, 60 steps after arriving.
- s21: hanging into the tall stockpot, 820-928.
- s18: from above, 620-900.

A few local pushes then count within ~100-350 steps. The slow successes s11 (516) and s23 (920) also kept the blade inside the pot the whole time.

### 2. Thin or long spatula slides out of the fingers during the stir, and is never re-grasped (5 primary, plus s0 and s22 as late consequences)

| episode | slip step | gripper width (sum of finger joints) | what pushed the handle |
|---|---|---|---|
| s12 | 926 | 0.022 → 0.001 | blade pushed into the pot |
| s15 | ~920 | 0.019 → 0.003 | blade pushed into the pot |
| s17 | ~1428 | 0.017 → 0.002 | blade pushed into the pot |
| s24 | ~940 | 0.010 → 0.003 | blade jammed into the pot |
| s13 | 1259 | → 0.002 | hand and blade pressed on the rim |
| s22 | 1301 | 0.024 → 0.007 slowly over 800 steps, then dropped | handle rotated in the fingers |

- The handle rotates and slides through the fingers when the blade meets resistance.
- The spatula is often held at the very end of the handle: s12, s13, s16, s25. That gives a long lever. Successes s18 and s4 also held it at the end, though, so this is a risk factor and not a sole cause.
- Gripper width while holding is fixed by the spatula model (spatula_9 ≈ 0.010, spatula_6 and spatula_8 ≈ 0.017, spatula_5 and spatula_7 ≈ 0.028). It does not separate successes from failures.

**No recovery after the spatula is lost.**
- Across all 8 episodes where the spatula was lost after lifting (s0, s6, s12, s13, s15, s17, s22, s24), it was re-grasped 0 times.
- In 7 of the 8 (all except s13), the fingers stay fully closed (≈ 0.001) for the remaining 300-840 steps, and the arm keeps hovering or "stirring" over the pot empty-handed.
- s13 reopens 85 steps after the slip and then wanders away.

### 3. Spatula grasp misses at the ends of the spatula (stage 2: s2, s6, s20)

All three descents targeted an end of the spatula rather than mid-handle:
- s2: tip of the thin stick handle of spatula_9, close at 706 (width 0.014).
- s20: butt end of spatula_8, closes at 426 and 457.
- s6: the flat paddle of spatula_5, close at 742. It lifted for 5 steps, then slid out.

After the miss, the policy continues with the stir motion empty-handed. s2 and s20 make further closes on air, or pass over the spatula without opening. Successful grasps were at mid-handle or at the handle end with the handle across the finger gap.

### 4. Closed-gripper persistence after any miss (cross-cutting)

The same "fingers stay closed after a miss" behaviour appears in stages 0-1:
- s10: closed on air beside the broccoli at 523. It never returned; went to the microwave, then parked by the pot.
- Successes s7 and s29 lost 230-350 steps to it: s7 closed 240-585, s29 closed 368-592. They recovered only when the fingers reopened by themselves.

This makes stage 1 slow in successes (s7: 616 steps, s29: 535, median 282).

### Smaller items

- **s9, out of order.** It grasped the spatula after only one eggplant (356), placed it into the pot like a vegetable (~590), fetched the second eggplant (1007), then wandered with an open gripper.
- **Reach to the far counter.** In layouts where the vegetables or spatula sit on the counter on the far side of the range from the pot, early stages are slow or fail:
  - s10: broccoli 0.84 m from the pot, missed.
  - s14: the garlic was released 0.29 m short of the pot (494), so stage 0 took 663 steps.
  - s17: hovered 340 steps over the spatula before closing, so stage 2 took 525 steps.

## Scene dependence (small n, confounded)

| pot model | successes | episodes |
|---|---|---|
| Pot075 (brown/light steel) | 5/8 | s5, s7, s18, s23, s29 succeeded; s2, s8, s17 failed |
| Pot065 (red cast iron) | 3/8 | s4, s11, s27 succeeded |
| Pot074 (tall black stockpot) | 1/4 | s21 succeeded |
| Pot069 (steel) | 0/3 | |
| Pot071 (deep black pot) | 0/7 | 6 of the 7 reached stage 3 with the spatula in hand |

Spatula models: spatula_8 succeeded 4/9; the others 1-2 each.
- spatula_9 (thin stick): two of its failures were grasp or slip failures (s2, s24); spatula_5 also has two (s6, s9).
- But s21 succeeded with spatula_9, and s5 with spatula_5.

Layout 9 succeeded in 1/7 episodes. I cannot separate the pot model from the layout and lighting; this is a hypothesis.

## Official-criterion and packet issues

1. **The stir check is asymmetric.** Only veg1 must touch the spatula; veg2 only has to move by 0.5 mm. Pushing only veg2 can never count. s15's blade was on the mushroom (veg2) when the spatula slipped.
2. **The stir check rewards shaking the pot.** The movement threshold is 0.5 mm per step, and the spatula can touch with any part. Success was logged while the hand was tipping or lifting the pot, with the spatula inside:
   - s27: 1036-1072
   - s29: 1052-1100. The pot hangs tilted above the element in `s29_tilt.png`; it presumably still touched the cooktop at the counted steps.
   - s4 succeeded right after tipping the pot (~800).
   So part of the "stir" credit comes from jostling the pot, not stirring.
3. **"In pot" needs veg-pot contact, and it is pot-relative.**
   - Stage 0/1 conditions flicker whenever a vegetable rests on the other one or bounces; most `undone_events` in this task are such flickers.
   - Stage 0/1 stay complete while the pot is lifted off the stove (s26). Only the stage-3 pot-on-burner clause catches it, and the packet has no pot pose.
4. **PickUpSpatula false positives.** The check counts "held and lifted" whenever closed fingers touch a spatula that rests above the counter:
   - s9 921-1003: the gripper was holding the eggplant and touched the spatula lying in the pot.
   - s0 900-911 and 1250-1269: closed fingers brushing the spatula on the pot rim.
   - s20 439-448: a pinched handle end with no real lift.
5. **The online log credits stage 2 from a 5-step lift** in s6 (752), so by the online log s6 "failed stage 3".
6. **Missing fields.** The packet has no pot position or pot-stove contact, no spatula-vegetable contact, and no per-step `success_time`.
   - Pot displacement, seen in 13 of the 16 stage-3 failures (all except s12, s15, s17), had to be read from the exposed flame.
   - Which steps counted (s19, s29) cannot be determined.
   - Suggest adding: pot distance to the task burner, pot-stove contact, spatula contact with veg1/veg2, and the per-step counter.
7. **Replay vs online.**
   - s7: replay `success_time` is 4, but the online log records a success at 976 (the known case).
   - s4 and s8: the `replay_matches_online_log` flag is false only because of 1-step condition flickers at the logged step (s4 stage 2 at 576; s8 stage 1 at 544). These are not real disagreements.
8. **Other quirks.**
   - `steps_since_previous_stage` is negative for s9 stage 2 (-637), because the order-free evaluation completed it before stage 1.
   - `tipped_over_intervals` are not meaningful for round vegetables (e.g. s11's avocado is "tipped" from step 8).

## Uncertainty

- Where the blade was during long occluded stretches: s16 840-1600, s25 1200-1600, s26 1200-1520 (low confidence). These might be blade-inside-touching-only-veg2 cases rather than blade-outside.
- Whether the pot left the 0.15 m radius or lost stove contact is inferred from the video (exposed flame, raised pot), never measured.
- The pot-model effect (Pot071 0/7) is confounded with layout and could be chance.
