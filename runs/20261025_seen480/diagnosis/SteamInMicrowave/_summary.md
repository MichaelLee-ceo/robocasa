# SteamInMicrowave: stage-level diagnosis (GR00T N1.5, 30 episodes, 10 succeeded, 20 failed)

Task: pick the vegetable from the sink and put it in a bowl on the counter (stage 0). Carry the bowl to the microwave, which is mounted high (bowl ends at z ≈ 1.4-1.5 m), and release it inside (stage 1). Close the door (stage 2). Press START (stage 3). Horizon 1400 steps.

Per-episode files: `SteamInMicrowave_s<N>_single.json`. Running notes: `_notes.md`. Contact sheets: `frames/`.

Key comparison sheets:
- `frames/cmp_press_success_wrist.png` and `frames/cmp_press_fail_wrist.png`: wrist view at the START press in all 10 successes, and at the closest press or approach in 10 failures.
- `frames/cmp_L2_door.png`: door approach in layout 2.
- `frames/cmp_L8_afterplace.png`: the moment after the bowl is placed, in layout 8.

## Counts

| root-cause stage | episodes |
|---|---|
| 0 (vegetable into bowl) | 2: s12, s20 |
| 1 (bowl into microwave) | 6: s1, s6, s10, s16, s22, s28 |
| 2 (close door) | 6: s8, s14, s23, s25, s27, s29 |
| 3 (press START) | 6: s0, s2, s5, s7, s11, s15 |

The packet's first incomplete stage differs from the root cause in s6, s16 and s28. All three show stage 0 incomplete because the vegetable fell out of the bowl while the bowl was being carried, which is a stage-1 event. In s6 and s28 the later stages were then done with an empty bowl.

| primary category | episodes |
|---|---|
| wrong_fixture_element (presses land next to START) | 4: s2, s5, s11, s15 |
| undid_previous_stage (vegetable spilled from the carried bowl) | 3: s6, s16, s28 |
| actuation_insufficient (door left at 0.51 / 0.85 / 0.94 closedness) | 3: s14, s25, s29 |
| stall_approach_retreat (at or near the keypad, no press) | 2: s0, s7 |
| missed_grasp / missed_grasp_continued_empty (vegetable in sink) | 2: s12, s20 |
| placed_then_fell_or_slid (bowl slid out of the cavity onto the range) | 1: s1 |
| dropped_in_transit (bowl fell to the floor) | 1: s10 |
| placed_off_target (bowl "inside" visually, official check fails) | 1: s22 (low confidence) |
| skipped_or_wrong_stage (went to keypad, never touched door) | 1: s23 |
| stall_at_fixture (parked below the open door) | 1: s27 |
| other: pushed the door from its inner side, against its stop | 1: s8 |

Time was not the limiting factor. The only abnormally slow earlier stage among the failures is s23's stage 1 (665 steps, p90 596), and s23 still had 508 steps left. The six stage-3 failures had 307-649 steps left after the door closed; the median stage 3 in successes is 169 steps (p90 345).

## Recurring failure mechanisms

### 1. Pressing START: the press lands one key or a few centimetres off, and is not corrected (6 root causes; 9 of 20 failures miss START at some point)

The policy reaches the keypad in almost every episode that gets there, but the start_button geom is small. Seen in the wrist view:

| episode | layout / microwave | what happened |
|---|---|---|
| s2 | L9 | about 10 presses (873-1361) on the number keys and panel border, several key widths from START |
| s11 | L2 | 3 presses (828-892) on KITCHEN TIMER, one key from START/ENTER |
| s15 | L10 | 5 presses (1021-1095) on Clock Time / Timed Defrost, diagonally next to the round Start key |
| s5 | L5, MW036 | 2 presses on the bezel just below the Start / Stop row |
| s1, s6 | L5, MW036 | same near-miss below the Start row, as a secondary error |
| s29 | L10 | 3 presses around Start (1064-1177), as a secondary error |
| s7 | L10 | hovered a few cm in front of the number keys (1116-1152), never touched, then drifted away |
| s0 | L9 | retracted, made one reach to the door glass, and never touched the keypad after 1052 |

In 3 of 4 MW036 episodes the presses landed below the row (s1, s5, s6). Success s9 hit START on the same panel.

- **The policy does not search after a miss.** It re-presses at about the same spot, shifts slightly, or drifts away and hovers.
- **Successes miss too.**
  - s4 first pressed KITCHEN TIMER at 1233, then hit START at 1381, 19 steps before the timeout.
  - s26's first contact at 1179 missed; its second, at 1214, hit START.
  - s19's START press was a by-product of the door-closing sweep.
- **Wrist roll (weak signal).** In `cmp_press_*`, 8 of the 9 successes I could judge see the keypad roughly upright with START at the fingertips; s19's frame shows only the door. Several near-miss failures see it rotated about 70-90 deg (s2, s7, s11, s15, s29). Success s9 is also rotated, and the L5 failures look the same as s9, so wrist roll is at most a contributing factor.

### 2. Door not fully closed, or not engaged at all (6 root causes)

- **Incomplete close, then moving on.** The policy pushes the door most of the way and then goes to the keypad as if the door were shut:
  - s25: 0.85 closedness. START was pressed at 848 with the door ajar (it latched), then STOP was pressed at ~1232.
  - s29: 0.94.
  - s14: 0.51, after ~570 steps of weak pushing at the free edge.
  - Successes close the door to 1.0 in one sweep of 60-200 steps.
- **Pushing the door the wrong way.** In 7 of 25 episodes that touched the door, the first contact swung it fully open (closedness to ~0).
  - Recovered within 32-80 steps: s1, s5, s15, s20, s28.
  - s19 (a success) pushed against the open door for ~530 steps (720-1248) and only closed it at 1372.
  - s8 never recovered. Its gripper went in between the cavity and the open door and pressed the door's inner face against its stop from 730 to 1051.
- **Never engaging the door.**
  - s23 went straight from the bowl to the keypad, with no microwave contact in the whole episode.
  - s27 parked below the open door's edge for 500 steps.
  - s22 retreated low-left below the door. Success s24 in the same layout did the same retreat for ~380 steps (760-1110) but came back.

### 3. Carrying the bowl to the high microwave: rim grasp plus large tilt (5 root causes)

The bowl is grasped by its rim and has to be lifted 0.55-0.73 m. It tips past 45 deg during insertion in about half of all episodes, successes included (s9, s18, s21, s24, s26). Outcomes:

- **Vegetable spilled.**
  - s28: a round tomato rolled out at 662 while the bowl was at 59 deg.
  - s6: a lemon wedge placed near the rim slid out at 412-428 as the bowl was lifted by that rim.
  - s16: the bowl was twisted onto its edge over the sink right after the grasp (tilt up to 130 deg); the garlic fell into the sink at ~560, and the policy fumbled for about 700 steps.
- **Bowl lost.**
  - s10: the bowl slipped out of a tilted rim grasp at 348 and fell to the floor.
  - s1: released while still tilted at the cavity lip (648); it slid out and fell onto the range.
- **The policy never notices a lost object.**
  - In s6, s20 and s28 it went on to put the empty bowl in and close the door.
  - In s20 and s28 it also turned the microwave on, so everything except "vegetable in bowl" held at the end.
  - In s1 it closed the empty microwave and pressed (off START).

### 4. Small vegetables in the sink (2 root causes)

- **s12 (lime):** four closes on air (147-425). A fifth grasp held the lime at the fingertips and it slipped back into the sink at 637. The policy then carried the empty bowl.
- **s20 (mushroom):** a single close beside the mushroom at ~97, with no retry.

Every other episode grasped the vegetable on the first close.

### Scene dependence (small n)

| layout | successes / episodes | failures |
|---|---|---|
| L10 (microwave on a tall cabinet, MW073) | 0/3 | s7, s15: START misses; s29: door at 0.94 plus START misses |
| L7 | 0/2 | s12, s28 |
| L5 (over-the-range MW036) | 1/4 | s1, s5, s6 all miss START low |
| L2 (built-in, door opens toward an obliquely parked base) | 1/4 | s8, s14: door failures; s11: START miss |
| L9 | 3/6 | |
| L8 | 2/4 | |
| L4 | 2/3 | |
| L6 | 1/3 | |
| L1 | 0/1 | |

## What successful episodes do differently

- **Door.** Successes reach the outer face or free edge of the door and close it to 1.0 in one sweep.
  - In layout 8 (s4, s17) they reach far to the left to the door's outer face; s23 in the same layout went right, to the keypad.
- **Press.** The fingertip lands on START on the first or second press. The keypad is usually upright in the wrist view.
- **Vegetable in the bowl.** It sits near the bowl centre (0.01-0.06 m). The spill in s6 started from a release 0.11 m off centre.
- **Slow stages that still succeeded.**
  - Stage 1 in s4 and s13: a failed first bowl close cost ~150-165 steps.
  - Stage 0 in s17: the mushroom was carried up and away first.
  - Stage 2 in s19 and s24: see mechanism 2.
  - Stage 3 in s4 and s21 (344 and 354 steps).
- **Finishing late.** s19 (28 steps spare), s4 and s24 (19 steps spare each) finished within the last 30 steps, so the success rate is fragile with respect to the horizon.

## Official-criterion and packet issues

1. **turned_on latches with the door ajar.** `Microwave.update_state` only blocks turn-on when the door is at least 90 % open, and the latch survives closing. So "press START, then close the door" counts as success.
   - s25 pressed START at 848 with the door 15 % open (stage-3 fixture value 1 while stage 2 was false); STOP was pressed later.
   - In s19 the press happened during the closing sweep.
   - This produced no false successes here.
2. **No upright check on the bowl in `check_obj_in_receptacle`.** It needs only contact and a horizontal distance below 0.7 × the bowl radius.
   - s16: stage 0 stayed "complete" 404-560 with the bowl standing on its edge.
   - s10: stage 0 was "complete" again 1296-1400 with the corn and the bowl both on the floor.
3. **s22: possible false negative of the bowl-inside check.** The bowl sits on the turntable and the packet puts its centre inside (distance 0.0), but the bbox-corner check never held. Given the large tolerance (th/|edge|, per the coordinator), a corner would have to be well outside the box, so I cannot tell whether this is a bbox artifact (bowl_9 in MW083) or real protrusion. It needs a state replay.
4. **`holds` and `height_drop_after_release_m` for the vegetable count fingertip contacts during the bowl carry as grasps.**
   - Affected: s7, s8, s9, s11, s13, s24, s25, s26.
   - Negative drops follow: s8 -0.372, s11 -0.314, s25 -0.332.
   - Likewise, the one-step stage-0 undo blips during the carry are the vegetable rattling in the bowl, not real undos.
5. **Button presses cannot be judged from the packet.**
   - The microwave's `gripper_touch_intervals` are identical for stages 2 and 3, so door contacts and keypad contacts look the same.
   - There is no start-button contact flag and no eef-to-button distance; every wrong-key judgment above comes from the wrist video.
   - Suggest logging `check_contact(gripper, start_button)` and `gripper_button_far` distances.
6. **Minor.** The s21 success shows stage 3 `complete_at_end=false` (a last-chunk drop, as the brief warns). The official labeler is meaningless on s20 (a "place" at step 0).

## Packet vs video

I found no disagreement that points to a measurement bug. Every hold, release, door value and turn-on event I checked matched the video.

- The one contact I could not see at first, s27's door contact at 1388-1392, is plausible: the fingers are next to the door edge in the wrist view.
- In s22, "distance 0" next to "not at target" is a difference between two definitions (centre vs bbox corners), not a bug.
- My very first packet summary was produced by another agent's same-named helper script in the shared scratchpad. I recognized it was for a different task and discarded it; all results here come from my own scripts, kept in a private subfolder and checked unmodified.

## Uncertainty

- **Low confidence: s22.** Off-target placement vs criterion artifact.
- **Medium confidence:**
  - s0 and s7: whether these were stalled approaches or wanders.
  - s5: the press offset is a 2-3 cm reading from the wrist view.
  - s8: which face of the door was pushed, inferred from the door value and an oblique view.
  - s14: why the push was so weak.
  - s16: why the bowl was twisted.
  - s27: why the arm parked.
- **Key identities** in the near-miss presses are read from 4-step wrist frames, and offsets are estimates in key widths.
- **Light checks.** The successes were checked from the packet plus an overview sheet, except s19, s4, s9, s18 and s26, which got event sheets.
