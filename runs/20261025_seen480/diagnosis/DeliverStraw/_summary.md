# DeliverStraw: stage-level diagnosis (GR00T N1.5, 30 episodes, 2 succeeded, 28 failed)

Task: open the top drawer (stage 0), take the straw out, drive to the dining counter and put the straw in the glass cup (stage 1). Horizon 1700 steps.
Per-episode files: `DeliverStraw_s<N>_single.json`. Contact sheets: `frames/`. Cross-episode sheets:
- `frames/all_handle_close_wrist.png`: wrist view at the first handle close, all episodes.
- `frames/all_at_cup_wrist.png`: wrist view just before release at the cup.
- `frames/all_start_left.png`, `frames/L10_start_left.png`, `frames/L10_drawer_open_left.png`: scene comparisons.

## Counts

| root-cause stage | episodes |
|---|---|
| 0 (open the drawer) | 16: s0, s1, s4, s6, s7, s8, s10, s11, s13, s14, s16, s18, s20, s22, s23, s28 |
| 1 (straw to the cup) | 12: s2, s3, s5, s9, s12, s15, s19, s21, s24, s25, s26, s29 |

s23 failed in stage 1 by the packet, but its root cause is stage 0. Stage 0 finished at 1055 (median 192), and the straw was still 1.48 m from the cup at the timeout.

| primary category | episodes |
|---|---|
| wrong_fixture_element (wrong drawer or cabinet door) | 7: s6, s13, s16, s18, s20, s22, s28 |
| actuation_insufficient (handle slipped out mid-pull) | 4: s0 (0.853), s7 (0.38), s8 (0.39), s23 (0.38, then 860 steps to reopen) |
| missed_grasp (closed on air at the handle, then carried on empty-handed) | 2: s4, s11 |
| stall_at_fixture | 2: s1, s14 |
| stall_approach_retreat | 1: s10 |
| stall_holding_at_target (straw held beside the cup, then released off it) | 5: s2, s12, s19, s21, s29 |
| placed_then_fell_or_slid | 2: s5, s15 |
| dropped_in_transit | 2: s9 (at the drawer), s25 (during the drive) |
| placed_off_target | 1: s3 |
| wander_away (never drove to the dining counter) | 1: s26 |
| other: base parked ~1.4 m short of the cup | 1: s24 |

Secondary categories that recur:
- missed_grasp_continued_empty: s4, s9, s11, s13, s20. In s4 and s9 the robot drove to the cup with an empty gripper.
- ran_out_of_time: s23, plus partly s19 (slow drawer and slow straw pick).

## Recurring failure mechanisms

### 1. The policy never gets the target drawer open (16 of 28 failures)

Three sub-patterns:

- **Wrong fixture (7).** The policy grasps a neighbouring handle and opens that fixture instead.
  - Cabinet doors: s6 (tall door, 169-300), s16 (door, 213-360), s18 (double doors below the drawer, closes on air at 230 and 437).
  - Another drawer: s13 and s20 (the drawer on the perpendicular wall of the layout-10 corner), s22, s28 (opened at 861 after hovering 200-850).
  - The policy then searches the empty fixture for hundreds of steps and never moves on to the target drawer. None of this shows up in the packet: the target drawer reads 0 with no touches. The only trace is an `on_nothing` close with width 0.03-0.05.
  - It is not purely a scene effect. s13 and s20 opened the wrong drawer, while s15 and s29 opened the correct one, with the same layout and the same drawer position (straw x ≈ 1.4).
- **Handle slips out mid-pull (4).** On a long horizontal bar handle the policy pulls with the bar across a half-open finger gap (hooked, not enclosed), and the handle slips out partway.
  - s0: slipped at 320 with the drawer at 0.853.
  - s7: slipped at 279, drawer at 0.38.
  - s8: handle post off-centre, slipped at 266, drawer at 0.39.
  - s23: slipped at 174, drawer at 0.38.

  In the two successes the handle was centred in the finger gap, and the first pull took the drawer to ≥ 1.0. Only s23 came back to the drawer: it hooked it from inside three times (522-639, 806-850, 1021-1074) and got it open at 1055. That was 860 steps late, and the episode timed out with the straw in hand, 1.48 m from the cup. The others did not retry:
  - s0 kept its fingers closed and hovered over the straw (480-780, closest approach 0.30 m), then wandered.
  - s7 hovered over the 10 cm gap with an open gripper for ~1400 steps.
  - s8 came back twice (1047-1068, 1265-1278), but only pushed the drawer front.

  None of the four tried to pick the straw through the gap.
- **Never closes on the handle (5).** s1 never closed at all. s14 pressed flat on the drawer face for 240 steps. s10 aligned with the handle but retreated. s4 and s11 closed on air beside the handle. After a close on air the fingers usually stay closed for the rest of the episode (s0, s4, s11, s14, s20), and the policy goes on with the next motions. s4 and s11 drove to the dining counter with nothing in the gripper.

The answers to the partial-opening questions are in the JSONs:
- s0: 0.853, pulled by hooking the bar handle.
- s7: 0.38, hooking the bar.
- s8: 0.39, off-centre partial grasp.
- s23: 0.38, then hooked from inside.

### 2. Straw brought to the cup but not put in (8 of 12 stage-1 failures)

In both successes the straw is held vertically directly over the cup, lowered, and released within ~40 steps of arriving (s17 at 743, 3.4 cm from the cup centre; s27 at 778, 0.5 cm). The failures arrive on time but then:

- **Hover beside the cup, then release away from it.** The straw stays a few cm beside or over the cup and moves around without contacting it:
  - s2: 700-870, released 15 cm off.
  - s12: 960-1260, two 2-step rim contacts, released 23 cm off and fell to the floor.
  - s19: 1380-1600, released 54 cm off and fell to the floor.
  - s21: 1140-1344, released 6 cm off.
  - s29: 680-910, released at the counter edge and fell to the floor.

  Then the arm moves off and opens. The release is usually over the counter edge, so the straw ends on the floor or a stool and cannot be retrieved.
- **Released at the rim, then it falls.**
  - s3: released at 797, 5.3 cm off and outside the rim. The policy re-grasped it at 805 and then dropped it onto a stool at 884.
  - s5: released at 977, 2.5 cm off, touched the cup for 2 steps, then fell off the counter.
  - s15: the straw was carried at ~30-45° from horizontal. It lay across the rim 795-836 and slid off after the 826 release.

Hypothesis, not tested: the wrist camera looks forward along the straw, and the cup is usually not visible below the fingers (`all_at_cup_wrist.png`). Fine alignment therefore relies on the agent view, where the transparent cup and the thin straw are a few pixels wide. The rim-side releases are 2.5-6 cm from the cup centre, roughly the cup's radius.

### 3. Losing the straw or the way (4)

- **Thin-straw grasp fails during transport.**
  - s9 grasped the straw by its tip (width 0.043). It slipped out at 331-337 at the drawer and fell to the floor. The robot then delivered nothing to the cup (closes on air at 886 and 1515).
  - In s25 the finger width shrank steadily, 0.016 → 0.001 over 374-822, as the straw slid through the fingers. It dropped mid-drive, 1.36 m from the cup.
  - In s23 the width also went to 0.001 during the hold, but the straw stayed in the gripper.
- **Navigation.**
  - s24's base stopped at ~640 at the near end of the island, ~1.4 m from the cup, and never moved again. The arm put the straw on the bare counter and re-grasped it three times.
  - s26 held the straw at the drawer for 300 steps, then turned toward the fridge and dropped the straw at 1636.

## Scene grouping

| layout | episodes | outcome |
|---|---|---|
| 10 | s0, s2, s7, s13, s15, s18, s20, s26, s29 | 0/9 success. 5 stage-0 failures: 2 bar-handle slips, 2 perpendicular-drawer, 1 door. 4 stage-1 failures |
| 9 | s3, s4, s5, s8, s10, s17, s22 | 1/7 success. 4 stage-0 failures, 2 stage-1 failures |
| 2 | s11, s14, s23, s27 | 1/4 success. 3 stage-0 failures (long bar handles) |
| 7 | s1, s6, s9, s24 | 0/4 success. 2 stage-0, 2 stage-1 |
| 8 | s12, s16, s19, s25, s28 | 0/5 success. 2 stage-0 (door, wrong drawer), 3 stage-1 |
| 4 | s21 | 0/1 success. Stage 1 |

- The two successes are in layouts 9 and 2.
- Layout 10 has every stage-0 sub-pattern, so I see no single scene-specific cause.
- Straw model does not separate the outcomes. Straw009 and Straw012 each appear in one success and in several failures.

## Official-criterion and packet issues

1. **The composite success does not check the drawer.** `DeliverStraw._check_success` requires only that the straw touches the cup within 0.5 m horizontally and that the gripper is far away. Stage 0 (≥ 0.95) comes from the subtask definition, not from success.
   - In s0 the straw was visible at 0.853, so the drawer was probably open enough. What failed was the policy: it never tried the straw.
2. **The drawer "fraction" is not a fraction of the travel.** The value is joint / (0.55 × depth). Fully open reads ~1.82, and the successes reach 1.6-1.82. The 0.95 threshold is ~52 % of the physical travel. So s0's "85 %" is ~47 % of the travel, and s7/s8's 0.36-0.39 is ~20 %.
3. **The success condition is very loose.** It requires contact with the cup and a centre within 0.5 m, about 10× the cup's radius, with no inside or upright test. In s15 the condition held 795-836 with the straw lying across the rim. Only the gripper-far clause prevented a false success. A straw leaning on the outside of the cup would also count.
4. **The straw tilt fields are inverted in meaning.** Tilt is measured from the straw lying flat, so `tipped_over` (> 45°) marks the straw being held or standing vertically, including in both successes.
   - In s21, `tipped_over` ends at 1095 although the straw is still held near-vertical in the video until ~1344. The tilt field is unreliable for this object; this may be a roll ambiguity rather than a tooling bug.
5. **Gripper width ≈ 0.001 does not mean "closed on nothing" for this straw.** s23 was still holding the straw at width 0.001, and s25 reached it just before the drop.
6. **Wrong fixtures are invisible in the packet.** It tracks only the target drawer, so s6, s13, s16, s18, s20, s22 and s28 look like "never touched the drawer". The wrong fixture can only be seen in the video.
7. The official labeler outputs a collapsed or meaningless segmentation for the 12 episodes where the drawer never moved (expected; context only).

I found no case where a packet measurement contradicts the video in a way that points to a bug in the measurement code, apart from the field semantics in items 2 and 4.

## Uncertainty

- **Depth at the cup** (s2, s21, s29) cannot be judged from the agent view. "Beside the cup" rests on the absence of contact plus the 2-D view.
- **s18 and s28:** which handle the policy was actually aiming at is uncertain (medium confidence).
- **s10 and s1:** the reason for retreating without closing is unknown.
- **s19:** I classified it as hovering at the cup, but ~600 steps were also lost upstream (slow drawer, missed first straw grasp). With more time it might or might not have succeeded.
- **s24 and s26:** why the navigation stopped or went the wrong way is not visible. The cup was not salient from s24's parking spot.
- **Successes:** only 2 exist, so the "typical" stage durations (192 / 589) are fragile.
