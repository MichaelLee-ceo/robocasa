# GetToastedBread: stage-level diagnosis (GR00T N1.5, 30 episodes, 0 succeeded)

**Task.** "Start the toaster. Once the lever pops up, take the bread to the plate on the dining counter." Horizon 2000 steps. There are two stages:
- Stage 0: lever pushed down with the bread in that slot pair, then popped back up.
- Stage 1: bread on the plate and the gripper away.

**Files.** Per-episode results are in `GetToastedBread_s<N>_single.json` and contact sheets in `frames/`. Running notes are in `_notes.md`.

**Extra measurement.** I read the bread's position from `run/.../_steps.npz` by finding the bread's free joint in the saved states (its start position matches `start_pos`). This was CPU-only. The bread carriage moves with the lever, so the bread's height shows whether the pressed lever belonged to the bread's slot pair.

## Toaster mechanics I relied on (code read, not modified)

- **Switching on.** A slot pair turns on when its lever reaches >= 0.9. The lever is then held at 1.0 for 500 steps and springs back up slowly, taking about 150-170 steps.
  - So the earliest possible stage 0 is about 820 steps, and toasting plus pop-up takes about 670 of the 2000 steps.
  - Lever value 1.0 means pushed down (on); 0 means up.
- **Visible effect.** When the pressed pair holds the bread, the bread drops 5-7 cm with the carriage. When the bread does not move while a lever is at 1.0, the pressed lever belongs to the other pair.
- **Toaster types.**
  - Layouts 2 and 10 have 4-slot, two-pair toasters (17 episodes).
  - Layouts 4, 8 and 9 have 2-slot, single-pair toasters (13 episodes).
- **Toasting credit.** The env's `_check_success` gives toasting credit only if the bread touches the slot floor of the pair that is on. The stage-0 checker also requires the pop-up (lever <= 0.1).

## Counts

| root-cause stage | episodes |
|---|---|
| 0: start the toaster | 22: s0, s2, s3, s5, s6, s7, s8, s11, s13, s14, s15, s16, s19, s20, s21, s22, s23, s24, s26, s27, s28, s29 |
| 1: take the bread to the plate | 8: s1, s4, s9, s10, s12, s17, s18, s25. All eight had stage 0 complete per the online log, and replay agrees. |

| primary category | episodes |
|---|---|
| skipped_or_wrong_stage | 9. In stage 0, went for the bread instead of the lever: s3, s8, s11, s13, s26, s29. In stage 1, drove off without the bread: s4, s10, s12. |
| actuation_insufficient | 8: s2, s5, s16, s19, s20, s21, s24, s28 |
| wrong_fixture_element (lever of the empty slot pair) | 4: s6, s7, s14, s15 |
| stall_pre_grasp (stage 1, hovered beside the popped bread) | 4: s1, s9, s18, s25 |
| stall_at_fixture | 3: s0, s22, s23 |
| wander_away (stage 1, went to the fridge) | 1: s17 |
| other: the bread's lever stayed jammed down after the cycle, so it never popped | 1: s27 |

**Common secondary categories.**
- wander_away: 10
- stall_at_fixture: 11
- missed_grasp on the untoasted bread: s0, s3, s11, s26
- wrong_fixture_element as an extra press: s0, s8, s13, s23, s27

**This is not one failure.** There are two systematic behaviours, one per stage. Each alone would give about 0 % success:
1. Stage 0 is completed in only 8/30 episodes.
2. In those 8, the policy never even touches the bread after the pop-up (0/8).

### Per-episode facts (lever = highest position reached by the bread's slot pair; "on" = the toaster switched on)

| ep | layout | bread's pair | bread's lever max | other lever | stage 0 | bread before toasting | what happened to the bread |
|---|---|---|---|---|---|---|---|
| s0 | 10 | p0 | 0.01 | 0 | - | grasped, 3 mm lift | left in the slot |
| s1 | 2 | p0 | 1.0 (176) | 0 | 816 | - | left in the slot, never touched |
| s2 | 10 | p1 | 0.19 | 0.03 | - | - | left in the slot |
| s3 | 8 | single | 0.46 (pushed via the bread) | - | - | pushed down, grasped, 1 mm lift | left in the slot |
| s4 | 9 | single | 1.0 (160) | - | 824 | - | left in the slot; robot ended empty-handed over the plate |
| s5 | 4 | single | 0.64 | - | - | - | left in the slot |
| s6 | 2 | p0 | 0 | **1.0 (384), empty pair toasted** | - | - | left in the slot |
| s7 | 10 | p1 | 0.07 | **1.0 (192), empty pair toasted** | - | fist push 663-692 | left in the slot |
| s8 | 2 | p0 | 1.0 (464), but the bread was wedged 2.8 cm up | 1.0 (1040) | - | **pulled up 2.6 cm (207-245), wedged** | wedged until 1232, then reseated |
| s9 | 8 | single | 1.0 (~1011) | - | 1680 | - | left in the slot, never touched |
| s10 | 4 | single | 0.77, then 1.0 (592) | - | 1239 | - | left in the slot; robot drove to the dining area empty-handed |
| s11 | 2 | p0 | 0.14 | 0 | - | **pulled up 4 cm (228-270), fell back** | left in the slot |
| s12 | 8 | single | 1.0 (176), re-pressed at 812 | - | 1517 | - | left in the slot; robot drove to the dining counter empty-handed |
| s13 | 10 | p0 | 0.32 | 0 | - | **pulled out (257), carried and released ON THE PLATE (797)** | untoasted, on the plate |
| s14 | 2 | p0 | 0 | **1.0 (480), empty pair toasted** | - | - | left in the slot |
| s15 | 10 | p0 | 0 | 0.30 (wrong pair) | - | - | left in the slot |
| s16 | 8 | single | 0.80 (fist on the bread top) | - | - | pushed down | left in the slot |
| s17 | 4 | single | 1.0 (172) | - | 842 | - | left in the slot; robot went to the fridge |
| s18 | 10 | p1 | 1.0 (272) | 0 | 950 | - | left in the slot, never touched |
| s19 | 8 | single | 0.30 | - | - | - | left in the slot |
| s20 | 10 | p0 | 0.18 | 0 | - | - | left in the slot |
| s21 | 4 | single | 0.32 (fist on the bread top) | - | - | pushed down | left in the slot |
| s22 | 10 | p0 | 0.02 | 0 | - | touch only | left in the slot |
| s23 | 2 | p0 | 0.05 | 0.11 (wrong pair) | - | - | left in the slot |
| s24 | 4 | single | 0.51 | - | - | - | left in the slot |
| s25 | 8 | single | 1.0 (192), re-pressed at 1789 | - | 899, undone at 1674 | - | re-toasting (down) at the end |
| s26 | 10 | p1 | 0.46 (pushed via the bread) | 0 | - | **3 grasps, no lift** | left in the slot |
| s27 | 2 | p0 | 1.0 (496), **jammed at 0.999 to the end** | 1.0 (672) | - | **grasp 279-306, lifted 0.8 cm, reseated** | down in the slot (lever jammed) |
| s28 | 8 | single | 0.58 | - | - | - | left in the slot |
| s29 | 10 | p1 | 0 | 0 | - | hovered beside it with finger closes, no contact | left in the slot |

**Answers to the four questions.**
- **Was the lever pressed fully?** The bread's own lever was fully pressed in 10/30 episodes: the 8 stage-0 episodes, plus s8 (bread wedged) and s27 (lever jammed).
- **Was the bread in the slot at that moment?** It was seated in every one of these except s8.
- **Did the robot reach in early?** No episode reached into the slot during toasting. In every case where the toaster was on, the robot waited.
- **What happened to the bread?**
  - 28/30: still in the toaster at the end.
  - s13: on the plate, untoasted.
  - s25: back down, re-toasting.

## Recurring failure mechanisms

### 1. After the pop-up, the policy never goes for the bread (8/8 stage-0 completions)

There was no bread contact after the pop-up in any episode: `objects.obj.gripper_touch_intervals` is empty for all 8. The policy had 1050-1180 steps left in s1, s4, s17, s18 and s25, so time was not the problem except in s9, which had 320 steps left. Its behaviours after the pop-up:

- **Hovering beside the toaster with the bread in view.**
  - s1: 850-2000
  - s9: 1680-2000
  - s18: 940-1260 against the toaster a few cm from the bread, then hovering
  - s25: 912-1640
- **Driving toward the dining counter empty-handed.** This is the stage-1 motion without the pick.
  - s4: ends with closed fingers right above the plate, 1900-1980
  - s10: passes the plate around 1560, then wanders
  - s12: at the dining counter at 1988
- **Wandering off.** s17 went to the fridge doors, 1140-2000.
- **Pressing the lever again.**
  - s12 pushed the rising lever at 812, starting a second 500-step cycle.
  - s25 pressed it again at 1789, putting the bread back down; stage 0 was undone at 1674.

**Closed fingers.** The lever is pressed with a closed fist (fingers closed at about 150-220). In s1, s9, s10, s12 and s17 the fingers stay closed until the timeout. In s4, s18 and s25 they reopen after the pop-up, but there is still no approach to the bread.

**Likely cause (hypothesis).** The popped-up scene is visually the same as the initial scene: the bread does not change colour, and the lever is up in both (`frames/cmp_before_vs_after_pop.png`). A policy without memory cannot tell "before toasting" from "after the pop-up". This fits:
- the re-presses in s12 and s25;
- the bread-first behaviour at the start (mechanism 2), where the policy sometimes picks the "take the bread" mode in the initial state.

Why it never picks that mode after the pop-up is less clear. One possibility is that the closed fist and the arm already at the lever keep it in the "operate the fixture" mode. This is an inference, not something I can test from the videos.

### 2. In stage 0 the policy goes for the bread instead of the lever (7 grasp attempts, plus s29, plus 2 bread-top presses)

**Grasps closed on the untoasted bread before any toasting** (`frames/cmp_bread_grasp_before_toast.png`):
- s0 (224-245)
- s3 (881-923)
- s8 (156-245)
- s11 (220-270)
- s13 (249-797)
- s26 (524-721)
- s27 (279-306)

**Outcomes:**
- **s13** did the whole stage-1 motion cleanly on untoasted bread: pulled it out at 257 (lift 0.19 m), drove to the dining counter, and released it on the plate at 797 (0.05 m from the centre). It then went to the toaster oven. The official check correctly fails it because the toaster was never on with the bread in it. **The policy can pull bread from the slot and deliver it**, but in this run it only did so when it skipped the toasting.
- **s8**: the lift slipped, and the bread stayed wedged 2.8 cm above its seat from 256 to 1232. The correct lever was pressed at 464, but the bread was off the carriage for the whole cycle, so there was no credit. A later press hit the empty pair.
- **s11**: the bread was lifted 4 cm, fell back, and the lever was never pressed.
- **s27**: lifted 0.8 cm and reseated, then the correct press at 496 (see mechanism 4).
- **s0, s3, s26**: no lift, then no effective lever press.

**Pressing on the bread top.** s16 and s21 pressed the fist down on the bread top instead of the lever handle. The carriage followed only to 0.80 and 0.32. s3 and s26 also pushed the carriage partway down this way (0.46) before their grasp attempts.

**Rate.** 10 of the 22 stage-0 failures handled the bread before any toasting, so this is the commonest stage-0 mechanism.

### 3. Lever pushes too shallow or too short, with no retry (stage 0)

Pushes on the correct lever were released below the 0.9 switch-on:

| episode | lever reached |
|---|---|
| s2 | 0.19 |
| s5 | 0.64 |
| s19 | 0.30 |
| s20 | 0.18 |
| s24 | 0.51 |
| s28 | 0.58 |
| s16 (via the bread) | 0.80 |
| s21 (via the bread) | 0.32 |
| s22 | <= 0.02, at the lever 340-524 without pushing |

**What happens next.** The lever springs back. The policy usually makes at most one weak retry, then hovers beside the toaster or wanders off: s16 to a wall cabinet, s19 to the stove, s24 to the sink, s28 through the kitchen to the dining counter. Before pushing, it often rests on the toaster top for 200-400 steps (s14, s15, s20, s21, s23, s26).

**What the 8 stage-0 completions did differently.**
- **One continuous push to 1.0** within about 20-70 steps of reaching the lever: s1 (160-176), s4 (144-160), s12 (145-176), s17 (147-176), s25 (176-192). The fist was already closed, the push was straight down on the handle, and the gripper stayed until the lever latched.
- **Retries that worked:** s10 (0.77, then 1.0 at 592), s18 (0.56 and 0.50 held, then 1.0 at 272) and s9 (0.17 at 592, then 1.0 at about 1011).
- **Placement of the fist.** In the failures the fist pushed beside the handle, on the housing or on the bread top, or let go early. The difference is in where the fist lands and whether it holds the push, not in reaching the toaster.

### 4. Two-pair (4-slot) toasters: the policy presses whichever lever it reaches, not the bread's

**Wrong pair pressed.**
- Fully pressed the empty pair: s6 (384), s7 (192), s14 (480). Those toasters ran empty.
- After a correct press, also pressed the empty pair: s8 (1040) and s27 (672).
- Partial pushes of the empty pair: s15 (0.30) and s23 (0.11).

**Rate.** That is 7 of the 17 two-pair episodes. In s6, s7 and s14 the robot then hovered for the rest of the episode and never tried the other lever.

**By toaster type.** Stage 0 was reached in 6/13 single-pair episodes but only 2/17 two-pair episodes (s1, s18). In layout 2 the bread was always in the left pair (p0). In layout 10 it was in either pair.

**s27 (lever jam, sim artefact).** The correct pair toasted 496-994 with the bread seated. When the 500-step cycle ended, the lever stayed at 0.999, drifting to 0.994 by 2000, and never popped. The bread stayed 5 cm down. The toaster re-armed and switched on again at 1994, when the cooldown expired with the lever still down.
- The bread had shifted 0.6 cm sideways in its slot during 624-688, while the other lever was being pressed. A wedged bread, like s8's, may be holding the carriage.
- The robot had already wandered to the coffee machine (800-2000), so it would probably have failed stage 1 anyway.

## Official-criterion and packet issues

1. **The env and the stage-0 checker differ.**
   - The env's `_check_success` latches "toasted" as soon as the bread touches the floor of a slot pair that is on. It never needs the pop-up.
   - The stage-0 checker needs the pop-up (lever <= 0.1).
   - In s27 the env latch was very likely met (bread seated, pair on 496-994), but the lever jam meant stage 0 never completed. This does not change any outcome here, because no episode delivered toasted bread.
2. **s8 is borderline but correct.** The bread sat in the switched-on pair's slot, wedged 2.8 cm above the carriage. Both checks refuse it, and I agree.
3. **s13 is correctly failed.** It has `at_target_intervals [[773,2000]]` (untoasted bread on the plate, gripper away). Stage 1 never holds because the toaster was never on. Someone reading only the object fields could mistake it for a delivery.
4. **The fixture fields hide which slot pair was used.**
   - `fixtures.stage0:toaster.value_every_16_steps` is the maximum over both pairs' levers.
   - `done_intervals` means "any pair on".
   - They report "done" for empty-pair toasting (s6, s7, s14) and merge both pairs' cycles (s8, s27).
   - The packet records neither which pair holds the bread nor the bread's height, so I took them from the saved states. Suggested fields:
     - `bread_pair`
     - per-pair on intervals
     - bread dz relative to the start (the carriage)
     - "bread touching the slot floor while its pair is on"
5. **Lever direction is not stated.** "Highest lever position" with 1.0 = pushed down is easy to misread as "popped up".
6. **`holds` overstates grasps.** It counts closed-fist pushes on the bread top as holds with lift 0: s16 104-244, s21 569-673, s7 663-692.
7. **`close_events` misses some closes.** It does not list partial re-closes from an already-closed hand: s29 264-384 (width up to 0.04).
8. **The official labeler output is meaningless on these failures.** For example, s0 is labelled "PickPlace" for the whole episode, and s8 has a "wait" at 464 for a cycle with the bread wedged off the carriage.
9. **No measurement bug found.** `replay_matches_online_log` is true in all 8 stage-0 episodes, and every lever, contact and bread-height reading I checked agreed with the video.

## Uncertainty

I rated nothing low confidence. The medium-confidence episodes and the open question in each:
- **s0, s22, s23:** whether the fist was on a lever handle while pressing with no effect.
- **s2:** whether 171-365 was meant as a push.
- **s3, s21, s29:** whether the actions were grasp attempts or presses on the bread.
- **s11:** what it did at the toaster's right end, 683-881.
- **s9:** stage 1 had only 320 steps. I still assign it to stage 1, because the other 7 episodes failed stage 1 with 760-1180 steps left.
- **s27:** the physical cause of the lever jam.

**Two inferences I cannot confirm from the videos:**
- the "pre- and post-toast scenes look the same" explanation for mechanism 1;
- "the closed fist keeps the policy in fixture mode".
