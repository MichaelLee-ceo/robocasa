# ScrubCuttingBoard: stage-level diagnosis (GR00T N1.5, 30 episodes, 13 succeeded, 17 failed)

Task: pick up the sponge (stage 0), scrub the cutting board with it, then release it (stage 1). Horizon 800 steps.
Per-episode files: `ScrubCuttingBoard_s<N>_single.json`. Contact sheets: `frames/`. Running notes: `_notes.md`.

**How stage 1 is judged.** The rule is in `scrub_cutting_board.py`. A board contact counts only if all of these hold:
- the sponge and the board are in geom contact;
- `OU.check_obj_grasped(sponge)` is True;
- the sponge's xy is at least 2 cm from every point already recorded.

Success requires all of these:
- at least 5 points;
- the diagonal of the points' bounding box is at least 0.10 m;
- the eef is more than 15 cm from the sponge centre.

There is no check of where the sponge ends up.

**Extra data used beyond the packets** (read-only, no simulation):
- the per-step replay traces `analysis/ScrubCuttingBoard/*_trace.json.gz`: sponge position, touch, grasp flag, counter per step;
- the saved states `run/.../*_steps.npz`, where state[12] and state[13] are the two finger joints.

## Counts

| root-cause stage | episodes |
|---|---|
| 0 (pick up the sponge) | 3: s11, s25, s29 |
| 1 (scrub, then release) | 14: s2, s3, s7, s8, s9, s10, s12, s13, s14, s17, s18, s19, s24, s28 |

| primary category | episodes |
|---|---|
| actuation_insufficient (too few counted scrub contacts) | 7: s3, s8, s10, s13, s17, s18, s19 |
| stall_holding_at_target (criterion met, kept scrubbing, never released) | 5: s2, s7, s9, s14, s24 |
| missed_grasp | 3: s11, s25, s29 (s29 recovered, then ran out of time) |
| official_criterion_issue (scrub happened but was not counted) | 1: s12 |
| dropped_in_transit (sponge slipped out mid-scrub) | 1: s28 |

The same failures, sorted by the outcomes you asked about:

| outcome | episodes |
|---|---|
| never grasped | s11, s25 (s29 missed first and re-grasped 240 steps later) |
| scrubbed too little or in one spot, then released | s3, s8, s10, s12, s17, s18, s19. All released at 241-463 with 1-4 counted contacts, and all moved more than 15 cm away within 10-25 steps. |
| scrubbed enough but never released | s2, s7, s9, s14, s24; also s29 after its late grasp. s13 never released either, but its scrub was also insufficient until 720. |
| released but stayed within 15 cm | none |
| sponge dropped | s28 (slipped out of the fingers on the board at 244) |
| ran out of time | s29 (late stage 0); s2, which started its return-and-release move at ~776 |

After an early release or a missed grasp the policy never went back:
- After a release it raised the arm and looked around until 800 (`frames/groupB_late.png`).
- After a missed grasp it kept the gripper closed and made empty-handed "scrub" passes over the board (s11, s25, and s29 briefly).

## Recurring mechanisms

### 1. When the policy stops scrubbing is unrelated to scrub progress (12 of 17 failures)

The time from the first counted contact to the release varies widely:

| group | first contact -> release |
|---|---|
| successes | 153-601 steps, median 361 |
| early releasers | 53-291 steps |
| never released | 344-635+ steps (still scrubbing at 800) |

The policy cannot see the counter, and the stop time is not tied to how much contact was made. Both tails fail:
- **Short tail (early release):** s17 pressed in one spot for ~90 steps and released at 273. s18 had ~100 steps, half of them off the board's far edge. s10 made one slow pass of ~145 steps. s19 made one 11 cm push.
- **Long tail (no release):**
  - s7 and s24 met the criterion at 330 and 327 and still had the sponge on the board at 800, about 470 steps later (`frames/groupA_end.png`).
  - s14: criterion at 385, still holding at 800.
  - s2 started lifting the sponge back towards its start only at ~776.
  - s9 made a return-like lift to +29 cm at 628-692, then put the sponge back on the board and kept scrubbing (`frames/s9_return.png`).
- **Successes near the long tail:**
  - s6 released at 700, 470 steps after its criterion.
  - s23 released at ~777-784 and passed at step 796, 4 steps before the horizon.
  - s4 released after 506 steps of scrubbing.
- `frames/cmp_release_vs_held.png`: at the moment of release, successes have lifted the sponge off the board and are carrying it back over the counter. The held failures still have it on the board at the same time.

### 2. Scrubbing in the air (s3, s13; partly s10 and s17)

The sweep is visually convincing, but the sponge rides a few mm to a few cm above the board, so the counter does not move although grasp=True and the sponge reaches new positions.
- **s3:** strokes of 34 cm during 276-412 with the sponge centre at +29..35 mm, against +25 mm when it rests on the board. The counter stayed at 4. It released at 463 with 4/5 contacts and sweep 0.099/0.10 (`frames/s3_scrub_zoom.png`, `frames/s3_release.png`).
- **s13:** waved 4-8 cm above the board during 300-530, which is visible as the board looking smaller in the wrist view (`frames/s13_mid.png`). The criterion was reached only at 720.
- **Contact density:** counted contacts per 100 scrub steps are 1.4 (s3) and 1.0 (s13), against 1.3-5.4 in successes. Rate alone does not separate the groups; s23 passed at 1.3 by scrubbing for a long time.

### 3. The scrub counter goes blind when the fingers shift sideways (a criterion artifact; decisive in s12, contributing in s8, s10, s19, s18)

`OU.check_obj_grasped` requires `qpos < 0.035` for both finger joints. Panda `finger_joint2` is negative (range -0.04..0), so its test always passes, and the predicate reduces to `touch and finger_joint1 < 0.035`. This matches the trace's grasp flag on 100 % of steps in all 30 episodes.

When the sponge is pressed on the board, the finger pair often slides sideways in the hand:
- one way: finger 1 goes to its open limit (+0.040), finger 2 sits at -0.016;
- the total width stays 0.058, and the sponge is still firmly held;
- in this direction every board contact is ignored;
- the opposite shift (finger 2 at its limit) is counted.

These blind windows typically start right after the first counted contact. Examples:

| episode | blind window | what happened inside it |
|---|---|---|
| s12 | 152-238 | the sponge slid ~36 cm along the board, flat and low; 1 contact counted in the whole episode |
| s19 | 167-243 | its only 11 cm push |
| s10 | 250-307 | 9 cm of pressed sliding |
| s8 | 202-345 | the main scrub |
| s23 (success) | 325 of 601 scrub steps | — |

- The wide **sponge_5** (total width 0.065, so finger 1 sits at about 0.032 when centred) is 3 mm from the threshold. sponge_5 succeeded in only 1 of 5 episodes.
- **Correlation, not a proven cause:** the early-release failures spent 9-56 % of their holding time in the blind direction, while five of the never-released failures spent 0 % (s7, s9, s14, s24, s29) and had the fingers shifted the other way. The exceptions are s2 (27 %) and s13 (11 %).

### 4. Grasp failures on a hard sponge pose (s11, s25, s29; `frames/cmp_grasp_wrist.png`)

- **s11:** the wide sponge_5 lay rotated ~25 deg to the fingers and squeezed out sideways as they closed (lift 1.4 cm). The fingers then closed on air.
- **s25:** the sponge spawned flush against the board's raised edge. The left finger came down at the edge, pinched only the top of the sponge, and the sponge flipped out.
- **s29:** an off-centre close on a ~20 deg rotated sponge slipped as the arm rose at 104. The policy carried on empty-handed, hovered, reopened at 268-284 and re-grasped at 341. That was 235 steps late; stage 1 then progressed normally (criterion at 559) and ran out of time.
- **s28** is related: a shallow, tilted hold, and the sponge was pulled out of the fingers while being dragged on the board (width 0.056 -> 0.040 -> 0.001 at 200-244). The gripper never opened.
- Slow but successful grasps on rotated sponges: s9 (stage 0 at 180) and s14 (162).

## What successful episodes do differently

- **Grasp:** usually a centred grasp within ~10-30 steps of the first touch. Stage 0 at 95-140, median 111.
- **Scrub:** on the board, with contacts accumulating at 1.3-5.4 per 100 steps.
- **Release:** after 153-601 steps they lift the sponge off the board, usually carry it back over the counter towards its start, open, and rise; the eef is more than 15 cm away 12-25 steps after the release.
- **Marginal passes:** s21 (exactly 5 contacts, sweep 0.121) and s26 (exactly 5 contacts).

**Scene dependence (small n).**
- Sponge models: sponge_8 succeeded in 5 of 6 episodes, sponge_4 in 4 of 10, sponge_6 in 3 of 9, sponge_5 in 1 of 5.
- Same layout, opposite outcomes, so release timing varies even within a layout: s21 vs s24 (L4), and s15 vs s7 (L10).
- The L7 episodes (s12, s28) and L2 episodes (s11, s14) all failed, each by a different mechanism.

## Official-criterion and packet issues

1. **Finger-1-only grasp test** (mechanism 3). It makes the counter ignore real, firmly held scrubbing. Most likely it flipped s12 (primary category) and it contributed in s8, s10, s18 and s19. A contact-based hold test, like the packet's own stage-0 check (`_held_lifted`), would avoid it.
2. **No check on where the sponge ends up.** The sponge may end anywhere, e.g. on its side next to the toaster in s16 (tilt 79 deg). The release test is only "eef more than 15 cm from the sponge".
3. **Stage 0 looks like a failure in every success.** The packet reports `first_incomplete_stage = 0` for all 13 successes and for all 8 early-release failures. The "hold and lift" condition is not latched, so it is necessarily false once the sponge is released. The real failed stage of those failures is 1; I set `failed_stage` = 1.
4. **Fragmented stage-0 intervals in s4 and s27** come from the held sponge touching the counter at the board's edge. These are not real undos.
5. **`gripper.close_events` misses about half of the sponge grasps.** It lists no close in 14 of the 28 episodes where the sponge was grasped and lifted, e.g. s7 (width 0.080 -> 0.059 at ~96-99), s13, s15 and s18. It also gives the s29 miss at 87, but its first real close was at 96-100. Use the per-step width and touch instead.
6. **The packet's `objects` is empty for this task,** so there are no holds, release or lift fields for the sponge; those came from the trace files.
7. **Replay and online log agree in all 30 episodes.** No disagreement between the packet counters and the video was found. The counter values are consistent with what the trace heights and the video show.

## Uncertainty

- The "air scrub" gaps (s3, s13) are inferred from replayed sponge-centre heights and the counter not moving. The video cannot resolve millimetres, and sponge tilt (up to 25 deg) blurs the height reading. A z-height contact proxy I tried over-counted badly, so I did not use it quantitatively.
- How many contacts the blind windows hid is estimated, not measured. Board contact cannot be read without the simulator. That s12 would have passed is likely but not proven (medium confidence), and s8, s10, s18 and s19 are less certain.
- Medium-confidence episodes: s3, s8, s10, s12, s13, s18, s19. All others are high.
