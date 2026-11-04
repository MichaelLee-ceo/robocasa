# SearingMeat: stage-level diagnosis (GR00T N1.5, 30 episodes, 9 succeeded, 21 failed)

Task: take the pan out of an (upper) cabinet and put it on the named burner (stage 0), put the meat from its plate into the pan (stage 1), turn on the named burner (stage 2). Horizon 2900 steps.
Per-episode files: `SearingMeat_s<N>_single.json`. Contact sheets: `frames/`. Running notes: `_notes.md`.
Key comparison sheet: `frames/all_first_grasp_wrist.png` (wrist view just after the first pan grasp, all 30 episodes; rows 1-2 are the tip/across grasps, the rest grasp at the handle base).

## Counts

| root-cause stage | episodes |
|---|---|
| 0 (pan from cabinet to burner) | 17: s0, s1, s2, s3, s4, s6, s9, s11, s12, s13, s16, s17, s18, s19, s21, s24, s26 |
| 1 (meat into pan) | 2: s10, s29 |
| 2 (turn on the named burner) | 2: s25, s28 |

s10's packet says first_incomplete_stage = 0, but the pan is on the target burner at the end (0.061 m). Stage 0 only "breaks" because the gripper is within 25 cm of the pan. The real failure is in stage 1.

| primary category | episodes |
|---|---|
| missed_grasp (pan) | 6: s0, s1, s4, s6, s12, s16 |
| dropped_in_transit (pan) | 3: s2, s9, s13 |
| placed_off_target | 7: pan s3, s11, s17, s18, s24, s26; shrimp s29 |
| release_timing (pan pushed off the burner before letting go) | 2: s19, s21 |
| wrong_fixture_element (wrong knob) | 2: s25, s28 |
| other: shrimp stuck to the open finger above the pan | 1: s10 |

`ran_out_of_time` is not the root cause anywhere. Two episodes had a very slow stage 0 but still had enough time left:
- s10: 1819 steps; it then failed in stage 1.
- s23: 1970 steps; it still succeeded.

In total, 9 episodes never got the pan onto the stove: s0, s1, s2, s4, s6, s9, s12, s13, s16. Another 8 placed it more than 0.15 m from the named burner: s3, s11, s17, s18, s19, s21, s24, s26.

## Recurring failure mechanisms

### 1. The pan is grasped at the tip of its handle or with the handle along the finger-closing axis (12 failures)

At closing, the wrist view (`all_first_grasp_wrist.png`) shows one of two things:
- **Base grasp.** The pan body fills the top half of the image and the handle base/ferrule runs down between the fingers. 16 episodes; 8 succeeded (s5's success is spurious, see below).
- **Tip/end or across grasp.** Only the handle or its end is visible. 0 of these first grasps succeeded:
  - handle tip/end: s0, s1, s2, s3, s9, s13, s17, s24
  - handle lying across the fingers: s6, s12, s16
  - flat cast-iron handle under the fingers: s4
  - s10 and s23 started this way and needed 3 and 2 retries.

The packet agrees: `pan.min_eef_dist_to_centre_m` (centre = bowl centre) is ≥ 0.10 m for tip grasps (s1, s2, s3, s9, s13, s17) and 0.025-0.08 m for base grasps.

What follows a bad grasp:
- **Missed / slipped, nothing carried** (s0 115, s1 116-205, s4 178, s6 166 and 794, s12 115, s16 169):
  - The fingers usually stay closed for the rest of the episode: s0 392-2848, s1 208-2900, s12 136-2900, s16 184-2900.
  - The arm then hovers in front of the pan or wanders for 2000+ steps.
  - Real retries are rare: s6 once; s4 about 10 times, lifted at 1697 and dropped it on the floor at ~1796.
- **Carried by the tip.** The pan hangs vertically and slips out during the base rotation:
  - s2 (~180): fell 0.5 m onto the counter, upside down
  - s9 (577): fell in front of the range after being held there for 300 steps
  - s13 (~262): fell off the counter
  - or it is lowered tilted and lands off-centre: s3 (0.151 m), s17 (dragged to rear_right, 0.37 m), s24 (on rear_center instead of front_center).
- **Continues the task without the pan** (`missed_grasp_continued_empty`):
  - s2: fish dropped on the bare cooktop, task knob on
  - s4: shrimp put on the bare front_left grate, task knob 3.13 rad
  - s13: steak on the bare left grate, task knob 2.8 rad
  - s1, s6, s9: went for the meat.
  - The only recovery from a drop is s23: the pan landed upright on the counter within reach, and it was re-grasped at the base at 1491 and finished.

Scene dependence (n is small):

| pan model | handle | outcome |
|---|---|---|
| pan_5 | long tan handle | 4 of 6 grasped at the tip/neck (s1, s3, s9, s17), all failed; the two pan_5 successes grasped near the body (s15, s23 on its retry) |
| pan_7 | cast iron | s4 flat-handle miss, s13 tip grasp; the other 6 gripped at the base |
| pan_6 | wooden with ferrule | s0, s12 and s10 failed at the end cap / across; s7, s8, s14, s18 gripped the ferrule |

By layout:
- Layout 9 (pan ~2 m from a 5-burner hob, high cabinet): 6 of 7 failed; 4 of those were tip/end grasps (s0, s2, s3, s13).
- Layout 8: 0 of 3 succeeded.
- Layout 6: 0 of 3 succeeded.

I cannot separate the layout effect from the pan-model effect.

### 2. Imprecise pan placement on 5-burner hobs, and pushing the pan while still holding it (8 failures)

The pan was carried correctly but ended more than 0.15 m from the named burner:
- **Wrong neighbouring burner, centre-column targets.**
  - s11 (rear_center): put on rear_right, 0.184 m, 684
  - s18 (front_center): put between left and front_center, 0.233 m, 377
  - s24 (front_center): put on rear_center, 0.193 m, 569
  - s26 (rear_center): put between left and rear_center, 0.19 m, 587
- Of 7 episodes with a centre-column target that reached the stove, 4 were off target and 2 only just inside the 0.15 m threshold (s5 0.140, s28 0.145). Only s27 was clean (0.09). Side targets that reached the stove were off in 3 of 13.
- **Release drift.** The pan was on target while still held, then pushed off before release:
  - s19: 0.068 m at 512, at_target 541-558, released at 0.19 m (582)
  - s21: 0.103 m at 256, at_target 277-290, released at 0.176 m (321)
  - s17: dragged from front_left to rear_right, 640-668.
- **Near miss.** s3 ended 1.2 mm beyond the threshold.

In successes, the pan is released 0-60 steps after reaching the burner, flat, at 0.056-0.11 m. The policy never re-placed a misplaced pan: s3 and s26 touched it once, moving it less than 1 cm. It always went on to the meat and the knob as if stage 0 were done.

### 3. Wrong knob (root cause in s25, s28; also present in s10, s18, s19, s21, s26)

The policy does use the instructed burner name: it turned the named knob first even with no pan on the stove (s2, s4, s13). But among the episodes where it turned a knob, the first knob was not the task knob in 8:
- **Front/rear swap on the same side:**
  - s10: front_right for rear_right
  - s19: rear_left for front_left
  - s25: rear_right for front_right, then stalled holding a knob 2000-2900
  - s15 (success): front_left and rear_left moved together. Layout 6 (s19, s25) has its knobs on a rear console.
- **The burner actually under or next to the pan:**
  - s18: left
  - s26: rear_right
  - s28: rear_right, 768-1248. It then turned the task knob 'center' on and straight back off twice (1632-1696, 1949-2016).
- **Left/right swap:** s21 (front_right for front_left).

All 9 successes turned the task knob first (s15 simultaneously with a second knob).

### 4. Meat release (s29, s10)

- **s29.** The pan was correct (381) and the task knob was on (1248), but the shrimp was released over the pan's near rim at the handle junction (655). It rests on the rim, 0.107 m from the pan centre (the in-pan threshold is 0.07 m).
- **s10.** The gripper opened fully above the pan (2224-2232), but the shrimp stayed stuck to the inside of one finger. It was carried to the knobs, and the wrong knob (front_right) was turned.
- Related errors in episodes that had already failed stage 0:
  - s26: shrimp released 15 cm above the bare grate, 0.55 m from the pan
  - s17: shrimp put on the empty target burner.

## What successful episodes do differently at the critical moments

- **Grasp.** They close once at the handle base/ferrule with the pan body in view (s5, s7, s8, s14, s15, s20, s22, s27, at steps 77-121). The pan is carried flat and released over the burner within 0-60 steps (s27 held about 55 steps).
- **Stage times.**
  - Stage 0: 321-524 steps, except s23 (1970, after a recovered drop).
  - Stage 1: 120-273 steps after stage 0.
  - Stage 2: they reach the knob row and turn only the task knob, 138-392 steps after stage 1.
- **Recovery.** The only successful recovery from a grasp error is s23. Its pan landed upright on the counter, and the second grasp was taken at the base.

## Official-criterion issues

1. **False-positive success in s5.**
   - The pan flipped over while carried by the handle (tilt 179° from step 216 to the end; the start tilt is 0).
   - It was placed upside down, and the drumstick lies on its upturned bottom (`frames/s5_upsidedown_vs_upright.png`).
   - Neither the stage checks nor `_check_success` test that the pan is upright, and `check_obj_in_receptacle` only needs contact plus an xy distance below 0.07 m.
2. **The burner order in `stove.check_obj_location_on_stove` (s28).**
   - It returns the first burner, in STOVE_LOCATIONS order (rear_left, rear_center, rear_right, …, center), that is both within 0.15 m and switched on.
   - The s28 pan was within 0.15 m of both rear_right (0.108) and center (0.145). With rear_right on, the function returns rear_right, so the task cannot succeed even while 'center' is on.
   - The same 0.15 m test counted the pan as "on center" although it sits mostly on rear_right.
3. **Near misses.**
   - s3: pan 0.1512 m from the target. It still had a stage-1 failure.
   - s29: shrimp on the rim, 0.107 m (threshold 0.07). This is a correct verdict, but it is the closest stage-1 miss.
4. **The gripper-away (≥ 25 cm from the meat) clause.**
   - s20: the burner was on from 564, but success fired only at 805, because the hand stayed at the knob near the meat.
   - Stage-0 intervals also break whenever the hand is near the pan (s10, s14, s25, s27, s29, …). This produces the misleading first_incomplete_stage = 0 in s10.
5. **Extra burners are not penalised.** s15 succeeded with front_left also lit.
6. **`fixtures.stage2:stove.done_intervals` only means "task knob ≥ 0.35".** It reads done with no pan (s2, s4, s13) and with the pan on another burner (s11, s18, s24, s26).

## Packet / tooling observations

- **`gripper.close_events` misses pan grasps on thick handles**, where the width falls only from 0.076 to about 0.055-0.06:
  - s13: grasp at ~106-120; only the 254 slip is listed
  - s20: grasp at ~90 not listed
  - s24: no pan close at all, although the pan was carried 136-569
  - s3's grasp close (130, width 0.051) is flagged `on_nothing`, because contact started 17 steps later.

  Look at `holds` and `width_every_8_steps` for these.
- **`holds.lift_m` is about 0 for most good pan carries** (s15 0.002, s22 0.002, s25 0.002). The pan starts at z = 1.44 m and is lowered, and lift counts only upward motion. Hold intervals can also split one continuous grasp (s20: 90-98 and 166-311).
- **No packet measurement contradicted the video** apart from the close-event misses above. The s5 tilt of 180° is real (verified visually). The pan's body origin is the bowl centre in all five pan models: each MJCF offsets the geoms by 0.045-0.06 m. So `pan_nearest_burner` and the meat-in-pan distances refer to the bowl centre.

## Uncertainty

- **s3:** root cause put in stage 0 (1.2 mm miss), but stage 1 also failed outright.
- **s10:** "shrimp stuck to an open finger" is my reading of the wrist view; it may be a contact artifact.
- **s17:** whether the pan was dragged by the arm or swung on release.
- **s28:** whether the short centre-on windows would have counted with rear_right off. The packet does not record gripper-far for the meat.
- **s15:** which knob the policy meant to turn.
- **Small sample.** The pan-model and layout groupings are based on 3-8 episodes each; treat them as hypotheses.
