# KettleBoiling: stage-level diagnosis (GR00T N1.5, 30 episodes, 20 succeeded, 10 failed)

Task: pick the kettle from the counter and place it on a burner (stage 0), then turn that burner on (stage 1). Horizon 1000 steps.
Per-episode files: `KettleBoiling_s<N>_single.json`. Contact sheets: `frames/`. `frames/all_first_touch_wrist.png` puts the wrist view at the first kettle contact of all 30 episodes side by side.

## Counts

| root-cause stage | episodes |
|---|---|
| 0 (pick and place the kettle) | 8: s2, s3, s4, s8, s10, s17, s24, s26 |
| 1 (turn on the burner) | 2: s5, s9 |

s4 failed in stage 1 by the packet (first incomplete stage), but the root cause is in stage 0. Stage 0 took 765 steps (median 271), and the knob was mid-turn (0.22 of 0.35 rad) at the timeout.

| primary category | episodes |
|---|---|
| missed_grasp | 6: s2, s3, s8, s10, s17, s26 |
| stall_pre_grasp (secondary: ran_out_of_time) | 1: s4 |
| other: dislodged the kettle on retreat after a long hold at the target | 1: s24 |
| wrong_fixture_element | 1: s9 |
| actuation_insufficient (secondary: wander_away) | 1: s5 |

Secondary categories that recur:
- skipped_or_wrong_stage (empty-handed "place" or knob after a missed grasp): s3, s8, s10, s17
- stall_pre_grasp (hovering above the kettle after a miss): s2, s3, s10, s26

## Recurring failure mechanisms

### 1. Kettle-handle grasp fails, and the policy does not retry effectively (6 of 10 failures; 7 counting s4's 500-step stall)

**Grasp timing.**
- Successes: a lifting grasp follows within 0-12 steps of the first kettle contact (20/20).
- Failures: never in s2, s3, s8, s10, s17 and s26 (max lift 0-1.4 cm); 497 steps in s4.

**What the wrist view showed at contact** (`all_first_touch_wrist.png` and the per-episode sheets):
- s2, s10, and s3's first attempt: the arched handle lies flat across the kettle, along the finger-closing direction, so there is nothing for the fingers to close around. I did not see this in any of the 20 successes.
- s17: the handle is upright but to the right of the finger gap. It lifted 1.4 cm and slipped (118-126).
- s8 and s26: the alignment looked like that in successes, so the cause is unknown. s26's ~45 deg handle also appears in successes s25 and s27; in s8 the kettle is wedged against a coffee machine.

**After the miss, the policy does not notice it is empty-handed.** It either:
- runs the rest of the motion without the kettle:
  - s3: "place" over the hob, 176-560
  - s8: place-like motion 216-500, then lit two burners (rear_left 2.39 rad, front_left 0.58) with the kettle still on the counter
  - s10: 160-240
  - s17: 440-560 and 712-800
  - s2: 200-240
- or hovers 15-40 cm above the kettle for hundreds of steps without coming down: s2 280-1000, s26 176-1000, s3 856-1000.

Where it does retry, the closes are short and do not lift: s10 seven times (max 5 mm), s17 five times (max 7 mm, pushed the kettle 4.5 cm away). The only successful re-grasp in the whole task is s14, which re-closed 5 steps after a missed first touch.

**Scene dependence (strong, n is small).** I grouped episodes by kettle start_pos and checked the agent views.

| layout | episodes | grasp outcome |
|---|---|---|
| kettle on the counter left of the range, start ≈ (0.5, -2.2) | s4, s10, s17 | 3/3 had grasp trouble: s4 stalled 500 steps, s10 and s17 never lifted |
| kettle right of the gas hob, start ≈ (4.85, -0.42) | s0, s2, s3, s13, s18, s20, s26 | 3/7 missed the grasp (s2, s3, s26); s0, s13, s18, s20 succeeded |
| all other layouts | 20 episodes | 1 grasp failure (s8) |

Kettle models also differ between episodes, so I cannot separate the effect of the layout from that of the kettle model.

### 2. Stage 1 errors after a good placement (s9, s5, plus s4's timeout)

- **s9, wrong knob.** The kettle was placed at 169 on a right-hand burner. The policy turned the left-most knob (340-512, rear_left to 1.32 rad); a flame is visible on the rear-left burner from 392 (`frames/s9_stove_zoom.png`). It then wandered for 485 steps.
- **Knob prior.** In the 20 successes, exactly one knob was on at the end, and it was front_left in 15, "left" in 3 (five-burner hob) and front_right in 2 (s6, s13). The policy seems to strongly prefer the left-most knob, and it works because the kettle usually lands on the front-left burner. The two stage-1 failures are both episodes where the kettle landed on a right-hand burner: s9 turned the wrong (left) knob, and s5 nudged its knob only 0.05 rad. The other two right-burner placements (s6, s13) succeeded, so 2/4 vs 0/16 for left placements. Small n; a hypothesis, not a finding.
- **s5, knob abandoned.** A large kettle sat above the knob row. The gripper worked the knob while rubbing the kettle (contacts 282-368), moved the knob only to 0.05 rad, backed off at ~360, and stayed parked for the last ~600 steps. Kettle obstruction is my hypothesis from the wrist view.
- **s4, too late.** Correct knob, 0.22 rad at timeout. In successes the knob goes from first contact to on in 12-167 steps (median ~38).

### 3. Holding the kettle on the burner instead of releasing (s24; tolerated in successes)

- In successes the gripper keeps holding after the kettle reaches the burner for a median of ~14 steps; the longest are s27 (97), s18 (51), s23 (42) and s29 (41).
- s24 held for 235 steps (224-459) and opened at 459. The tipping kettle then got caught again (hold 468-490), was lifted 17 cm as the arm rose, and fell on its side off the burner centre (~490-500).
- The policy then went to the knob as if stage 0 were done (0.05 rad) and wandered. It never tried to right the kettle.

## What successful episodes do differently at the critical moments

- **Grasp.** The handle runs across the finger gap at the first descent (upright to about 45 deg in the wrist view). The gripper closes once and lifts 9-44 cm.
- **Release.** Within ~0-50 steps of reaching the burner, mostly on the front-left burner within 0.01-0.11 m of its centre. Then it backs off cleanly.
- **Knob.** It reaches the knob of the kettle's burner 68-196 steps after release (median ~137) and turns it past 0.35 in 12-167 steps of contact. The slow ones are s22 (167 steps of repeated knob contacts) and s23 (115). s7 only reached 0.358 rad on its very last step.

## Official-criterion and packet issues found

1. **No upright check.** In s24, stage 0 held 501-522 with the kettle lying on its side, about 15 cm from the burner centre. A burner lit in that window could have scored a toppled kettle.
2. **The gripper-away (>= 25 cm) clause distorts stage timing in this task.** The knobs are often right in front of the kettle.
   - Stage 0 intervals break whenever the gripper is at the knob: s0, s13, s14, s18, s22, s28.
   - Stage 0 can first hold only after the burner is already on. In s12 the kettle was placed at 216 and the burner was on at 345, but both stages first hold at 376.
   - So `steps_since_previous_stage` shows s12 as a "slow stage 0" (376 > p90 348) with a 0-step stage 1, and s18 and s28 as slow stage 1. The packet's typical medians (271/152) include this effect.
3. **`fixtures.stage1:stove.done_intervals` means "any knob >= 0.35", not "the burner under the kettle".** It says done in s8 (688-1000) and s9 (386-1000) although stage 1 never held. The packet has no field for which burner the kettle is on, so the wrong burner in s9 had to be read from the flame in the video.
4. **s13 looks failed in the packet.** It is a success, but the packet shows `first_incomplete_stage = 0` and `complete_at_end = false` for both stages, because the conditions dropped in the last 3 steps (334-336).
5. **`holds` overstates grasps.**
   - It includes 1-2-step contacts and pushes with lift 0 (s2 122-124, s8 127-128, s5 282-369).
   - `height_drop_after_release_m` goes negative when the kettle is re-grasped within 2 s (s14 -0.133, s24 -0.068).
   - A close on air registers nothing: s3's first attempt at ~140 is invisible in the packet.
   - `min_eef_dist_m` is measured to the kettle centre and is ~0.16-0.20 m even for successful handle grasps, so it does not tell you whether the gripper reached the handle.

## Uncertainty

- The handle-orientation explanation for the grasp misses is my reading of 4-step wrist frames. It fits s2, s3 and s10, does not explain s8 or s26, and is only partial for s17.
- The s5 kettle-obstruction explanation and the s24 "re-closed vs never let go" question are medium confidence.
- Burner names for the right-hand placements (s5, s9) are not in the packet.
