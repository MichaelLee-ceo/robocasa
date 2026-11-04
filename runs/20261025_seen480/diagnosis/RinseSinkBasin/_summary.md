# RinseSinkBasin: stage-level diagnosis (GR00T N1.5, 30 episodes, 17 succeeded, 13 failed)

Task: turn on the faucet so the water runs (stage 0), then, with the water running, swing the spout to the left, centre and right zones in any order (stage 1). Horizon 900 steps (45 s).
Per-episode files: `RinseSinkBasin_s<N>_single.json`. Contact sheets: `frames/`. Running notes: `_notes.md`.

**How stage 1 is judged** (from `rinse_sink_basin.py` and `sink.py`):
- The env checks every control step.
- Spout zones: centre is |q| < 30 deg (0.524 rad), "right" is q >= +0.524, "left" is q <= -0.524.
- The spout starts at 0, so the centre zone is credited the moment the water comes on. In practice stage 1 needs one sweep of at least 30 deg to each side while the water is running.

**Spout trace.** The packet has no spout-angle time series, only the final `washed_loc` and `spout_ori`. I read the spout joint from the saved states in `run/<tag>_steps.npz` with numpy (no simulation):
- The handle is the column that matches the packet's `handle_joint` samples. The spout is the next column (two columns on in layout 2).
- Check: the zones this trace visits (with the water on) reproduce the packet's `washed_loc` and final `spout_ori` in 30 of 30 episodes.
- Every spout angle and "side reached at step X" below comes from this trace and agrees with the video.

## Counts

| root-cause stage | episodes |
|---|---|
| 0 (water never turned on) | 4: s7, s13, s17, s20 |
| 1 (water on; one side never reached) | 9: s0, s3, s4, s10, s15, s22, s26, s27, s29 |

| primary category | episodes |
|---|---|
| actuation_insufficient | 5: handle s7 (0.346 of 0.40 rad), s17 (0.23); spout s0 (swung back only to centre), s3 (left to -0.365), s29 (left to -0.520, 0.004 short) |
| stall_at_fixture | 4: s20 (at the handle, 0.02 rad); s4, s26, s27 (stayed at or pressed the spout on the first side) |
| wander_away | 3: s10, s15, s22 |
| missed_grasp | 1: s13 (closed beside the lever twice) |

Recurring secondary categories:
- `skipped_or_wrong_stage`: worked the spout with the water off (s7, s13, s17).
- `stall_at_fixture`: hovered after the main error (s0, s3, s13, s17).
- `official_criterion_issue`: s29.
- `ran_out_of_time`: s22.

**The questions from the brief:**
- **Did the water come on?** No in 4 of 13 failures.
- **Which side was missed?** Left in 8 (s0, s3, s4, s10, s15, s22, s26, s29), right in 1 (s27). Every water-on failure got the centre plus exactly one side, and 8 of 9 ended with the spout on the right.
- **Was the water turned off by accident?** Never, in all 30 episodes.
- **Was time the limit?** Only for s22, whose right side came at 858. The other 8 water-on failures still had 197-572 steps left when they reached their first side. In successes the second side follows the first by a median of about 90 steps (range 52-227).
- **Online log vs replay:** they agree in all 30 episodes.

## Recurring failure mechanisms

### 1. One-sided sweep: after the first side, the policy stops working the spout (9 of 13 failures; all of the stage-1 failures)

**What successes do.** In 15 of 17 successes, gripper contact with the faucet is one unbroken segment from taking the spout until the second side is reached. The policy sweeps to side A and straight back through centre to side B:
- typical example s9: 323-445, right at 354, left at 440;
- s2 (layout 9) held the right side for about 170 steps (256-410) and then still swept to the left (466) without letting go.

The two exceptions:
- s28 has a 27-step gap.
- s23 let go after the left side (417), came back at 599 and finished the right side at 638.

**What failures do.** In every water-on failure, contact ends or freezes after the first side. There are three variants:

- **(a) The swing toward the second side stops short, then reverses.**
  - s0: right at 375, swung back to centre (+0.01 at 508), pushed right again (516-572), let go at 627, hovered 632-900.
  - s3: pushed left only to -0.365 (377), reversed to the right (416), then held the spout there in contact for about 440 steps.
  - s29: pushed left to **-0.520 at 617**, 0.004 rad (0.2 deg) short of the zone, then reversed to the right (703) and held it at +1.1 until the end.
- **(b) Reaches the first side, holds or presses there, then lets go or stalls.**
  - s4: right at 403, pressed the ball head at the right end for 430-620, then wandered and pushed right again.
  - s26: right at 328, held 348-590, let go, hovered up and to the right for 620-900.
  - s27: left at 363, but over-rotated to -1.02. After repositioning over the left basin it sat at the spout tip for 640-900 and moved it only from -1.06 to -0.78.
    - The same scene in s23 shows the same repositioning succeed: s23's left push had stopped at -0.56, and the gripper came down onto the nozzle and swept it to the right in about 35 steps (`frames/s23_vs_s27_return.png`).
- **(c) Leaves the faucet after the first push.**
  - s10: wrist stared at the back corner of the counter for 300-760.
  - s22: away for 235-785; right only at 858, timeout.
  - s15: large arm swings to the left of the faucet and back (400-700). The right side was credited only because its forearm knocked the spout (632-648).

`frames/after_first_side_compare.png` puts the moment about 60-100 steps after the first side side by side:
- successes s2, s9 and s12 are mid-sweep, still in contact;
- failures are parked at the spout (s0, s26, s3), pressing it (s4), or looking away (s10, s22, s27).

**Hypothesis (not proven).** The policy sees only the current images and has no memory of which sides are done. "Water on, spout at one side" looks the same whether or not the other side was washed, and at the end of half the demonstrations it is the final state. Once the sweep's momentum within a chunk sequence is lost, the policy behaves as if the task were finished: it backs off and hovers (s0, s26) or drifts (s10, s22).

This fits two observations:
- Successes almost never break contact between the two sides.
- The one success that did (s23) needed a lucky return.

It does not explain the short reversals in (a).

### 2. Handle turned only part of the way, then abandoned (s7, s17); stage-0 root cause

- **s7** (layout 10) pushed the lever to 0.346 rad (112-176) and left. The threshold is 0.40, and successes always reach the 0.50 stop.
- **s17** (layout 8) rotated the paddle to 0.23 rad (200-224) and pulled away.

In both, the policy then went on to swing the spout with no water running: s7 at 250-272 and 700-900, s17 at 371-429 (to +0.83). So it does not check the water stream before moving on.

The success s28 shows recovery is possible. Its first push stuck at 0.13 rad for about 80 steps (pressing the neck next to the lever, 176-264), then it shifted onto the lever and turned the water on at 286.

### 3. No effective contact with the lever, then hovering (s13, s20; layout 9)

- **s13** closed beside the down-hanging lever twice: at 104 (width 0.014) and 268 (0.009). The handle moved at most 0.03 rad. It then hovered 10-25 cm away for about 450 steps (288-740).
- **s20** sat at the faucet base for 124-200 with three partial closes. The handle reached only 0.02 rad. From 300 to 900 it hovered 15-30 cm to the right of the lever with the lever in the wrist view, and never touched it again.

The approach of s20 looks like that of success s18 (the wrist sees only the wall at the moment of contact). The difference is in the finger-lever contact, which the 256-pixel frames do not resolve (`frames/L9_handle_compare.png`).

## Scene dependence (strong)

| layout (faucet) | episodes | failures |
|---|---|---|
| 9: tall pull-down spring faucet, side lever hub, double basin | 7 | 5 (s0, s3, s13, s20, s26); successes s2, s18 |
| 8: steel sink, short swivel spout with ball head, paddle lever on top | 5 | 4 (s4, s10, s17, s22); success s8, which had a slow stage 1 of 416 steps |
| 10: stem faucet, horizontal spout arm, side lever | 3 | 3 (s7, s15, s29) |
| 1: L-shaped spout tube | 2 | 1 (s27); success s23 |
| 2, 4, 5, 6, 7: simple goosenecks | 13 | 0 |

Layouts 8, 9 and 10 hold 12 of the 13 failures (12 of 15 episodes there fail), and all 4 stage-0 failures.

In these episodes the layout id equals the style id, and each layout has its own faucet model. So I cannot separate the faucet geometry from the rest of the kitchen.

What the faucet-specific failures look like:
- **Layout 8:** both wanderers (s10, s22) fixate on the back counter corner above the faucet.
- **Layout 9:** the failures hold the spout head at the right and then let go.

## What successful episodes do differently

- **Handle.** One approach, often with a partial close (~0.04) on the lever hub, and the handle goes straight to its 0.50 stop.
  - Water on at 123-286 (median 189).
  - The only slow one is s28 (286), which self-corrected.
- **Overview.** After the water comes on, nearly all successes raise the arm to a pose with the wrist looking down at the stream (about 50-100 steps), then descend onto the spout. Failures do this too, so it is not discriminative.
- **Spout.** They take the spout neck or head and sweep to one side and straight back to the other while keeping contact. Order: left first in 9, right first in 8.

## Official-criterion issues

1. **The 30-degree zone threshold decides outcomes at the margin.**
   - s29 failed at -0.520 against the -0.5236 threshold (0.2 deg). The stream had visibly been moved to the left.
   - Successes were credited with small margins: s18 +0.54, s16 +0.55, s23 -0.56, s19 -0.57.
   - Not a bug, but the success rate is sensitive to a few degrees of spout travel.
2. **The centre zone is free.** It is credited at water-on because the spout starts at 0, so the "three locations" are really two sweeps.
3. **The water-on threshold (0.40 rad)** agreed with the video in every case I checked: no stream is rendered at s7's 0.346 or s17's 0.23.

## Packet and tooling issues

1. **No spout timeline.** The packet has no spout-angle (or per-direction "first washed") time series. `stages[1].complete_intervals` only marks the last few steps of a success, and on failures it is empty. Diagnosing this task needs the spout trace I rebuilt from the saved states. I suggest adding the spout joint (and `washed_loc` per step) to the packet.
2. **Handle value near zero looks like "fully on".** `fixtures.stage0:sink.value_every_16_steps` (and `state_every_32_steps.handle_joint`) shows **6.2832** when the handle sits a hair below 0 (s2@96, s11@160, s13@112, s20@128). It is the angle wrapped mod 2*pi, not |angle| as the brief describes. `done_intervals` and `water_on` are correct, but a reader of the raw value would think the handle is far past "on". Small fix: report the signed angle.
3. **Arm-link contacts are invisible.** `gripper_touch_intervals` count only gripper contact with the whole sink fixture: handle and spout are not separated, and contacts by arm links are not recorded. The spout moved with no recorded touch in s15 (628-648, +0.21 to +0.62, which credited "right"), s22 (430-500) and s28 (475-502).
4. **`close_events.on_nothing` carries no information here.** It is always true, because there are no task objects. Partial closes of about 0.04 at the lever are normal in successes.
5. **Spout column offset.** In layout 2 the saved-state column order differs (spout = handle + 2). This matters only for anyone automating the spout trace.

## Uncertainty

- **Low or medium confidence:**
  - s13 and s20: why the lever contact failed is not visible at video resolution.
  - s10 and s15: what touched the spout while the wrist looked elsewhere.
  - s27: which side of the spout tip the fingers pushed.
- **The "no memory of washed sides" explanation for mechanism 1 is a hypothesis.** It fits the contact-continuity data (15 of 17 vs 0 of 9) but was not tested.
- **The scene effect is strong but confounded.** The faucet model, layout and style all change together. n is 3-7 per layout.
