# RinseSinkBasin running notes

## Criterion (code: robocasa/environments/kitchen/composite/cleaning_sink/rinse_sink_basin.py, models/fixtures/sink.py)
- water_on = 0.40 < handle_joint < pi (handle saturates ~0.50).
- spout_ori from spout joint q (mod 2pi): centre |q| < pi/6 (0.524); "right" = +0.524..pi; "left" = -pi..-0.524.
- washed_loc updated in reward() -> every control step. Spout starts at 0 = centre, so centre is washed the moment
  water turns on. Stage 1 really needs only one sweep to each side (>= 30 deg) with water on.
- Spout trajectory: NOT in the packet. Recovered from run/<tag>_steps.npz 'states' (flattened sim state): the handle
  column is the one matching packet state_every_32_steps.handle_joint; spout = handle col +1 (layout 2: +2).
  Validated: reconstructed washed_loc + final spout_ori match packet process_progress_end in 30/30 episodes.
  (scratch script: spout.py / tl.py in scratchpad; read-only numpy on saved states, no sim.)

## Per-episode spout/water timeline (W = water on; L/C/R = official spout zone)
Failures:
- s0  L9: water 210; R 375-453, back to C 454-546, R 547-900 (spout parked +0.79). Never < -0.02.
- s3  L9: water 233; spout drifted to -0.13 (256), -0.35 (384, LEFT side but short of -0.524), then swung to R 416-900.
- s4  L8: spout pushed to +0.39 while turning the handle; water 270; R 403-900. Never negative.
- s7  L10: water NEVER on (handle max 0.346 < 0.40!). Spout pushed right +0.36 (272), then +0.64..+1.09 (704-900) water off.
- s10 L8: spout pushed +0.49 during handle turn; water 205; R 485-900. Never negative.
- s13 L9: water never on (handle max ~0.025). Spout -0.19 at 784.
- s15 L10: water 140; spout +0.21 (368-624), R 644-900 (+0.62). Never < -0.06.
- s17 L8: water never on (handle max 0.23). Spout pushed right (R from 395) water off.
- s20 L9: water never on (handle ~0.02), spout never moved.
- s22 L8: water 191; spout ~0 to +0.18 for 500 steps; R only at 858.
- s26 L9: water 161; R 328-900 (parked +0.79..0.85). Never negative.
- s27 L1: water 236; L 363-900 (parked -1.02); never positive -> right missed. Only "right-missed" failure.
- s29 L10: water 387 (slow stage 0, typical 189); spout to -0.46/-0.47 (544-624), min -0.520 (0.004 rad short of LEFT
  threshold 0.5236!) then reversed to R 703-900.
Successes: first side reached 283-486, second side 335-638. 9 go L first, 8 go R first.
- No failure ever turned the water OFF once it was on.

## Scene grouping (layout == style here)
- L9: s0 s3 s13 s20 s26 FAIL; s2 s18 ok   (5/7 fail)
- L8: s4 s10 s17 s22 FAIL; s8 ok           (4/5 fail)
- L10: s7 s15 s29 FAIL                     (3/3 fail)
- L1: s27 FAIL; s23 ok
- L2 (s11,s14), L4 (s21,s24), L5 (s1,s6,s9), L6 (s16,s19,s25), L7 (s5,s12,s28): all success (13/13)
=> 12/13 failures in layouts 8/9/10.

## Failure JSONs written (13/13) — classification
No water (stage 0 root cause, 4): s7 actuation_insufficient (handle 0.346<0.40), s17 actuation_insufficient (0.23),
  s13 missed_grasp (closes beside lever 104/268, then hover), s20 stall_at_fixture (at handle 124-200, 0.02; hover 300-900).
  s7/s17 then did spout motions with water off (skipped_or_wrong_stage).
Water on, one side missing (stage 1, 9):
  (a) swing toward missing side stopped short, then reversed: s0 (back to centre +0.01), s3 (-0.365), s29 (-0.520, 0.004 short!)
  (b) push to first side, then hold/press there, then disengage: s4 (press right 430-620), s26 (hold 348-590, hover after), s27 (left; sat at tip 640-900, moved only 0.28 rad)
  (c) leave the faucet after first push and wander: s10 (back-corner stare 300-760), s15 (big excursions 400-900), s22 (235-785)
Key contrast: in 15/17 successes, gripper contact with the spout is ONE continuous segment from the first push
through BOTH sides (second side 52-227 steps after the first, median ~90). Exceptions s23 (released 417, came back
599-640, recovered) and s28 (27-step gap). In every water-on failure the contact ended or froze after the first side.
Time: not the limit except s22 (right at 858). s29 had slow stage 0 (387) but after the near-miss it held right 200 steps.
No failure turned water off by accident.
Measurement notes: arm-link contacts move the spout without gripper_touch intervals (s15 628-648, s22 430-500).
Packet has no spout-angle timeline (only end washed_loc / spout_ori) -> recovered from saved states.

## Successes (17 JSONs written)
- Pattern: lever -> raised "overview" pose (wrist looks down at the stream) -> take spout -> side A -> side B in one
  continuous contact. Order: 9 L-first, 8 R-first. Gap between sides 52-227 (median ~90).
- Odd ones: s2 (held right ~170 steps, then swept left; same layout as s0/s26 which let go instead), s8 (L8, slow creep
  right 280-472, stage 1 416 > p90), s23 (released after left at 417, returned 599 and swept right; same scene as s27,
  but left angle only -0.56 vs s27's -1.02), s28 (stage 0 286: pressed neck, handle stuck 0.13 for 80 steps, shifted to
  lever and recovered; contrast s17 which abandoned at 0.23).
- Narrow credits: s18 R +0.54, s16 R +0.55, s23 L -0.56, s19 L -0.57 (threshold 0.524) -> s29's -0.520 is luck-level.
## Tooling
- packet value_every_16_steps shows 6.2832 when the handle is a hair below 0 (wrapped angle, not |angle| as the brief
  says): s2@96, s11@160, s13@112, s20@128. done_intervals/water_on are correct.
- gripper-only touch intervals miss arm-link contacts that move the spout (s15 628-648, s22 430-500, s28 475-502).
- close_events.on_nothing is always true here (no task objects) -> uninformative.
