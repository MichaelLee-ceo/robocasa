# PreSoakPan running notes

## Tooling facts (checked in code)
- Stage 0 / 1 in the packet are NOT "pan" / "sponge": scripts/atomic_steps.py uses _order_step(["obj1","obj2"], ...):
  stage 0 = the FIRST of {pan, sponge} fully in sink + gripper >25 cm from it; stage 1 = BOTH in + gripper far.
  _task.json's definitions (0 = pan, 1 = sponge) are a simplification. Packet objects[*].target_of_stage = 0 for both.
- Stage 2 = env._check_success(): water_on and both obj_inside_of(partial_check=False, th=0.05) and gripper >25 cm from both.
- water_on = 0.40 < handle_joint(mod 2pi) < pi; handle joint range [0, 0.5] in all sink models (damping 0.5, frictionloss 1).
  So the lever must be pushed to >= 80 % of its travel.
- fixtures.stage2:sink.value = |handle_joint mod 2pi|: a tiny negative joint value (limit penetration) shows as 6.283.
  "sink max = 6.283" in s0, s13, s19, s20, s29 is this wrap artefact, not a real turn.
- fixtures.sink.gripper_touch_intervals = contact with ANY geom of the sink fixture (basin, rim, faucet, lever), not only the lever.

## Episode list
Failures (12): s4 s8 s12 s16 s17 s19 s20 s22 s23 s26 s27 s28
Successes (18): s0 s1 s2 s3 s5 s6 s7 s9 s10 s11 s13 s14 s15 s18 s21 s24 s25 s29

## Layout dependence (success / n)
L1 0/3 (s12 s23 s27) | L8 1/6 (s4 s8 s17 s22 s28 fail; s10 ok, slow st2 452) | L6 1/3 (s16 s19 fail; s25 ok)
L9 3/5 (s20 s26 fail) | L2 3/3 | L5 4/4 | L10 4/4 | L4 1/1 | L7 1/1

## Per-episode findings
- s22 (L8, single-lever faucet with a small paddle lever right of the spout base): stage 0 pan (handle grasp) 140-259,
  sponge 368-507, both in. Gripper at the lever 754-953 (~200 steps) touching, sweeps it 968-992 -> handle 0.214 rad
  (needs >0.40), lever slips out of the finger gap at ~992 and the arm leaves. Re-approaches 1220-1256 and 1508-1556
  without turning it up; the last visit pushes it back to 0.119. s10 (same layout, success) has the same 200-step
  dwell 752-960 then sweeps with the lever kept between the fingers to 0.5 at 1013.

## Criterion code finding (important)
OU.obj_inside_of compares dot(u, p) with dot(u, p0) - th where u = px - p0 is NOT normalised -> the effective
tolerance is th / |edge| metres: for basins (0.25-0.65 m wide, 0.25-0.45 deep, 0.1-0.2 tall) that is ~0.08-0.2 m
horizontally and ~0.25-0.5 m vertically. Explains: sponge on the counter with its centre ~10 cm outside the basin
region counts as inside (s16, s19; dist_to_target_start 0.103 / 0.094), held objects count as inside well above the
basin (s26 sponge from 564 while held, s12 pan from 455 while held).

## All failures done (JSON written)
| tag | layout | root stage | category | one-line |
| s4  | L8 | 2 | actuation_insufficient | lever 0.164 (wrist-roll turn), slipped at 864, wandered 736 steps |
| s22 | L8 | 2 | actuation_insufficient | 200-step dwell then sweep to 0.214, slipped at 992; 2 re-approaches, last pushed back to 0.119 |
| s8  | L8 | 2 | stall_at_fixture | pressed on the round spout head 643-838 (lever below, untouched), then wandered |
| s28 | L8 | 2 | stall_approach_retreat | spout head 770-858, hover near lever 880-1120 w/o contact, retreat/re-approach |
| s17 | L8 | 2 | wander_away | slow stage 1 (622) but 604 left; wandered 270, spout head 45 steps, wandered |
| s12 | L1 | 2 | stall_at_fixture | hover beside stick lever 920-1330, 10 steps light touch |
| s23 | L1 | 2 | stall_at_fixture | 180 steps contact at lever base, 0.003 rad |
| s27 | L1 | 2 | stall_at_fixture | 740 steps hover beside stick lever, zero contact |
| s26 | L9 | 2 | stall_at_fixture | stage 1 882 steps (hand wedged on faucet carrying sponge to pan); 300 steps pressing on column, 0 rad |
| s19 | L6 | 2 | stall_approach_retreat | two approaches to lever housing, each closes fingers on air; criterion: sponge at step 0 |
| s16 | L6 | 1 | missed_grasp | pan handle tip under one finger at 179-192; sponge counted at step 0 |
| s20 | L9 | 1 | stall_pre_grasp | gripper stayed closed after sponge slipped (176-1436), poked pan handle 430-793, then turned water on w/o pan |

Faucet types: L8 = horizontal spout w/ round spray head + small paddle lever at column base under the spout arm.
L1 = tall black gooseneck + thin black stick lever beside column base (dark on dark). L9 = black pull-down gooseneck,
small stub lever on the column side; successes push it from above (wrist pitched up at the wall). L6 = side lever
housing (cylinder with red dot) + hanging stick handle.
No failure had the pan/sponge leave the basin during stage 2 (all at_target to 1600).

## Successes done (light): all 18 JSONs written. Odd ones: s14 (partial push 0.327 at 928, back at 1567, success 1574),
s10 (L8 dwell 752-960 then good sweep), s11 (gradual lever pushes 811-960), s24 (premature faucet detour 540-580),
s13/s21 (sponge put on pan on counter, then pan carried), s25 (300-step hover beside lever, then push).
Coordinator note applied: sink value ~6.283 = wrap artefact (min(v, 2pi-v)); touch intervals = finger contacts only
(checked: every handle change is within 24 steps of a finger contact).
Summary written: _summary.md
