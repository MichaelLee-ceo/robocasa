# PrepareCoffee: stage-level diagnosis (GR00T N1.5, 30 episodes, 6 succeeded, 24 failed)

Task: take the mug from an upper cabinet or shelf, put it under the coffee machine's dispenser (stage 0), then press the start button (stage 1). Horizon 1200 steps.

What the official checks require (from the code):
- **Stage 0:** the mug centre is within 0.04 m (xy) and 0.10 m (z) of the machine's `receptacle_place_site`, and the gripper is more than 0.25 m from the mug.
- **Stage 1** is `_check_success`: stage 0's placement, plus the gripper more than 0.25 m from the mug, plus `_turned_on`, plus the gripper more than 0.15 m from every start-button geom.
- **`_turned_on`** latches on the first gripper contact with a start-button geom, whether or not the mug is there.

Files:
- per-episode results: `PrepareCoffee_s<N>_single.json`
- contact sheets: `frames/`
- running notes: `_notes.md`

## Counts

| root-cause stage | episodes |
|---|---|
| 0 (get the mug under the dispenser) | 16: s4, s5, s6, s7, s9, s12, s14, s15, s19, s20, s21, s23, s24, s25, s26, s29 |
| 1 (press the button and withdraw) | 8: s0, s2, s3, s10, s11, s13, s17, s18 |

The packet's `first_incomplete_stage` differs from the root-cause stage in two episodes:
- **s5:** the packet says stage 1, but stage 0 is the root cause. The mug lies on its side under the spout and still counts as placed.
- **s17:** the packet says stage 0, but stage 1 is the root cause. The mug was placed and the machine turned on, but the gripper never left.

s0 and s2 also show `first_incomplete_stage = 0`, for the same release-clause reason.

| primary category | n | episodes |
|---|---|---|
| missed_grasp (in the cabinet) | 5 | s12, s15, s19, s20, s25 |
| other: mug tipped in the grip during placement | 4 | s5, s9, s21, s23 |
| placed_off_target | 4 | s4, s7, s14, s29 |
| dropped_in_transit (slipped out of a rim pinch) | 2 | s24, s26 |
| release_timing | 1 | s6 (held at the target 130 steps, pushed past it, tipped) |
| other: missed button press | 4 | s3, s10, s13, s18 |
| other: no retreat after the button press | 3 | s0, s2, s17 |
| wrong_fixture_element | 1 | s11 (pushed the portafilter handle for 480 steps) |

No failure was caused by running out of time. Every stage-1 failure had 450-800 steps left after the placement.

## Placement distances (holds[].dist_to_target_at_release_m, then 2 s later)

| episode | release (m) | 2 s later (m) | what the video shows |
|---|---|---|---|
| successes | 0.003-0.036 | 0.005-0.036 | s27 0.003, s28 0.010, s1 0.012, s8 0.014, s16 0.021, s22 0.036. Upright, released 38-68 steps after the mug first reached the target. |
| failures with stage 0 met | 0.011-0.036 | | s17 0.011, s18 0.012, s13 0.031 (re-grasped and nudged to 0.012), s3 0.018, s10 0.018, s0 0.021, s5 0.026 (lying on its side), s2 0.034, s11 0.036 |
| s7 | 0.043 | 0.069 | Under the portafilter at release, 3 mm outside. The fingers dragged the rim on the retreat (368-373). |
| s29 | 0.047 | 0.043 (end 0.043) | Tall mug held tilted against a low portafilter for 180 steps. Settled upright just in front of the group-head centre. |
| s9 | 0.046 | 0.042 | Lying on its side on the tray after a 650-step hold of the tipped mug |
| s4 | 0.051 | 0.066 (end 0.115) | Front-right edge of the tray, in front of and beside the nozzles. Later knocked over by the hand at the button. |
| s14 | 0.072 | 0.088 | Beside the group head (left in the wrist view), upright |
| s6 | 0.081 | 0.079 | Under the dispenser 426-558 while held, then pushed past it. Tipped in the grip, fell on its side at the left of the tray. |
| s23 | 0.100 | 0.126 | Tipped in the grip at the tray, released, fell |
| s26 | 0.178 | 0.186 | Slipped out of a 6 mm rim pinch beside the machine, about 12 cm above the counter |
| s21 | 0.240 | 0.287 | Tipped in the grip, carried for 420 steps, dropped from about 40 cm |
| s24 | 0.285 | 0.278 | Slipped out of an 8 mm rim pinch in transit |

So the cases off by more than 4 cm are mostly not precision misses. They are the mug rotating or slipping in the fingers (s5, s6, s9, s21, s23, s24, s26). The genuine precision misses (s4, s7, s14, s29) were 4.3-7.2 cm off at release, mainly in front of or beside the spout.

## Recurring failure mechanisms

### 1. A rim-pinch grasp that lets the mug rotate or slide out at the machine (9 of 16 stage-0 failures)

**The grasp.** The policy always takes the mug by pinching one side of the rim wall: finger width 0.006-0.021 during the hold, in successes too.

**Failures.** The mug pitches over in the fingers as it meets the tray, the slot or the portafilter while it is still held. It is then held tipped for a long time, or released on its side:
- s5: narrow slot, 252-283
- s6: pushed past the dispenser, 558-571
- s9: lowered to the tray, 240-283, then held tipped to 939
- s21: 340-385, then carried tipped until it was dropped at 808
- s23: about 304, held tipped to 535

Other effects of the pinch:
- s24 and s26: the mug slid out of a very narrow pinch (0.006-0.008) in transit.
- s29: a tall mug was held tilted against a low portafilter for 180 steps.

**Long holds.** Failures hold the mug for 379-828 steps (s3, s6, s9, s17, s21, s23, s29); successes hold it for 174-281.

**What successes do differently.** They lower the mug straight down onto the tray and open within 38-68 steps (`frames/cmp_L7_insert.png`). The same machine and mug model as s5 succeeded in s28 that way.

**The policy never rights a tipped mug or re-grasps a dropped one.**

### 2. Premature close in the cabinet, then the gripper never reopens (5 episodes)

| episode | first close | what happened |
|---|---|---|
| s15 | 88 | Closed a few cm short of the mug wall. Width 0.001 to the end; 1100 steps of approach and retreat with closed fingers. |
| s20 | 78 | Same pattern |
| s25 | 127 | Same pattern |
| s19 | 132 | Closed on the bulging body or handle of the round mug, slid off, and never reopened |
| s12 | 105 | Short of the wall. Then five more closes on air 10-20 cm away. |

- Three of the five involve the tall tapered "tea" mug_8, which went 0/8 overall: 3 never grasped, 1 slipped, 2 tipped, 1 precision miss, 1 button miss.
- In successes the fingers close around the rim from above (`frames/cmp_first_close_wrist.png`).

### 3. Pressing the button: missed presses (4), wrong control (s11, plus s7/s23 secondary) and not withdrawing (3)

**Missed press.**
- s10 and s18: the fingers came at the button face head-on, open, and the button sat between the two fingertips (wrist view). Contact was on the machine body only.
- s3 and s13: closed fingertips at or just below the button. Body contact only.
- After one attempt, each backed off and hovered or wandered for the remaining 500-800 steps. s13 and s18 reached for the cabinet distractor.
- The successes on the same capsule machine (layout 8: s8, s22, and s17's press) came at the round face from the right side, edge-on (`frames/cmp_layout8_button.png`, `cmp_layout9_button.png`). s22 itself missed head-on at 501-520, then pressed from the side at 680.

**Wrong control.**
- s11: the open gripper pushed the portafilter handle for 480 steps, with the start button visible above it.
- s7 and s23 (layout 10, CoffeeMachine051): the hand went to the lower cup-icon control or gripped the central dial. The start-button geom is high on the upper left. These are secondary, because their mugs were already off target.

**No retreat.**
- s0 and s2 (layout 9) pressed at 443 and 440, then rubbed the machine's lid until timeout.
- s17 pressed at 793 and stayed at the button face.
- In all three the mug was placed and the machine was on. Only the "gripper more than 15 cm from the button and more than 25 cm from the mug" clause failed.
- Successes also linger after the press: 20-240 steps (s8 174, s22 236, s28 182).

**Never attempted the button:** s5 (tipped mug) and s29 (mug 3 mm outside the tolerance). Neither ever touched the machine.

### 4. Scene dependence

| layout / machine | successes | notes |
|---|---|---|
| L9, brown CoffeeMachine036 with two stacked buttons | 0/5 | s20 grasp; s13 and s18 missed presses; s0 and s2 never withdrew |
| L10, CoffeeMachine051 | 0/4 | The machine sits at or beyond the right edge of the agent camera. s7 and s23 worked the dial or lower control; s29 never tried; s15 grasp. |
| L8, capsule CoffeeMachine102 | 2/6 | |
| L2, CoffeeMachine045 | 0/3 | |
| L3, L5, L6, L7 | 4/9 together | |

| mug model | successes |
|---|---|
| mug_7 | 3/5 |
| mug_5 | 2/6 |
| mug_9 | 1/7 (tipping: s5, s6, s9) |
| mug_4 | 0/4 |
| mug_8 | 0/8 |

Small n. These are hypotheses, not findings.

## Official-criterion issues

1. **No upright test.**
   - s5: stage 0 counts as complete from 367 to the end with the mug lying on its side (tilt 83 deg). Pressing the button would have scored a success.
   - s9: the mug ended on its side at 0.042 m, 2 mm from the same false positive.
2. **4 cm near misses.**
   - s29: ends at 0.0429 m, upright under the group head and slightly forward. By eye it is "under the dispenser".
   - s7: released at 0.043 m with the rim visibly under the portafilter spout. The later drag to 0.069 would have failed it anyway.
   - s4: 0.051 m. Visibly off the nozzles, so that call is fair.
   - s22 (a success, 0.036) and s11 (0.036) passed by about 4 mm.
3. **The machine turns on regardless of the mug.** s4 and s26 pressed the button with no mug under the spout.
4. **The release clauses distort stage timing.**
   - s0, s2 and s17 look like stage-0 failures in the packet although stage 0 was done physically.
   - Stage 0's `first_complete_step` lags the release by up to 560 steps: s11 340 -> 902, s28 266 -> 816, s22 395 -> 914.
   - The typical stage durations (median 506 / 31) are therefore "until the gripper left", not "until placed / pressed".

## Packet vs video (tooling)

No case where the packet contradicts the video in a way that suggests a measurement bug. The at-target, tipping, turned-on and touch intervals matched the video everywhere I checked. Limitations:

- **`holds[].release_step` also fires for slips.** In s24 (190), s26 (303) and s21 (808) the fingers closed further (0.008 -> 0.002) instead of opening. Only `height_drop_after_release_m` reveals the drop. Suggest a flag for whether the width increased at release.
- **No gripper-to-button distance.** It is the actual blocker in s0, s2 and s17, and the cause of the slow successes s8, s22 and s28. There is also no record of which button geom was hit, or of the nearest-button distance on missed presses (s3, s10, s13, s18). Machine `gripper_touch_intervals` do not separate the button from the body or the portafilter (s11).
- **Scratchpad clashes.** The scratchpad is shared between agents, and same-named helper scripts there overwrote each other. One of my calls briefly wrote a sheet into `diagnosis/StackBowlsCabinet/frames/s0_liquid_zoom.png`; I deleted it. Another agent's call wrote a `frames/s4_overview.png` here, which I overwrote.

## Uncertainty

- **Finger-gap explanation (s10, s18) and "too low" (s3, s13).** Both are read from 4-step wrist frames, without depth.
- **CoffeeMachine051's start button.** I located it from the model file (upper left), not from a visible press.
- **s21 and s23.** The machine is largely off the agent camera, so the mug's final pose comes from the packet.
- **s12.** Whether the first close was short of the wall or above the rim.
- **s28.** What the gripper did on the upper panel from 450 to 593 before pressing.
