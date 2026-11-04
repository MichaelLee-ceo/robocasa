# SteamInMicrowave running notes (diagnosis agent)

Scripts: private scratch at /tmp/claude-1007/.../scratchpad/steam_mw/ (summ.py packet summary, fr.sh frame wrapper).
NB: scratchpad is shared with other agents; keep own files in steam_mw/.

## Episode list
Successes (10): s3 s4 s9 s13 s17 s18 s19 s21 s24 s26
Failures by first_incomplete_stage (packet):
- fi=3 (door closed, button not pressed): s0 s2 s5 s7 s11 s15
- fi=2 (door not closed): s8 s14 s23 s25 s27 s29
- fi=1 (bowl never in MW): s1 s10 s22  (s1: door closed 844 without bowl!)
- fi=0 (veg not in bowl at end): s6 s12 s16 s20 s28  (s6/s28/s20 later stages held -> veg lost after?)

## Per-episode findings
- s0 (L9): stage 3. Door closed 1057 (343 left). Arm retracted low, looked at MW from afar, one reach to the door glass 1300-1364, no MW contact after 1052. stall_approach_retreat. Wrist rolled ~90deg.
- s2 (L9): stage 3. Door closed 825 (575 left). Pressed keypad repeatedly (10 contacts 873-1361) on number keys/border, START several key widths away. Wrist rolled ~90 deg vs s18 success. wrong_fixture_element.
- Reference: s18 (L9 success) pressed START at 1248 with normal wrist orientation (panel vertical in wrist view, START at fingertips). s9 (L5, over-range MW) START at fingertips 888-904.
- Microwave model per style (kitchen_styles/test, assumed): st1 MW066, st2 MW035, st4 MW083, st5 MW036, st6 MW034, st7 MW071, st8 MW084, st9 MW071, st10 MW073.
- Turn-on logic: any gripper contact with start_button geom while door < 90% open latches turned_on (stays until stop pressed or door reopened >=90%). Success needs closed (opening <= 0.005).
- s5 (L5, over-range MW like s9): stage 3. Door closed 1093 (307 left). Retracted 1124-1260, then 2 presses (1277, 1345-1354) on panel bottom edge between Stop/Reset and Start, START ~2-3 cm up-right of fingertips. wrong_fixture_element (near miss).
- s7 (L10 MW on tall cabinet): stage 3. Door closed 984 (416 left). Door-bar touches to 1052, hovered at number keys 1116-1152 (START ~10 cm aside) no contact, drifted up/away 1190-1400. stall_approach_retreat. Carrot touches fingers during bowl carry (fake veg 'hold' 576-799).
- s11 (L2 wall-stack MW like s3): stage 3. Door 751 (649 left!). Presses 828-892 on KITCHEN TIMER, START one key to the right. Then 1025/1138 other spots, wander 1200-1360. wrong_fixture_element (near miss).
- s15 (L10 like s7): stage 3. Door 880 (520 left). 5 presses 1021-1095 on Clock Time/Timed Defrost, round Start one key diagonal. Then hovered above keypad to end. wrong_fixture_element (near miss).
- s8 (L2): stage 2. Base oblique right of MW. Gripper pushed the open door from the inner/cavity side -> door to fully open (0.07->0 at 736), pushed against stop 730-1051, hovered after. other: wrong side of door.
- s14 (L2): stage 2. Gripper at the door free edge 722-1240, door crept to 0.2-0.34 then 0.51 by 1296, arm moved on to the keypad as if done. actuation_insufficient.
- s3 (L2 success): base also oblique; gripper reached the door's outer face/free edge at 812-840, base moved left, door closed at 903.
- PATTERN: L2 (MW035 built-in, left-hinged door opening toward robot): 2/4 door failures (s8, s14), 1 button near miss (s11), 1 success (s3).
- s23 (L8 over-range MW): stage 2 skipped. Bowl placed 892 (slow stage 1: first bowl grasp slipped 317-377). Went to keypad, never touched MW at all; hovered at keypad 1160-1400. L8 successes s4/s17 reached far left to the door's outer face by ~960/1028. skipped_or_wrong_stage.
- s25 (L6 over-range MW): stage 2. Door swept to 0.85 by 832 and left. START pressed ~848 (turned_on latched w/ door 15% open), then STOP ~1232. actuation_insufficient. NB turned_on can latch with door ajar.
- s29 (L10): stage 2. Door pushed to 0.94 by 976, left for keypad. Then 3 near-miss presses around Start 1064-1177 (turned_on still 0), hovered. actuation_insufficient + wrong_fixture_element.
- s27 (L1 only ep; over-range MW above electric range): stage 2. After bowl at 794 the arm parked over the cooktop ~40 cm below MW 880-1400, no door contact (one at 1388). stall_at_fixture.
- s22 (L4): stage 1 never official: bowl released 691 at front-right of cavity, centre inside (d=0) but bbox-corner check fails (x 7-17 cm off vs s24/s21). Policy then never touched MW, wandered low-left. placed_off_target + possible criterion issue. LOW confidence.
- s1 (L5): stage 1. Bowl carried tilted (tipped 552-647), released at the cavity lip 648, slid out and fell onto the range (upright, squash still in). Policy closed empty MW 844, pressed below Start/Stop row 1128-1245 (same near-miss as s5). placed_then_fell_or_slid.
- s10 (L8): stage 1. Bowl rim grasp 291, slipped out during arm swing 348, fell to floor w/ corn. Then hovered over counter, closed fingers at croissant distractor 958/978/1188. dropped_in_transit. Stage 0 false positive 1296-1400 (corn+bowl on floor).
- s6 (L5): veg lost during bowl lift: wedge placed near rim (0.11 m), bowl lifted by that rim tilts, wedge slides out 412-428 onto robot base. Rest done w/ empty bowl; START presses also below the row (3rd MW036 near miss). failed 0, root 1, undid_previous_stage.
- s28 (L7): tomato rolled out of the bowl (tilted ~59 deg while lifting to high MW) at 662. Rest done incl START at 1177 -> only spill cost success. failed 0, root 1, undid_previous_stage.
- s20 (L9): missed mushroom grasp at ~97 (closed beside it), no retry, rest done empty incl START 1272. missed_grasp_continued_empty.
- s12 (L7): 4 missed lime grasps 147-425 (closed on air, lime ahead of gap), 5th lifted at 610 & slipped back into sink 637; then empty bowl carried to MW, held tipped ~80deg at MW 1140-1400. missed_grasp.
- s16 (L6): bowl rim grasp 302 then twisted on edge over sink (tipped 404, 130 deg); garlic spilled into sink ~560; fumbled bowl in sink ~700 steps; heads to MW empty at ~1340, timeout. root 1 undid_previous_stage. Stage0 held while bowl on edge (no upright check).
- s19 SUCC (L6): stage 2 706 steps: pushed door fully open at 720 (as s8) and stalled against it to ~1248, then outer face, closed 1264-1372; closing sweep ended on START -> success at 1372 (28 steps spare).
- s24 SUCC (L4): after bowl 733 the arm retreated low-left below the open door 760-1110 (as failed s22) but came back at 1113; door 1209, START 1381 (19 steps spare).
- s4 SUCC (L8): bowl 1st close 371 failed, regrasp 535 (~165 lost). Presses: 1233 KITCHEN TIMER (near miss), 1381 START. 19 steps spare.
- s13 SUCC (L9): bowl 1st close 306 failed, regrasp 452. START 1204.
- s17 SUCC (L8): stage 0 slow 430: mushroom carried up/away (lift 0.84) before placing. START 1285.
- s21 SUCC (L4): stage 3 354 steps (retracted, then START 1332).
- s3/s9/s18/s26 SUCC: normal. s26 first keypad contact 1179 missed, 2nd 1214 hit START.
- Wrist-view compare at press (cmp_press_*): successes mostly see keypad upright-ish with START at fingertip; near-miss failures often rotated ~70-90 deg (s2, s7, s11, s15, s29), but L5 (MW036) is rotated in success s9 too -> roll is suggestive, not conclusive.
- Coordinator note: obj_inside_of tolerance = th/|edge| (often >10 cm) -> s22 false negative less likely to be pure bbox-corner artifact; keep low conf. gripper_touch_intervals = finger contacts only.
- s27 end: gripper parked below the open door's edge; contact 1388-1392 plausible (door edge in wrist view). Not a tooling bug.

## Layout tally (successes/total)
L1 0/1 (s27 F2) | L2 1/4 (s3 S; s8 F2 door-wrong-side; s11 F3 button; s14 F2 door 0.51) | L4 2/3 (s21,s24 S; s22 F1)
L5 1/4 (s9 S; s1 F1 bowl slid out; s5 F3; s6 veg spill) -- 3/4 START presses on MW036 land on bezel below row
L6 1/3 (s19 S lucky; s16 bowl twisted; s25 door 0.85) | L7 0/2 (s12 lime grasp; s28 tomato spill)
L8 2/4 (s4,s17 S; s10 bowl drop; s23 skipped door) | L9 3/6 (s13,s18,s26 S; s0,s2 button; s20 mushroom miss) | L10 0/3 (s7,s15 button; s29 door 0.94 + button)
