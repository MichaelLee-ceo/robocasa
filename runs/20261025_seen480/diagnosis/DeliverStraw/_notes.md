# DeliverStraw running notes (agent scratch; cross-episode patterns)

## Criterion / tooling facts (verified in source)
- Official `DeliverStraw._check_success` = `check_obj_in_receptacle(straw, glass_cup, th=0.5)` AND `gripper_obj_far(straw)`.
  -> straw must TOUCH the cup and its body centre be < 0.5 m horizontally from the cup centre. No drawer condition,
  no "inside"/upright condition. th=0.5 m is ~10x the cup radius: a straw leaning on / lying against the outside of the cup counts.
- Drawer value = slide joint / (0.55 * drawer depth) (cabinets.py Drawer.get_door_state). NOT a fraction of travel:
  fully open reads ~1.82 (=1/0.55). Successes reach 1.6-1.8. Threshold 0.95 = ~52% of the physical travel.
  So s0 "0.853" = ~47% of travel; s7 0.375 / s8 0.36-0.40 = ~20% of travel.
- Straw tilt fields: rest pose in drawer reads 0-14 deg; held vertically / standing in the cup reads ~65-90 deg.
  So `tipped_over_intervals` (>45 deg) for the straw mostly mark "straw held/standing vertical", not "fell over". Ambiguous (roll about long axis).
- Straw end z ~0.01 = on the floor (s9, s12, s19, s25, s26, s29).

## Groups
- Stage-0 fail, drawer never moved (value 0): s1, s4, s6, s10, s11, s13, s14, s16, s18, s20, s22, s28 (12; s1/s14 touched it)
- Stage-0 fail, partial: s0 (0.853), s7 (0.38), s8 (0.36-0.40)
- Stage-1 fail (drawer done): s2, s3, s5, s9, s12, s15, s19, s21, s23(drawer slow, 1055), s24, s25, s26, s29
- Success: s17 (layout 9), s27 (layout 2)

## Per-episode findings
### Stage-0 failures (video checked, overview + dense)
- s0 L10: hooked bar handle w/ half-closed fingertips 247-315, drawer to 0.853 (~47% travel) at 320, fingers closed on air 308-320 and STAYED CLOSED to end. Hovered above drawer w/ straw visible 600-780, never descended; wandered from ~840. Never touched straw.
- s1 L7: never closed gripper all episode; reached drawer column 440-580 (touch 552-557), handle never between fingers; hovered/wandered. stall_at_fixture.
- s4 L9: handle above-left of finger gap 232-352, closed on air 336-352 (no drawer touch), drove to dining counter 560-1000 w/ closed empty gripper, hovered at cup. missed_grasp -> continued_empty.
- s6 L7: grasped vertical handle of TALL CABINET DOOR at right (169-216), swung it open 216-300; hovered in that cabinet rest of episode. wrong_fixture_element.
- s7 L10: hook pull 248-287 -> 0.38 (~21% travel), closed on air 279; hovered over the narrow gap ~1400 steps (straw visible), brushes 1294-1318,1637; no re-grasp. actuation_insufficient + stall.
- s8 L9: handle post right of finger gap 244-260, dragged drawer to 0.39 then fingers closed on air 266 and arm swung low; hovered/pressed on drawer front 1047-1068, 1265-1278; no re-grasp. actuation_insufficient + stall.
- s10 L9 corner: handle at finger gap ~200 but no close; rose, wandered drawers, went to floor level 520-1400; closes on air 797,1264,1582. stall_approach_retreat.
- s11 L2: closed on air at a long bar handle 251 (bar not between fingers), stayed closed to end; hovered over cooktop; drove to dining counter 1440-1696 empty. missed_grasp + continued_empty.
- s13 L10: opened WRONG drawer (left/corner one) ~160-200, searched empty drawer 280-680, closed on air ~680, left w/ closed gripper. wrong_fixture_element.
- s14 L2 island: pressed flat on target drawer face above its handle 228-464 (touch = target drawer), closed on air 459, closed to end; hovered/wandered. stall_at_fixture.
- s16 L8: grasped vertical CABINET DOOR handle 213, swung door open ~280-360, stuck at that cabinet to end. wrong_fixture_element.
- s18 L10 (other wall): passed drawer handle, went to double-door handles below, closed on air 230/437, stalled at doors 700-1300, closed on air 1381 over counter. wrong_fixture_element.
- s20 L10: opened WRONG (left/corner) drawer ~170-280, closed on air inside empty drawer 346, left w/ closed gripper, wandered. wrong_fixture_element + continued_empty.
- s22 L9 corner: grasped gold handle 205, opened WRONG drawer ~280, searched empty drawer 360-1700. wrong_fixture_element.
- s28 L8 corner: hovered at a curved handle 200-850 (no close), grasped 861 and opened WRONG drawer ~900-1000, searched it. wrong_fixture_element + stall.
- Wrist at first handle close (frames/all_handle_close_wrist.png): success = handle post centred between fingertips. s0/s7 bar handle lies ACROSS the finger axis (hook only). s4/s8/s16 handle off-centre.
- Dense sheets later confirmed: s13/s20 (L10) opened the drawer on the LEFT (perpendicular) wall; s15/s29 (same L10, straw x~1.4) opened the right one -> same scene position, different outcome (not purely scene-determined). s22 (L9) opened a different drawer; s28 (L8) hovered at a handle 200-850, then opened a wrong drawer ~900-990. s16 swung a cabinet door 216-360.

### Stage-1 failures (drawer done)
- s2 L10 Straw012 bendy: grasp 370, at cup ~700-870 straw moved left/above/right of cup, no contact; released 942 15 cm off on counter. stall_holding_at_target -> placed_off_target.
- s3 L9: at cup 740-797 straw vertical beside cup, released 797 at 5.3 cm (no contact); RE-GRASPED 805, lifted 13.6 cm, released 884 -> fell onto stool. placed_off_target + disturbed_placed_object.
- s5 L9: released 977 at 2.5 cm, straw diagonal on rim, touched cup 983-984, then fell off counter onto stool. placed_then_fell_or_slid.
- s9 L7: grasp 263/283 at straw end; slipped out 331-337 at the drawer, fell to floor; drove to cup EMPTY, closes on air 886, 1515. dropped_in_transit + continued_empty.
- s12 L8: drawer 310 (slowish, closes on air 243/263/308); grasp 483/493; at cup 960-1260 (300 steps), touched cup 1223-1231 only; carried away, released 1517 0.23 m -> floor. stall_holding_at_target.
- s15 L10: straw held ~30-45 deg from horizontal; lay across rim 795-836 (at_target), released 826, slid off onto counter 8.7 cm. placed_then_fell_or_slid. Near-miss: if straw had stayed across rim + gripper away -> official success w/o straw inside.
- s19 L8: drawer 380 (handle work 165-383); straw: closed on it 586 no lift, hovered 600-970, close on air 839, grasp 971/991; at cup 1380-1600 beside it (contacts 1573-1581), released 1640 0.54 m -> floor. stall_holding_at_target (+ time lost upstream).
- s21 L4: grasp 351, slow nav (369-976), at cup 1140-1344 straw vertical above/beside cup, never contacted; released 1350 at 5.9 cm onto counter. stall_holding_at_target.
- s24 L7: base parked ~640 at island corner ~1.4 m from cup and never moved again; released straw on counter 891, regrasped 954/1001/1566; end 1.15 m. other: navigation stopped short.
- s25 L8: straw slid through fingers during drive (width 0.016->0.001 over 400-820), dropped 822 to floor 1.36 m from cup. dropped_in_transit.
- s26 L10: drawer 289; hovered above straw 440-630, close on air 613, grasp 637; base did not move 700-1000, then turned toward fridge (1300-1660); straw dropped 1636 to floor 3.3 m away. wander_away.
- s29 L10: grasp 345, at cup ~680-910 straw (transparent mint) vertical right of cup, never contacted; released 930 at counter edge 0.35 m -> floor. stall_holding_at_target.
- s23 L2: stage 0 root cause: first pull 0.38 at 192 (fingers slipped off bar handle, close 174 to 0.003), then hooked/pushed drawer from inside 522-639, 806-850, 1021-1074; done 1055. Straw grasp 1219, base slowly turning, 1.48 m from cup at timeout. actuation_insufficient + ran_out_of_time.
### Successes
- s17 L9: drawer 205, grasp 368, nav 371-588, straw vertical above cup ~700-743, released 743 at 3.4 cm -> stands in cup (leaning). success 765.
- s27 L2: drawer 179, grasp 356, nav 358-691, bendy straw vertical over cup 740-778, released 778 at 0.5 cm; lower end in cup, bent part over rim. success 797.
- Success = straw brought VERTICAL directly over the cup and released within ~20-40 steps. Failures hover 100-300 steps beside the cup (few cm off) then release beside/away.
- Wrist cam (frames/all_at_cup_wrist.png) looks forward along the straw; cup usually NOT visible below the fingers -> alignment depends on the agent view where cup+straw are a few px (hypothesis).
- Gripper width ~0.001 while still holding straw (s23 end, s25 before drop): width can't distinguish 'closed on nothing' from 'pinching a thin straw'.
