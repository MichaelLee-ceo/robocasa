# SearingMeat diagnosis notes (running)

Failures (21): fis0: s0 s1 s2 s3 s4 s6 s9 s10 s11 s12 s13 s16 s17 s18 s19 s21 s24 s26; fis1: s29; fis2: s25 s28
Successes (9): s5 s7 s8 s14 s15 s20 s22 s23 s27

## Per-episode quick log
- s0 (L9, pan_6): missed grasp at handle END on high shelf (115 air, 375 air); fingers stay closed 392-2848 hovering near pan. Pan never moved.
- s1 (L5, pan_5 long tan handle): partial pinch at handle neck 116 (w 0.028), slipped 205, fingers closed for rest; went to range/fish empty, pushed fish w/ closed fingers.

## Patterns
- Successes grasp the handle at its base next to the pan body, handle running toward the camera between fingers (all_success_grasp_wrist.png). Failures s0/s1/s3/s9/s17 are at the handle END or with handle running away (all_failure_grasp_wrist.png).
- pan_5 (long tan handle): s1 F, s3 F, s9 F, s17 F, s15 S, s23 S(1970 steps) -> 4/6 fail, 1 slow.
- After a failed grasp, the fingers often stay closed for the rest of the episode (s0, s1).
- lift_m is ~0 in successes too (pan starts at z=1.44 in upper cabinet and goes DOWN to stove); don't use lift as grasp evidence. Use close width ~0.03-0.04 + duration of touch.
- s2 (L9, pan_8): pinched handle TIP (121, w 0.037), pulled pan out, slipped ~180, pan fell to counter upside down. Continued: fish dropped on bare hob 960, turned 3 knobs incl task knob (0.353 end). dropped_in_transit + continued_empty.
- s3 (L9, pan_5): held handle TIP, pan hung tilted, set down at rear-left of 'left' burner; end 0.1512 m (official near-miss by 1 mm). Then shrimp missed at plate rim (685-835), fingers closed, wandered. root st0 placed_off_target + criterion issue; medium conf.
- s4 (L8, pan_7 flat cast-iron handle): closed ABOVE flat handle 178, dragged pan off shelf -> fell on edge vs toaster oven. 1500 steps of re-grasps; lifted at 1697, dropped to FLOOR ~1796. Then shrimp on bare burner + task knob 3.13 rad.
- s6 (L5, pan_4 black): handle lying ACROSS the image (along finger-closing axis) at 166 & 794 -> slid off twice; hovered 192-650; then went to shrimp, dropped it near range front. Pan in cabinet.
- s9 (L5, pan_5): grasped handle TIP 141, pan hung vertical, held in front of range 280-576 never laid flat, slipped 577, fell 1.15 m. Then shrimp miss & hover.
- s10 (L8, pan_6): stage 0 slow (handle END 148, diagonal slips 944/1276, success 1391 handle toward camera), placed 1788 rear_right ok. Stage 1: OPENED over pan 2224-2232 but shrimp STUCK to open finger; carried it to knobs, turned WRONG knob front_right (task rear_right). fis=0 is artifact of gripper-near clause. root st1 'other: stuck'.
- s11 (L2 induction 5-zone, pan_7): good grasp; pan put on rear_right zone instead of rear_center (0.184). Fish in pan 856, CORRECT knob rear_center 3.07. placed_off_target (adjacent zone, approached from the right).
- s12 (L7, pan_6): handle across image along closing axis, single close 115 on nothing; fingers closed & wandered 2700 steps, never retried.
- s13 (L9, pan_7): held handle TIP (hole), pan hung vertical during base rotation, slipped 262 fell off counter. Steak on bare left burner + task knob on.
- s16 (L6, pan_4 black): handle across image (closing axis), close 169 nothing; fingers closed & hovered in front of pan 2700 steps.
- s17 (L8, pan_5): grasp at neck, pan hung; lowered near front_left (at_target 629-631) but dragged to rear_right while setting flat; release 669 0.32 m off. Shrimp then put on EMPTY front_left grate (policy believes pan there). Knobs touched, none turned.
- s18 (L9, pan_6): good grasp; pan set between left & front_center, 0.233 from front_center (target), 0.11 from left. Fish ok. Knobs left, front_center(task), front_right all on. placed_off_target.
- s19 (L6 electric, knobs on rear console, pan_4): pan at front_left while held (0.068@512, at_target 541-558) then pushed right before release 582 (0.19). Fish ok. WRONG knob rear_left 2.04. release_timing + wrong_fixture_element.
- s21 (L4 induction 5-zone, pan_8): pan on front_left while held (277-290) then pushed back, release 321 at 0.176. Skipped shrimp, turned WRONG knob front_right; hovered over shrimp w/o closing; wandered. release_timing.
- s24 (L4 induction, pan_8): pan passed over front_center (0.104@512) but set on rear_center (0.193 from target). Steak ok, task knob turned FIRST (correct), then 3 more knobs. placed_off_target.
- s26 (L9, pan_7): good grasp; pan set between left & rear_center (0.19). Shrimp released over bare grate 0.55 m from pan. Re-grasped pan 815-880 (1 cm). Knobs rear_right then rear_center(task). placed_off_target.
- s29 (L10 electric, pan_7): pan fine at 381. Shrimp released over near rim/handle junction 655 -> lies on rim 0.107 m from centre (not in pan). Task knob rear_left on at 1248. Root st1 placed_off_target; near-miss official.
- s25 (L6 electric, rear console knobs, pan_7): stages 0,1 fine (383, 628). Turned REAR_RIGHT (960-1504) instead of front_right; stalled holding a knob 2000-2900. wrong_fixture_element.
- s28 (L7 gas 5-burner w/ center, pan_8): pan at border center/rear_right (0.145/0.108): stage0 counts. Fish ok. Lit REAR_RIGHT (under pan) first; turned task knob center on then back OFF twice (1632-1696, 1949-2016); rear_left later.
  OFFICIAL QUIRK: check_obj_location_on_stove returns first burner in STOVE_LOCATIONS order within 0.15 & on -> rear_right before center -> success impossible while rear_right on.
- s5 SUCCESS (L7, pan_4): pan FLIPPED during carry (tipped 216-688, tilt 179), placed UPSIDE DOWN; drumstick on upturned bottom -> counted. FALSE POSITIVE (no upright check). Compare frames/s5_upsidedown_vs_upright.png.
- s7 S: clean; stage0 524 (approach), stage2 363 (approach to knob row). stage1 undo 820-956 = contact flicker.
- s8 S: clean, fast.
- s14 S: clean (378/651/833).
- s15 S: pan_5 grasped near BODY (unlike failed pan_5). Knob: rear_left (task) AND front_left moved together 736-768; front_left element visibly lit. Success with extra burner on.
- s20 S (L9, pan_7 grasp at junction): clean; stage2 burner on at 564 but success 805 (gripper-far clause; stayed at knob near meat). close_events missed pan grasp (width only 0.06).
- s23 S (L1, pan_5): stage0 1970! slip 327-352, pulled out at 721 then dropped ~792 onto counter (upright, handle to robot); re-grasped from counter 1491 at body-side, placed 1949. Recovery works when pan lands upright on counter within reach.
- s22 S, s27 S: clean. s27 same kitchen/target as s24 (front_center, L4) -> success set it on front_center; s24 put it on rear_center.

## Cross-episode findings (after all 30)
- all_first_grasp_wrist.png: tip/end grasps (only handle visible in wrist): s0 s1 s2 s3 s9 s13 s17 s24 (+s10 first) -> 0/8 success. across-axis: s6 s12 s16 (s10 2nd/3rd). Base grasps (pan body visible above fingers): 16 eps -> 8 successes.
- pan min_eef_dist_to_centre: tip grasps >= 0.10 (s1 .101 s2 .186 s3 .173 s9 .174 s13 .103 s17 .127), base grasps 0.025-0.08.
- placement off: center-column targets s11 s18 s24 s26 off, s5 0.140 s28 0.145 borderline, s27 ok.
- knob: first-turned knob != task in s10 s15 s18 s19 s21 s25 s26 s28 (front/rear swaps s10 s15 s19 s25).
- after empty-handed miss: meat placed on bare burner + task knob on (s2, s4, s13).
- official: s5 false positive (upside-down pan); s28 burner-order quirk; s3 1.2 mm; s29 rim; s20 gripper-far delay; s15 extra burner.
- tooling: close_events miss thick-handle grasps (s13 ~110, s20 ~90, s24 ~200); s3 close 130 flagged on_nothing though it became the pan grasp.
