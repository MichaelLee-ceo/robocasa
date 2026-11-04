# GetToastedBread running notes

## Mechanics (from code: robocasa/models/fixtures/toaster.py, get_toasted_bread.py, runs/scripts/atomic_steps_seen11.py)
- Toaster lever value: 0 = up, 1 = fully down. Turns on when lever >= 0.90 (and cooldown 0); then held at 1.0 for 500 steps,
  then turned_on=False and the lever springs back up slowly (~150-170 steps from 1.0 to 0).
- Bread carriage moves with the lever: bread in the pressed pair drops ~5-7 cm (dz) when lever = 1. Bread NOT moving while a lever
  is at 1.0 => the pressed lever belongs to the other slot pair (two-pair toaster) or the bread is not seated.
- Two-pair toasters (state keys "0","1"): layouts 2 and 10. Single-pair: layouts 4, 8, 9.
- Stage 0 checker (_toast_done): bread-slot contact latched (cleared if bread grasped before toast latched); that pair turned_on ->
  toast latched; complete when that pair is off and its lever <= 0.1.
- env _check_success (= stage 1): toaster_on latched only if bread touches the floor of the slot pair that is on at that step;
  then bread in plate + gripper far. Note: the env does NOT require the pop-up.
- packet fixture done_intervals = "any pair turned_on" (toasting), NOT stage 0.
- Bread z/xy from run/.../_steps.npz states (found qpos index by matching start_pos) — my script
  scratchpad/gtb/bread_gtb.py; dz = bread height change vs start.

## Episode buckets (from packets + bread trajectory)
- Stage 0 complete (online log), stage 1 never: s1 816, s4 832, s9 1680, s10 1248, s12 1517, s17 842, s18 950, s25 912.
  In ALL 8 the bread was never touched after the pop-up (obj touch intervals empty; bread dxy 0).
- Toaster turned on but no stage 0:
  - s6: pair 1 on 378-877, bread (pair 0) did not move -> other pair's lever.
  - s7: pair 0 on 180-679, bread did not move -> other pair's lever.
  - s14: pair 1 on 479-978, bread did not move -> other pair's lever.
  - s8: bread pulled up 2.8 cm (207-256) and stuck there; pair 0 on 464-963 and pair 1 on 1040-1536 without bread moving; bread fell
    back at 1232.
  - s27: correct pair 0 pressed 496 (bread -5.8 cm), pair 1 also pressed ~640-672; pair 0 lever stayed at ~0.999 (never popped)
    to the end -> stage 0 never completes. Why stuck? check video.
- Lever never >= 0.9 (17): s0 (0.005), s2 (0.19), s3 (0.46), s5 (0.64), s11 (0.14, bread lifted 1.7 cm 256-272), s13 (0.32, then
  bread pulled out 288 and carried to the plate, released 797 at the plate, tipped), s15 (pair-1 lever 0.30, bread in pair 0),
  s16 (0.80 at 256; repeated partial presses), s19 (0.30), s20 (0.18), s21 (0.32), s22 (0.02), s23 (pair-1 lever 0.11),
  s24 (0.51), s26 (pair-1 0.46 + bread grasps), s28 (0.58), s29 (0.0, never touched lever).

## Scene grouping
- layout 2 (two-pair): s1 ok0, s6, s8, s11, s14, s23, s27
- layout 10 (two-pair): s18 ok0, s0, s2, s7, s13, s15, s20, s22, s26, s29
- layout 4 (single): s10 ok0, s17 ok0, s5, s21, s24
- layout 8 (single): s9 ok0, s12 ok0, s25 ok0, s3, s16, s19, s28
- layout 9 (single): s4 ok0
- Stage 0 reached: 6/13 single-pair vs 2/17 two-pair.

## Per-episode findings
- s17: press 172 (bread down), hovered at toaster 720-812 while the bread rose; at ~824 lifted arm away, base turned, went to the
  fridge door (1140-2000), never reached into the slot.
- KEY: gripper width runs (packet width_every_8_steps): most episodes close the fingers early (fist to press the lever) and
  NEVER reopen: s1, s9, s10, s12(after 720), s17 closed from ~200 to the end. Reopened after pop only s4 (688-1040), s18 (1152+),
  s25 (1072-1800). Even when open, no approach to the bread.
- Stage-0-complete group (8), stage 1 behaviour (no bread contact in any):
  - stall_pre_grasp (hover at/beside the toaster): s1 (850-2000, closed), s9 (1680-2000, closed, only 320 steps left),
    s18 (940-1260 against toaster next to bread, opened 1150, then hovered right), s25 (912-1640 at lever end, then re-pressed
    the lever 1664-1792 -> toaster on again -> stage 0 undone).
  - drove off empty-handed toward the dining counter (skipped the pick): s4 (at plate 1900-1980), s10 (passes plate ~1560 then
    wanders), s12 (at dining counter 1988; also re-pressed lever at 812 -> 2nd toast cycle).
  - wander to fridge: s17.
  - Lever re-press during pop-up: s12 (812), s25 (1789). Partial first pushes: s9 (0.17), s10 (0.77), s18 (0.56/0.5).
- JSON written: s1, s4, s9, s10, s12, s17, s18, s25.
- Bread pair per scene (from wrist view at 100 + which lever moved bread): layout 2: bread in LEFT pair (pair 0) in all 7.
  layout 10: s0 slot2(p0), s2 slot4(p1), s7 slot4(p1), s13 slot2(p0), s15 slot2(p0), s18 slot4(p1), s20 slot2(p0),
  s22 slot1(p0), s26 slot3(p1), s29 slot4(p1).
- Wrong lever (other pair) fully pressed: s6 (right lever, 384), s7 (left lever, 192; bread right), s14 (right, 480);
  s8 second press (right, 1040), s27 second press (right, 672). Policy seems to press whichever lever is centred / nearest in
  the wrist view; no evidence it checks which pair holds the bread.
- Bread pulled before pressing: s8 (lifted 2.6 cm, slipped, wedged 2.8 cm up -> correct press at 464 didn't count),
  s27 (lifted 0.8 cm at 293-306, re-seated, then correct press 496).
- s27: correct pair toasted 496-994 but pair-0 lever jammed at 0.999 to the end (sim artefact?) -> no pop -> stage 0 never.
- JSON written so far: s1, s4, s6, s7, s8, s9, s10, s12, s14, s17, s18, s25, s27.
- (coordinator) helpers moved to scratchpad/GetToastedBread_helpers/ (verified intact; my files were never overwritten after
  the first summ.py collision, which only affected a script I re-wrote). gripper_touch_intervals = finger contacts only.
- Remaining-bucket verdicts (lever never >= 0.9):
  - actuation_insufficient (correct lever, shallow push, no follow-up): s2 (0.19), s5 (0.64), s19 (0.30), s20 (0.18),
    s24 (0.51), s28 (0.58); pressed bread top instead of lever: s16 (0.80), s21 (0.32).
  - went for the bread / slot top instead of lever: s0 (pull 3 mm), s3 (push+grasp), s11 (pulled 4 cm, fell back),
    s13 (pulled out and DELIVERED untoasted to plate, tipped), s26 (pushed+3 grasps), s29 (hover beside bread).
  - wrong lever partial: s15 (0.30), s23 (0.11 wrong pair, late 0.05 right pair). stall at the correct lever: s22.
- All 30 JSONs written.
