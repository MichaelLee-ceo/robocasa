# StirVegetables running notes

## Official stage-3 criterion (env source: robocasa/environments/kitchen/composite/sauteing_vegetables/stir_vegetables.py)
- success_time += 1 on each step where: spatula grasped (check_obj_grasped: both finger joints < 0.035 AND gripper-spatula contact)
  AND pot within 0.15 m of the task knob's burner AND veg1 in pot AND veg2 in pot AND _detect_stirring.
- in pot = veg-pot CONTACT and horizontal dist to pot centre < 0.7 * pot horizontal radius. A veg stacked on the other veg (no pot contact) is NOT in pot.
- _detect_stirring: both vegs moved > 0.5 mm horizontally since the previous grasped step (i.e. > ~1 cm/s), both in pot,
  and contact_with_spatula -- contact flag is set from veg1 only (not reset per object): spatula must touch veg1; veg2 need only move.
  -> asymmetry: stirring only veg2 never counts.
- success = success_time >= 5 (accumulated, not consecutive).
- packet stage-3 intervals come from replay (unreliable); trust online log + success.

## Episode list
Successes: s4 s5 s7 s11 s18 s21 s23 s27 s29
Failures: s0 s1 s2 s3 s6 s8 s9 s10 s12 s13 s14 s15 s16 s17 s19 s20 s22 s24 s25 s26 s28
- spatula never lifted (stage 2 never/brief): s2, s6, s20
- stage 1 never: s10
- spatula dropped after lift (stage 2 not at end): s0, s9, s12, s13, s15(st=2), s17(st=1), s22, s24
- spatula held at end, no stir registered: s1, s3, s8, s14, s16, s19(st=3), s25, s26, s28

## Per-episode findings
- s0: stage 3 (spatula jammed in pot -> arm LIFTED THE POT off the burner 748-890 -> spatula slipped 878 -> gripper stayed closed 0.001 for 700 steps, never re-grasped). Spatula held near blade end, blade down.
- s1: stage 3, spatula held 592-1600 but blade never in pot: hovered right of pot blade-up 640-1020, then pushed pot's outer wall 1040-1600 (pot tilted/displaced, flame exposed). success_time 0.
- s2: stage 2 missed_grasp on thin stick handle (spatula_9) END, off-centre, 706 w0.014; then empty-handed stir trips to pot 800-880, 1200; closes on air 1057, 1401.
- s3: stage 3, tall stockpot Pot074; spatula held 804-1600; blade only at pot's right rim 1060-1600, pot shoved left (flame exposed 1260+). success_time 0. Base moved 880-1000.
- s6: stage 2 missed_grasp on flat PADDLE end of spatula_5 (blue silicone) 742; slipped 752-760; gripper stayed closed 0.001 for 840 steps; empty stir at pot.
- s8: stage 3, spatula held 715-1600; base re-positioned, blade at pot's right rim 1100-1600, pot tilted/shoved (flame visible 1320+). Arm occludes pot 1340+. success_time 0. replay_matches false (why? online log only 0-2).
- s9: stage 2 skipped_or_wrong_stage: grabbed spatula (paddle) after only 1 veg at 356, PLACED IT INTO THE POT ~590; then veg2 721-1007 (hover 170 steps); open-gripper wander 1060-1600. Stage-2 false positive 921-1003 (eggplant-holding gripper touching spatula in pot).
- s10: stage 1 missed_grasp: broccoli at far-left counter (0.84 m from pot), arm fully stretched, closed on air 523 beside it; went to microwave (close on air 701), parked by pot closed 760-1600.
- s12: stage 3 spatula (spatula_7, held at BUTT end of handle) slipped out 926 when blade pushed into pot; before that 220 steps hovering high with blade pointing at wall (680-900); gripper closed 0.001 rest; no regrasp.
- s13: stage 3 spatula slipped 1259 while pressing hand/blade on pot's right rim (pot tilted, flame exposed); spatula_8 held at butt end; then wander 1344-1600. Stage 0 slow (miss at 100 + empty trip to pot, regrasp 397).
- s14: stage 3, pot LIFTED/TIPPED off induction burner 1320-1600 by hand/spatula (like s0); slow stage 0 (garlic dropped short of pot 494, regrasp 576; stage0 663). Objects far left counter (layout 2).
- s15: stage 3 NEAR MISS success_time 2; blade vertical in pot 720-920, spatula slipped ~920 (w 0.019->0.003), fell in pot; closed gripper rest. spatula_6 held near blade.
- s16: stage 3, spatula_5 held at the very END of handle (hole), blade hanging below occluded; hovered at pot rim 840-1600, pot shoved left (burner exposed 1080+). success_time 0. LOW confidence.
- s17: stage 3 spatula slipped ~1428 after 300 steps blade in pot (success_time 1). Stage 2 slow: hovered 560-906 over handle on far-left counter (525 vs 152). Layout 8, objects on LEFT counter.
- s19: stage 3 NEAR MISS success_time 3; spatula_5 at handle end; 560-1150 hovered with paddle UP over pot; 1180-1400 stir (3 counted); ~1420 pot dragged right off burner. Layout 6.
- s20: stage 2 missed_grasp on BUTT END of spatula_8 handle 426/457; gripper closed 0.001 for 1120 steps, empty stir at pot, passes over spatula without opening. NOTE scratchpad shared with other agents -> helpers now in scratchpad/stirveg/.
- s22: stage 3, blade bumped pot off-centre 504-540, then ~750 steps turner hanging OUTSIDE pot's right wall; handle slowly rotated/slipped (w 0.024->0.007), dropped 1301. Layout 8.
- s24: stage 3 spatula_9 (thin stick) pinched w0.010; blade jammed into tall stockpot Pot074 860-890 tilting pot; slipped ~940; closed gripper rest.
- s25: stage 3, spatula_8 held at handle END (turner hangs out of wrist view); turner outside pot right wall 740-860; hand presses pot top 960-1140 (tilt); arm occludes 1200-1600. LOW confidence. Layout 6.
- s26: stage 3, turner pointing up/away 880-1040; hand tipped/raised the pot 1060-1520 (contorted arm). Stage 0/1 stay complete (in-pot relative to pot). Stage 0 slow (slip at 120). LOW confidence on pot motion.
- s28: stage 3, spatula_9 stick pinched w0.011; blade pointing at wall 880-1080; hand drove into pot, pushed & tilted it off burner 1100-1600. Layout 7.
- s4 (succ): pot TIPPED ~800 by hand, fell back near burner; turner inside pot 840-904 -> success. Stage-2 replay mismatch = 1-step flicker at 576.
- s5 (succ): clean; paddle lowered VERTICALLY blade-down into middle of pot 600-620, stir 620-680. Stage 3 = 134 steps.
- s7 (succ): stage 1 slow 616: missed sweet potato 211, gripper CLOSED & hovering 240-585, reopened 592, regrasp 660. Stir ~100 steps blade over right rim down into pot. Replay success_time 4 vs online success.
- s11 (succ): stir slow 516 steps; turner inside right half of pot 900-1300 with small pushes, pot stayed put; counts accumulated slowly.
- s18 (succ): spatula_7 at butt end (like s12 fail) but blade down INTO pot from above 620-900; pot jostled; success 897.
- s21 (succ): spatula_9 at the stick END, blade hanging vertically down into tall stockpot 820-928; success at exactly 5.
- s23 (succ): stir VERY slow 920 steps; blade in pot from right 720+, pot shoved back/right ~860-940 and ~1260 but stayed within radius; reached 5 at 1424.
- s27 (succ): turner into pot from right 960-1000; hand tipped the pot 1020-1060; success at 1072 coincides with the pot tipping (stir partly = pot shake).
- s29 (succ): stage 1 slow 535 (missed lime 346, closed-gripper empty trip to pot 368-592). Stir 960-1040 turner in pot; pot LIFTED & tilted above element 1052-1100; success logged 1094 during the lift. s27/s29 success coincide with pot being jostled/lifted.

## Cross-episode patterns (after all 30)
- Root-cause stage: 3 -> 16, 2 -> 4 (s2, s6, s20, s9), 1 -> 1 (s10).
- Stage-3 failures, 4 sub-mechanisms: spatula slipped while blade pushed into pot (s12, s13, s15, s17, s24); stir misdirected beside/on rim (s1, s3, s8, s16, s22, s25); pot lifted/tipped off burner (s0, s14, s26, s28); late stir, near-miss (s19).
- Spatula lost after lift in 8 episodes (s0, s6, s12, s13, s15, s17, s22, s24): 0 re-grasps. Gripper stays closed (0.001) afterwards in 6/8.
- Closed-gripper persistence after a miss also in successes s7 (240-585) and s29 (368-592) - they reopened after 230-350 steps.
- Pot model: Pot071 0/7, Pot069 0/3, Pot074 1/4, Pot065 3/8, Pot075 5/8 (confounded with layout).
- Spatula hold width is set by spatula model (spatula_9 ~0.010, _6/_8 ~0.017, _5/_7 ~0.028), not by outcome.
- Successes s4, s18, s23, s27, s29 also jostled/tipped the pot; s27 and s29 success logged while the pot was tipped/lifted.
- Pre-stir hover with blade pointing sideways/up: s1 (640-1020), s12 (680-900), s19 (560-1150), s26 (880-1040), s28 (880-1080).
- Coordinator note: PickUpSpatula not latched -> failed_stage set to 3 (online log) for s0 s6 s12 s13 s15 s17 s22 s24; s6 online logged stage 2 at 752 from a 5-step lift.
