# SetUpCuttingStation running notes

## Episode list
Failures (16): stage-0 incomplete: s7 s8 s12 s13 s15 s16 s17 s20 s21 s22 s27 s29; stage-1 incomplete: s0 s11 s18 s23
Successes (14): s1 s2 s3 s4 s5 s6 s9 s10 s14 s19 s24 s25 s26 s28

## Tooling notes
- scratchpad is shared with other agents: own helpers in scratchpad/sucs/ (summ.py, fr.sh). frames.py: options must come AFTER the step list.
- fixtures {} in this task (drawer pre-opened).

## Per-episode
- s0 F: stage1 missed grasp on tilted steak_6 edge (596, lift 0.5cm), empty place on board 648-720, wander 800-1440 closed gripper, late retry 1563 no lift.

- s11 F: stage1 wander_away; meat 2.6 m from board behind oven column; went back to knife drawer 680-1000 (close air 928), glimpsed plate ~1200, turned into corner. Navigation scene (meat far).
- s18 F: stage0 slow approach-retreat over drawer 100-795 (open-finger push 669-681), knife on board 962; stage1 wander to island 1160-1600, meat never touched though only 0.46 m from board and 638 steps left.
- s23 F: stage1 placed_off_target; fish grasped by tail, released 791 over board front-left corner 0.196 m from centre, fell half off; then close on air over empty plate 1451. Knife: slipped 298, regrasp 371 ok.
- s7 F: stage0 dropped_in_transit x3 (butt-end handle grasp slips 204, 716; upright-blade grasp slips 1150), knife ends standing on edge in drawer; 4th grasp 1419 carried vertical, released at board edge 1501 -> toppled against drawer. Board only 0.15 m from knife.
- s8 F: stage0 stall_pre_grasp 340-1260 (handle off-centre near right pad), grasp 1413 near bolster, slipped ~1440, empty 'place' over counter 1568. Board 1.08 m away (nav).
- s12 F: stage0 three air closes short of knife 129-275 (wood handle on wood drawer), butt-end grasp 514 pivot+slip ~558, then wandered 620-1600 never retried. Board 1.29 m away.
- s13 F: stage0 pure stall_pre_grasp, ZERO closes in 1600 steps; knife at back of drawer partly under counter overhang (layout 9). Knife moved 4.4 cm with no recorded touch.
- s15 F (layout10 like s7): air closes 286/646/892 + hover 400-990; handle grasp 1002 lifts 0.10 then slips ~1035; EMPTY place on board 1248 (gripper shut 1040-1240); 2nd grasp 1363 slips ~1395.
- s16 F: stage0 skipped: black knife in black drawer (layout5 style5 black cabinets), gripper went to wrong side 200-280, went to steak ~320; steak released at board near edge 954, fell into drawer. Knife moved 16cm w/o touch (steak hit it).
- s17 F (layout8): clean centred handle grasp 260, knife slipped out during the swing/wrist rotation toward board ~361-376; empty place on board 448; knife ended flipped/propped on its edge; 3 failed regrasps 616-904; then placed meat on board 1255 (stage-1 obj done, knife not).
- s20 F (layout9): open-finger push on handle 163-177 tipped knife on edge; 5 failed closes 311-495; abandoned knife, navigated to meat; 10 failed grasps on thin flat steak_8 on wooden plate_8 882-1535.
- s21 F (layout4): off-centre close on handle at right pad 381, knife rolled+slid out; hover 440-900; abandoned knife -> 4 failed grasps on flat steak_8 1062-1583.
- s22 F (layout8, green knife_9): 5 closes on air (134,376,896,1232...), never touched knife; empty carry 896-1008; hover rest. Same drawer as s4/s8/s10/s17.
- s27 F (layout9, green knife_9): knife at back of drawer under counter overhang, only handle end exposed; hovered ~200-1200 (min 0.22 m), air closes 1059, 1146; switched to fish 1545. SAME mechanism as s13.
- s29 F (layout10 diff spawn): clean knife grasp 632, released 826 between plate and narrow board_5 (0.165 m) -> knife across plate rim + counter; then drumstick fails 1230/1309, wander.
## Successes
- s10 S (layout8): rolled knife onto edge 237-319, 3 failed closes, 4th at 618 lifted; knife placed on its edge on board (tilt 69, criterion OK). Stage0 773 (slow).
- s3 S (layout9): stage0 slow 1052 due to 2.4 m navigation while holding knife (431-1050); minor misses 323/410.
- s19 S: stage1 slow (801 steps): 3 failed fish closes 691-1199 then tail grasp 1303. Stage0 14 one-step contact flickers 'object left target' with knife static (also s4 s9 s14 s23 s26) -> criterion/tooling artefact.
- s1 S: stage1 slow 496: fish miss 558, regrasp 856, placed partly on knife 942 (knife stayed).
- s2 S (layout9 green knife at back like s27 but reached it 397). s14 S (navigation both stages, clean).
- s4 S (layout8 same drawer as s8/s17/s22): clean centred grasp 342. s5 S clean.
- s6 S, s9 S clean (s9 has 4 one-step stage0 flickers).
- s24 S clean (layout4 like s18/s21). s25 S: steak_8 (same as s20/s21 failures) grasped by closing AROUND its narrow bone end (width 0.06) - close_events MISSES this close (tooling: wide closes not detected).
- s26 S fastest (515). s28 S detour ~600 before fish.

## Cross-episode patterns / tooling (running)
- close_events on_nothing MISLABELS successful grasps where contact begins 15-20 steps into the close: s5 225 (touch 241), s3 410 (touch 431). Also misses wide closes (s25 steak 0.08->0.06).
- 1-step 'object left the target' stage-0 flickers with static knife: s4 s9 s14 s19(14x) s23 s26.
- knife tilt 24.5 at rest for knife_9 (s18, s22, s27); meat tipped from step 0 for fish_9/drumstick models (s1, s8, s11) - tilt axis meaningless for these models.
- dist_to_target for knife is horizontal: knife in drawer below board reads 0.08-0.30 m.
- No case of meat placement pushing knife off the board (stage 0 complete_at_end true in all 4 stage-1 failures; meat often placed touching/on knife in successes s1 s2 s4 s28).
- Layout 10: 0/3 (s7 s15 butt-end slips; s29 off-board release). Layout 8 drawer (3.85,-2.6): 2/5. Layout 9 drawer (0.5-0.6,-3.45): 2/6 success, knife at back under counter overhang (s13, s27 stall).
- Slipped knife grasps: handle diagonal across finger gap and/or held at butt end (s7, s8, s12, s15); s17 centred but slipped during swing+wrist rotation. Successes: handle straight through finger gap, centred. Slips happen 20-45 steps after grasp at lift 0.09-0.15 m (s17 0.24).
- steak_8 (flat T-bone): s20 10 closes, s21 4 closes, max lift 9 mm (pinching the thin edge, closes to 0.004-0.015); s25 succeeded by closing around the narrow bone end (width 0.06).
