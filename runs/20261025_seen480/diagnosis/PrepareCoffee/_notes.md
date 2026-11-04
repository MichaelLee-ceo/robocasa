# PrepareCoffee running notes (diagnosis agent)

## Official checks (read from code)
- stage 0 = check_receptacle_placement_for_pouring (xy < 0.04 m from receptacle_place_site, |dz| < 0.10) AND gripper > 0.25 m from mug centre.
- stage 1 = env._check_success(): stage-0 placement AND gripper far from mug (0.25) AND coffee_machine._turned_on AND gripper > 0.15 m from every start_button geom.
- _turned_on latches on the FIRST gripper contact with any start_button geom, regardless of whether the mug is there (s4, s26 turned it on with the mug off target).
- So "machine on, mug placed, but no success" = the gripper never got 15 cm from the button / 25 cm from the mug at the same time (s0, s2, s17).

## Packet triage (before video)
- successes: s1, s8, s16, s22, s27, s28. release dist 0.012, 0.014, 0.021, 0.036, 0.003, 0.010.
- never lifted mug from cabinet: s12, s15, s20, s25 (no touch), s19 (touch, no lift)
- mug placed OK (<4 cm), machine never on: s3, s5 (no machine touch; mug tipped!), s10, s11, s13, s18
- mug placed OK, machine on, no retreat: s0, s2, s17
- mug off target near machine: s4 0.051, s6 0.081, s7 0.043, s9 0.046, s14 0.072, s29 0.047, s23 0.100
- far off / dropped: s21 0.24 (drop 0.40), s24 0.285 (drop 0.19), s26 0.178 (drop 0.12)

## Tooling note
- scratchpad is shared between the 15 agents: helper scripts named fr.sh/cmp.sh/summ.py got clobbered. Use scratchpad/preparecoffee_agent/pcfr.sh only.
- frames/s4_overview.png was written by another agent's call through my old fr.sh (unknown steps) -> overwrite with my own.

## Per-episode findings
- s0: mug 0.0205 at 371 (good). Lower front button pressed 443 (on). Then gripper on TOP of the brown machine 664-1200 rubbing the lid, never retreats. Stage 0 only 663-688. root stage 1, no retreat.
- s2: identical to s0 (same layout 9 brown machine). mug 0.034 at 338; button 440; then on the lid 480-1200.
- s17: layout 8 black capsule machine. slow placement (hold 156-585) but 0.011. press 793, then parks at button face to 1200. no retreat. Packet failed_stage 0 is artifact.
- Successes post-press: s1 543->563, s27 478->501, s16 479->554, s8 646->820 (touch to 795), s22 680->916 (touch to 886), s28 634->816 (touch to 794). So lingering 20-240 steps after the press is normal; s0/s2/s17 never leave (400-750 steps).
- Machine models (style == layout here; test styles): L1 CM067, L2 CM045 (1 button), L3 CM041, L4 CM037, L5 CM042, L6 CM047 (1 button), L7 CM039, L8 CM102 (capsule, SINGLE/DOUBLE on round face), L9 CM036 (brown, 2 stacked front buttons), L10 CM051 (1 button, left side top).
- Layout tally: L1 s6 | L2 s11 s14 s26 | L3 s27* | L4 s21 s24 | L5 s1* s4 s9 | L6 s16* s19 | L7 s5 s12 s28* | L8 s3 s8* s10 s17 s22* s25 | L9 s0 s2 s13 s18 s20 | L10 s7 s15 s23 s29  (*=success)
- s3: L8. slow placement (hold 114-579, 0.018). One press 604-648, closed fingers at SINGLE button, machine contact only, no button. Then hovered. missed button.
- s10: L8. placement 348 0.018. press 452-545 with OPEN fingers; SINGLE button sits in the finger gap (wrist zoom). closed poke 566. then hover 700-1200. missed button.
- L8 successes s8/s22 and s17 pressed from the RIGHT side of the round face (wrist sees the face edge-on); s3/s10 came head-on.
- s11: L2 espresso machine CM045 (single cup-icon button on upper housing, portafilter handle below). mug 0.0364 at 340 (2-3 mm inside). 383-866 open gripper pushing the PORTAFILTER HANDLE, button in view above. wrong_fixture_element. then shelf.
- s13: L9 brown. mug 0.031 at 396 then re-grasp and nudged to 0.012 by 557 (self-correction). Press 612-648 head-on, fingertips just under the lower button, body contact only. Then reached up to the pear in the cabinet (close 1015 on nothing). missed button + wrong_object.
- L9 pattern: s0/s2 pressed (from right side, fingers at lower button) then stuck on the lid; s13 head-on press too low.
- s18: L9 brown. mug 0.012 at 345 (excellent). Press 395-560 head-on with open fingers, DOUBLE button centred in the finger gap; body contact only. Then cabinet/jug (close 976 on nothing).
- MECHANISM "button in the finger gap": s10 (L8), s13 & s18 (L9) pressed head-on with open (0.08) fingers centred on a button; s0/s2 (L9) registered when one fingertip was offset onto the button (cmp_layout9_button.png).
- s5: L7 CM039 narrow slot. mug pushed into the slot 252-280, TIPPED in gripper (278), released on its side 306 at 0.0255. Official stage 0 complete (no upright test). No machine contact ever; hovered 500-1200. root stage 0 physically (official: stage 1).
- TIPPING theme: tipped intervals s4 487-, s5 278-, s6 571-, s9 283-, s21 385-, s23 305- (several while held).
- s4: L5 CM042. held mug at front-right of tray, released 347 at 0.0513 (near miss, nozzles to the left/behind the mug in wrist view). pressed button 447 with mug off; hand knocked mug over 464-492 (tipped 487), end 0.115.
- s6: L1 CM067. at target 426-558 while HELD (130 steps), pushed past, mug tipped in grip 571, released 625 at 0.081, fell on side. release_timing. Then button attempts 752-807, 920-946 missed (button at fingertips).
- s7: L10 CM051 (machine at right edge of agent cam). mug under group head 300-352, released 364 at 0.043 (3 mm out; visually under the spout, wide mug). fingers dragged rim 368-373 -> 0.069. stage 1: fingers at the LOWER cup-icon control 458-656 (start_button geom is upper-left, z=+0.157) - probably wrong control. root stage 0 placed_off_target near miss.
- s9: L5 CM042. rim grasp; mug pitched forward in grip while lowered (240-283, tipped 283). Held tipped mug at tray 283-939 (650 steps!). released 939 at 0.046, lies on side at 0.042 (2 mm outside: would have counted toppled!). 
- s14: L2 CM045. mug held LEFT of group head 340-388, released 393 at 0.072 -> 0.088 (beside the spout). no correction, wandered; no machine contact.
- s29: L10 CM051. tall mug_4 held tilted against the low portafilter 300-480; released 495 at 0.0466; settled upright under group head slightly forward at 0.0429 (2.9 mm out) -> criterion near miss. No machine contact ever; hovered away 800-1200.
- s23: L10 CM051 (machine invisible in agent cam). rim grasp; mug pitched over in grip ~304 at the tray; held tipped 316-535; released 535 at 0.100, fell (drop 0.07), end 0.126 on its side. Stage 1: fingers around the central DIAL 584-1036, never the upper-left button.
- L10 pattern (0/4): s7 & s23 work the lower cup-icon / central dial, never the upper-left start button; s29 never tries; s15 never grasps.
- s21: L4 CM037 (machine at right edge beside open cabinet door). mug tipped in grip 340-385 near the tray; carried tipped 432-808 incl. base turn ~480; released 808 ~40 cm above counter at 0.24, fell (drop 0.40). No machine contact.
- s24: L4. rim pinch 0.008; mug SLIPPED out at 184-190 (fingers closed to 0.002, never opened), fell 19 cm to counter 0.28 m off. empty gripper to machine top 340-416, wander. dropped_in_transit.
- GRIP-WIDTH FINDING: successes all open deliberately (0.04-0.08 within 16 steps of release). 'Releases' where fingers instead closed to 0.002 (= slip off the rim): s3 (579), s5 (306), s6 (625), s13 hold1 (396), s19, s21 (808), s24 (190), s26 (303). Rim pinch (hold width 0.006-0.021) lets the mug rotate (tipping) or slide out.
- MUG MODELS: mug_4 0/4 (s4,s11,s13,s29); mug_5 2/6; mug_7 3/5; mug_8 0/8 (s7,s15,s18,s20,s21,s23,s25,s26; 3 never grasped); mug_9 1/7 (tipping s5,s6,s9).
- s26: L2. tea mug_8 pinch 0.006; slipped out beside the group head ~303 (12 cm drop), counter 0.18 off. Empty gripper pressed the start button 565 (works on CM045 when centred on it). wander.
- s12: L7 red mug_5 in dark cabinet. first close 104-116 just in front of the wall (no contact); then hovered 10-20 cm away closing on air x5 (479-830). missed_grasp + stall_pre_grasp.
- s15: L10 tea mug_8 on open shelf. premature close at 88 a few cm short of mug; gripper NEVER REOPENS (0.001 to 1200); approach/retreat with closed fingers. missed_grasp.
- PATTERN 'closed for good': s15 (close 88), s20 (78), s25 (127), s19 (132, touched mug, no lift) - gripper stays at 0.001 for the rest of the episode. 3 of 4 are mug_8.
- s19: L6 blue mug_7 handle toward robot. closed at mid-body/handle (contact 146-159, no lift), fingers slid off; never reopened; nudge 332-348; hover. missed_grasp.
- s20: L9 tea mug_8 on navy shelf, premature close 78, never reopened. s25: L8 tea mug_8 in cabinet, close 127 at the wall, no contact, never reopened.
- s20/s25/s19 done (closed-for-good pattern). Successes done: s1 s8 s16 s22 s27 s28 (s22 recovered a head-on miss by pressing from the right side at 680; s8/s22/s28 linger 170-240 steps after the press).
- FINAL tally: root stage 0 = 16 (s4 s5 s6 s7 s9 s12 s14 s15 s19 s20 s21 s23 s24 s25 s26 s29), stage 1 = 8 (s0 s2 s3 s10 s11 s13 s17 s18).
- Successes: hold grasp->release 174-281 steps; release 38-68 steps after first at-target; release dist 0.003-0.036; press 119-368 steps after release.
