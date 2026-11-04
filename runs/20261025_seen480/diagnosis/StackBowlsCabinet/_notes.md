# StackBowlsCabinet running notes

## Task facts (from source)
- bowl1 scaled 0.7 ("smaller" per labeler), bowl2 scaled 0.95 ("larger"); model differs per episode, so real size order is model-dependent.
- Success: both bowls obj_inside_of cabinet (all 8 bbox corners within interior box +-5 cm), AND (bowl2 in bowl1 OR bowl1 in bowl2) = contact + xy-centre distance < 0.7*receptacle horizontal radius, AND gripper > 25 cm from both bowl centres. Either nesting direction accepted -> "smaller on top" NOT enforced. No upright check.
- Stage 0 (packet tooling) = either bowl inside cabinet and gripper not touching it.
- Cabinet = upper wall cabinet, bowls end at z~1.45-1.48 (counter ~0.95).

## Packet overview
- Failures: s0, s1, s3, s5, s13, s17, s20, s22, s23 (9)
- All 21 successes: grasp bowl1 (small) first ~step 60-110, nest into bowl2 on the counter, re-grasp the pair (close touching both) ~150-270, carry both into the cabinet. (s16: bowl2 grasped first - check.)
- Successes: dist_to_target_start ~0.46-0.49 m for both bowls. Failures s3, s13 (layout 9/style 9): bowls 2.4-2.8 m from the cabinet (different counter, needs base navigation). s22 bowl1 0.86 m. s20 bowl1 0.57 m.

## Per-episode findings (failures)
- s0 (L5/S5, red bowl_9 small, tan bowl_7 large): DONE. While carrying small bowl over to nest (~168), held bowl/gripper hit large bowl's rim -> large bowl flipped upside down (175-232). Small bowl released on the dome (275), slid off. 10+ futile grasps on inverted bowl. Stage 0 root cause, "knocked target bowl over". No finger contact recorded with bowl2 before 288 -> packet blind to the knock.
- s1 (L5/S5, brown bowl_8 small, striped bowl_6 large): DONE. Small bowl released leaning on the big bowl's rim (194, tipped 204-338), pair pinched by both rims (438) and lifted, slipped ~485, both fell 0.3 m to counter; small bowl at counter edge, pushed off onto the robot base (z 0.42) ~720 by re-grasps. Big bowl alone into cabinet 930 (stage 0). Wander after. Root stage 1, dropped_in_transit. Agent view hides bowls behind the arm in this layout.
- s3 (L9/S9 FAR layout, bowls on island 2.5 m from cabinet; striped bowl_6 small, brown bowl_8 large): DONE. Base drove to island (0-300), picked small bowl 341, held it at big bowl's rim 360-540 without release, then held it 850 more steps, arm wandering, base static. stall_holding_at_target + wander + far-layout.
- s13 (L9/S9 FAR island layout, like s3; brown bowl_8 small, tan bowl_7 large): DONE. Drove to island; small bowl held/released BESIDE the big bowl's rim (423, leaned on outside), re-lifted 0.53 m (reaching for a non-existent upper cabinet?) and released in the air (617, fell back), then hovered at rim 806-1360. Never nested. stall_holding_at_target + far layout.
  -> PATTERN A: far-layout (L9 island, 2.4-2.8 m): s3, s13 both fail the nest step at the island; no navigation back.
- s20 (L9/S9 near-cabinet counter; brown bowl_8 small, tan bowl_7 large, bowls ~0.49 m apart): DONE. Missed small bowl at 168 (gripper centred over bowl, not the rim; cmp_first_close_wrist.png), switched to large bowl (one-by-one), lifted 0.225, slipped/fell at ~453, 2nd miss on small bowl 609, hovered 640-1400 w/ closes on air. missed_grasp + dropped_in_transit + stall_pre_grasp.
  -> successes at first close: bowl rim across the finger gap, bowl off to the left of the wrist image. s20 bowl centred.
- s5 (L7/S7; black bowl_5 small - shallowest model, red bowl_9 large): DONE. Nest failed at 179 (small released offset, landed beside). Large bowl alone, slow 272-718 (370 steps dithering at counter height near small bowl), stage 0 at 718. Stage 1: 3 closes on small bowl rim (879, 1145, 1381) width 0.014 no lift; empty trips to cabinet after first two. root stage 1 missed_grasp + continued_empty.
- Bowl sizes (mesh extents x registry 2.0 x object_scale): bowl1 (x0.7) dia 0.16-0.18 m; bowl2 (x0.95) dia 0.22-0.25 m -> bowl1 always the smaller one. bowl_5 is shallow (h/d 0.29), bowl_9 0.28, bowl_7 0.33, bowl_6/8 0.41.
- s17 (L8/S8, both bowl_8 brown on brown wood counter): DONE. Nest failed at 160 (2nd close at 203 touched only small bowl; wrist shows single rim vs double rim in successes, cmp_pair_grasp.png). Small bowl alone to cabinet (stage 0 305). Large bowl: 900 steps of side-on closes on its outer wall near backsplash, max lift 8 mm. root stage 1 missed_grasp.
  -> PATTERN B: nest fails -> one-by-one -> second bowl (alone) cannot be grasped: s5 (small bowl second), s17 (large bowl second). Policy seems to only know rim-grasp from above in nest context.
- s22 (L8/S8 but target = green cabinet over the sink, not the wooden shelf used as target in s4/s10/s17; red bowl_9 small, black bowl_5 large; bowls 0.56-0.86 m from cabinet): DONE. Nest + pair-lift clean (391-536); stack placed on the adjacent wooden upper shelf, 0.17-0.21 m outside target interior (539). Wandered after. placed_off_target / wrong fixture.
- s23 (L1/S1, both bowl_7 tan): DONE. Nest + pair-lift ok (small bowl tilted >45 inside during carry from 380); stack released at cabinet front lip (456), stage 0 held 436-463, then fell 0.45 m inverted (464-484). Righted large bowl ~780, 5 low regrasps of small bowl, no re-nest. root stage 0 placed_then_fell. Official inside-check counted perched stack as inside.
## Successes (light pass)
- s2,s6,s7,s8,s9,s10,s11,s12,s15,s18,s19,s24,s25,s26 clean nest-then-carry. s4: pair-grasp miss 225/245 -> recovered 390 (slow stage 0 503). s14: aborted 7 cm pair lift 264-294 -> re-grasp 354 (slow 453). s16: LARGE bowl picked first (88), moved next to small, then small into large, carry (427). s21: re-close 245->266. s15: inner bowl tilted 44 deg at end, still counted nested.
- Stage 0 often fires BEFORE the final release in nest-then-carry (inner bowl untouched inside cabinet while pair still held): s8 -42, s10 -36, s12 -27, s15 -26, s16 -36, s19 -41, s26 -22. So 'stage 1 median 28 steps' = time from inner bowl entering to gripper 25 cm away.
- TOOLING BUG: make_packets.py undone reason 'an object left the target' fires when at-target COUNT changes, including an object ENTERING (s26 step 255: bowl2 entered, bowl1 stayed).
## Housekeeping
- frames/s0_close.png was a stray sheet written by another agent through the shared scratchpad fr.sh (name collision at ~14:50); deleted. s0_overview/s0_flip regenerated with my private helper (scratchpad/stackbowls_agent/) at the same steps.
- Coordinator FYI: OU.obj_inside_of tolerance is th/|edge| metres per axis (edge vectors not normalized) -> often 10+ cm, not 5 cm. Consistent with s23 (stack perched on the cabinet's front lip counted inside 436-463).
- ALL DONE: 30 episode JSONs + _summary.md written (2026-10-25).
