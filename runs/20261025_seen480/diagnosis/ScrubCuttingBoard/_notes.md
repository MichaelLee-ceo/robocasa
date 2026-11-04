# ScrubCuttingBoard running notes

## Criterion facts (from env source robocasa/environments/kitchen/composite/sanitizing_cutting_board/scrub_cutting_board.py)
- update_state every control step: if sponge-board geom contact AND OU.check_obj_grasped(sponge) (both finger joints < 0.035,
  i.e. each finger < 3.5 cm; sum width ~0.058 for sponge_4/6/8, ~0.065 for sponge_5) AND sponge xy >= 2 cm from every
  recorded point -> new contact point. contacts = number of points; sweep = diagonal of the bbox of the points.
- _check_success: contacts >= 5 and sweep >= 0.10 and |eef - sponge centre| > 0.15 m. No explicit release check and no
  check of where the sponge ends up (it can be dropped anywhere / held loosely?? -> eef must be > 15 cm from sponge centre so
  in practice released).
- Stage 0 check (_held_lifted): gripper touching sponge and sponge NOT touching the counter. So stage 0 stays "complete"
  while the sponge lies on the board and the gripper touches it.
- Per-step trace (read-only): analysis/ScrubCuttingBoard/<tag>_trace.json.gz -> sponge pos/touch/grasp/eef_dist, contacts,
  sweep per step. grasp = check_obj_grasped (same predicate the env counter uses). replay matches online log in all 30.

## Episode groups (from packets + trace)
- A: scrub criterion met (>=5, >=0.1) but sponge still held at 800: s2(met 289), s7(330), s9(472), s13(720), s14(385),
  s24(327), s29(559; stage-0 regrasp at 341 after miss at ~87-104).
- B: released early with too few counted contacts: s3 (4 / 0.099 at 463), s8 (4/0.258 at 456), s10 (2/0.182 at 379),
  s12 (1/0 at 312), s17 (3/0.144 at 273), s18 (2/0.108 at ~306), s19 (1/0 at 378), s28 (2/0.081; width 0.056->0.04->0.001
  around 224-256: slip/drop?).
- C: never really grasped: s11 (closed to 0.005 at 170, no lift), s25 (close 113 width 0.019, lift 2 cm, lost at 130).

## Successes: release (last touch) steps 332-784; scrub duration first contact->release 153-601 steps.
  sponge_8: 5/6 success; sponge_5: 1/5; sponge_6: 3/9; sponge_4: 4/10.

## Per-episode findings
- s2: sponge next to board; scrubbed 165-776 (18 contacts, criterion met 289); at 776-800 was lifting the sponge off the
  board to the right (back towards its start) when time ran out. Never released.
- s3: contacts 1-4 at 172-276 then sponge moved widely (34 cm, 276-412) at z +29..35 mm vs rest-on-board 25 mm -> 4-9 mm
  ABOVE the board -> no new contacts ("air scrub"); released on board at 463, retreated. 4 contacts / sweep 0.099 (1 short).

## KEY criterion finding: the scrub counter's grasp test only checks finger 1
- Saved states run/ScrubCuttingBoard/<tag>_steps.npz: state[:,12]=finger_joint1 (0..+0.04), state[:,13]=finger_joint2 (-0.04..0)
  (|q1|+|q2| matches trace grip_width to 5e-5).
- OU.check_obj_grasped: all(qpos < 0.035) -> q2 is always negative so its test is vacuous; grasp == touch & q1 < 0.035
  (reproduces trace grasp flag on 100% of steps in s0, s8, s12, s19).
- When the sponge is pressed on the board the finger pair often slides sideways in the hand: q1 -> +0.040 (open limit),
  q2 -> -0.016. Sum width unchanged (0.058), sponge still firmly held, but grasp=False -> board contacts NOT counted.
  Shift the other way (q1 0.017, q2 -0.040) counts fine. Windows typically start right after the FIRST counted contact
  (s0 181-239, s12 152-238, s19 167-243, s17 143-199, s18 145-252, s8 202-345, s10 250-307, s28 166-240, s23 186-332...).
  Sponge_5 (width 0.065, q1 ~0.032 at rest) is only 3 mm from the threshold.
- A z-height proxy for board contact was tried and is NOT calibrated (overcounts) -> not used as evidence.

## After an early release (group B) the policy never re-grasps: arm goes up / base turns / looks around until 800
  (frames/groupB_late.png: s3 s8 s10 s12 s17 s18 s19 s28 at 600 and 796).
- s8: wide sponge_5; real scrub 196-300 counted 1 (blind 202-345); drifted off board right; re-pressed 450; released 456 -> 4/5.
- s10: long carry (board 50 cm), one slow pass 232-376, blind 250-307 while pressed & sliding 9 cm; released 379 -> 2.
- s12: strongest artifact case: 36 cm stroke on board 156-236 entirely in blind window (q1=0.040); released 312 -> 1.
- s17: pressed in one spot ~90 steps (148-228), slid off edge, returned sponge, released 273 -> 3 (scrub genuinely short).
- s18: sponge_5 tilted 20 deg; 100-step scrub, overshot off board's far edge 232-256; released 300 -> 2.
- s19: one slow 11 cm push 168-264 in blind window; carried back, released 378 on counter -> 1.
- s28: sponge slipped out of fingers while dragged toward robot 200-244 (width 0.056->0.040->0.001), not a commanded open;
  retreated with closed empty gripper -> 2.
- s11: wide sponge_5 rotated ~25 deg; squeezed out of the fingers at 180-184 (lift 1.4 cm), fingers closed on air, never reopened;
  empty closed-gripper 'scrub' over the board 280-800. missed_grasp + continued_empty.
- s25: sponge_4 spawned flush against the board's edge; left finger at the board edge -> pinched the top edge, sponge flipped
  out at 128-132; empty 'scrub' 168-320; wandered. missed_grasp + continued_empty.
- Group A detail: s2 (criterion 289, scrubbed small +/-5cm then tilted-edge wipes; began return lift at 776), s7 (crit 330,
  12 pts by 405 then 400 steps on small board w/o new pts), s9 (slow grasp 180; crit 472; return-like lift 628-692 to +29 cm
  then came BACK to scrub 708-800), s13 (sponge_5; mostly waved 4-8 cm above board, crit only 720 -> actuation_insufficient),
  s14 (slow grasp 162; crit 385, no new pts after 561), s24 (crit 327; wiped along board's far edge then middle),
  s29 (missed grasp 100-104 slipped, empty carry + hover, re-opened & re-grasped 341 -> stage 1 normal, ran out of time).

## STATUS: all 30 episode JSONs written; _summary.md written (2026-10-25).
Final counts: root stage 0: s11 s25 s29; stage 1: 14. Primary: actuation_insufficient 7 (s3 s8 s10 s13 s17 s18 s19),
stall_holding_at_target 5 (s2 s7 s9 s14 s24), missed_grasp 3 (s11 s25 s29), official_criterion_issue 1 (s12), dropped 1 (s28).
Packet/tool issues: close_events misses 14/28 grasps; objects{} empty; first_incomplete_stage=0 in every success (pick
condition not latched).
