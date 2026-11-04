# LoadDishwasher running notes

## Setup facts (checked in code)
- Stage 0/1 check (scripts/atomic_steps_seen11.py `_order_step`): a dish enters the "order" list once it touches the
  TOP rack (rack1 body geoms, `Dishwasher.check_rack_contact`) AND the gripper does not touch it; after that the
  release part is latched: stage k holds while >= k+1 ordered dishes touch the rack, even if the gripper touches them
  again (s4: stage 1 holds 866-1200 while the gripper touches the bowl to the end).
- Stage 2 = env._check_success: both dishes touch rack1 and door closedness >= 0.95 (is_closed th=0.05). No release
  or upright requirement. Rack contact means ANY contact with the rack body (a dish lying on its side counts: s9, s6
  cup tilt 90).
- The scene starts with the door open and the top rack slid fully out (slide_rack value 1.0). Successful closes:
  push the rack back into the tub, go low to the door's outer edge, get under it, lift (0.1 -> 1.0 in ~30-50 steps).
- `dist_to_target` = distance to the dishwasher interior box. With the rack slid out, a dish on the rack is 0.1-0.3 m
  "away" yet at target; in successes it becomes 0.0 once the door pushes the rack in. Not a useful field here.
- Scratchpad is shared with other agents: my helpers live in scratchpad/loaddishwasher_agent/ (ldfr.sh, ld_digest.py).
  An early call with an overwritten shared fr.sh wrote diagnosis/StackBowlsCabinet/frames/s0_close.png (their s0
  video, steps 484-640) -- report it, do not touch.

## Scene groups (robot side from all_start_left.png)
- L5: s6, s9 (robot right of DW, black door) both fail at stage 2 door; s1 (robot left of DW) fails at stage 2 (wrong fixture).
- L2: s11 (stage 2 door stall), s14 (stage 1 slow cup).
- L9 (robot left of DW): s0 s13 s18 s26 succ; s2, s3, s20 fail.
- L8: s4, s8 (same side) fail; s17 F, s10 S, s22 S (other side).
- L7: s5 F, s28 F, s12 S.  L6: s16 s19 s25 all F.  L10: s15 F, s7 s29 S.  L1: s23 F, s27 S (same cup_5/bowl_7!).
- L4: s21 s24 S.

## Per-episode quick verdicts (failures)
- s9: stage 2. Both dishes on by 652 (normal). Rack pushed in 700-800. Gripper at the door tip 832-1200 (touch
  911-1200), door max 0.08. stall_at_fixture. Wrist does not show the door underside (successes do).
- s6: stage 0 slow (703): missed bowl grasp 175-210 (black bowl_5), empty trip to the rack 228-400, cup grasped
  sideways 512, held over rack ~576-700. Bowl in 989. Door: lifted to 0.26 at 1136-1152, fell back by 1168. Timeout.
- s1: stage 2 wrong fixture: from 832 the gripper goes to the drawer/cabinet handle LEFT of the DW (wrist shows bar
  handle on shaker panel), closes on air 836/880/960, stays there to 1200. Rack never pushed in, DW door touched once.
- s11: stage 2 door stall 750-1200 at door outer corner, value 0.00 throughout; rack pushed in ~720.
- s4: stage 2 never attempted: after the bowl (865) the gripper stays in the rack touching the bowl to 1200; re-close
  on bowl 946. Cup knocked over 848-864 by the bowl placement, righted by 900. Stage 0 slow (base moved 0-240).
- s3: stage 0 bowl held over the rack 212-543 (330 steps), stage 1 slowish, stage 2: looked around 880-1060, door
  lift began ~1110, 0.165 at timeout.
- s2: stage 1: cup (grasped by rim, sideways) lowered onto the tilted bowl, ended lying INSIDE the bowl (no rack
  contact). Door closed 639 anyway. placed_off_target. ("nested cup")
- s5: stage 0: bowl_7 missed at 209, 415; empty trips to rack 240-360, 780-860; approach/retreat 500-780; grasped 943,
  placed 1069 (dropped 0.21). Cup grasped at 1193. missed_grasp root, ran out of time.
- s8: stage 1: cup lowered straight into the bowl (released 630 upright in the bowl). Then poked at it 760-1200 (air
  closes 926/952/1016). placed_off_target ("nested cup" again, s2-like).
- s14: stage 0: cup held a few cm above the rack 260-917 (~650 steps), released, lay on its side. Bowl never touched.
- s15: stage 0: cup released at 299 on the rack's side edge, fell 0.45 m to the open door; bowl released at the same
  edge 582, slid onto the cup. Policy pushed the rack in and closed the door over them (993). placed_then_fell_or_slid.
## Patterns so far
- A: door-closing failures at stage 2 (s9, s11 stall at the door edge; s6 lift slipped; s1 wrong handle; s4 never left
  the bowl). Successes: wrist filled by door underside during the lift.
- B: second dish dropped INTO the first (bowl) -> no rack contact (s2, s8).
- C: long holds above the rack (s14 650 steps, s3 330, s1 300) -> time.
- D: release at the rack edge -> falls off (s15; s23 known).
- s16: bowl_9 (flat) rim pinch 499 without grip, dragged off the counter edge 512-536, fell to the floor (0.88 m).
  Policy closed the door at 814 anyway. Stage 1 'complete' 1 step (533) as the falling bowl brushed the rack.
- s17: bowl placed ~257-310, then parked with open fingers on the bowl rim to 1200 (one re-close 767). Cup never touched.
- s19: cup placed 251, then wandered 290-900 (upper cabinets/toaster), bowl attempts 1064 (air), 1167 (slip).
- s20: cup placed 373, parked at the cup 373-980; bowl gripped 1069, held over rack at timeout. root 0 (stage 0 at 645).
- s23: bowl carried tilted, released 903 at the rack FRONT edge resting on the open door (dist 0.335); counted via
  contact with the front wires; rack pushed in without it; rising door slid it off (1169). Door 0.75 at 1200.
- s25: bowl rim grasp slipped 536, pinch 670, empty trip to rack 920-1040 (air close 935), bowl gripped 1131, in transit.
- s28: cup lowered to the rack's front rail, released 595, fell 0.36 m onto the door; door closed over it 740.
## Successes (all 12 written). Odd ones: s26 stage 2 570 steps (door-top stall 520-900, recovered); s22 bowl held
  over rack ~250 steps; s10 120-step door-top detour. s23 vs s27 (same kitchen/objects): s27 put the bowl first,
  deep in the rack.
## Helper hygiene
- Coordinator warned about shared-scratchpad name collisions (2026-10-25 ~15:15). My helpers have lived in
  scratchpad/loaddishwasher_agent/ since 14:51 and are intact. s9_overview.png (the only sheet made with the old
  shared fr.sh) was regenerated with ldfr.sh and matches the JSON. diagnosis/StackBowlsCabinet/frames/s0_close.png
  (which my mis-routed call may have overwritten) no longer exists.
