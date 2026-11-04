# WashLettuce running notes

## Task mechanics (from env code)
- stage 0: sink handle_joint in (0.40, pi) rad -> water_on.
- stage 1: env.washed_time += 1 per env step when lettuce body centre is within lettuce.horizontal_radius (xy) of the water
  site AND lettuce z < water site z + site half-height AND water on. Success at washed_time >= 25 (1.25 s).
- Lettuce starts INSIDE A COLANDER on the counter left/right of the sink. Policy (s0) grasps the colander rim and carries
  colander+lettuce under the stream. The colander is not a tracked object.

## Packet tooling notes
- `objects` is {} in every WashLettuce packet: make_packets only builds object entries for kind=="pnp"; stage 1 is kind
  "process". So no holds / lettuce position / dist-to-stream fields at all.
- close_events "touching" only checks gripper-vs-lettuce contact; grasping the colander shows as on_nothing=True.
  Successes s0, s25, s27 have NO close event (fingers only go 0.08 -> 0.06-0.07 around the colander rim, below the 0.02/10-step trigger).

## Failures (11): s2 s3 s4 s6 s8 s10 s14 s18 s19 s26 s28
- water never on: s4 s8 s10 (all layout 8 / style 8; s22 is the only L8 success)
- water on, washed_time 0: s2 s3 s6 s14 s18 s19 s26 s28
- layout 9: s2 s3 s18 s26 fail, s13 success

## Layout 8 (lever handle at right of spout base) - s4 s8 s10 fail, s22 success
- s22 turned the lever 0->0.50 in ~50 steps of contact (169-235), stream visible at 260.
- s4: fingers resting on lever 116-594, lever creeps to 0.14, 0.29 at withdrawal. actuation_insufficient.
- s10: grasped lever 240-571 (w 0.02), handle ~0 for 330 steps, 0.28 at release; base drifted. actuation_insufficient.
- s8: closed on air beside lever 166-186, then rotated closed gripper around spout 220-640, lever 0; went to colander 660, hovered to timeout. missed_grasp_continued_empty.
- Fixture value 6.28 = tiny negative angle wrapped (mod 2pi) -> packet presentation quirk.
- All 3 wander/abandon after failing; s8 skips to the lettuce with water off.
## Water-on failures
- s2 (L9, romaine Lettuce009): water 206, colander grasped 392, at sink ~520, held with stream on colander LEFT rim, lettuce in right half, 520-1100. washed_time 0. Category label: "other: held_beside_stream" (use consistently).
- L9 success s13: colander held directly under spout, stream through colander centre, 528-541.
- Wrist cam cannot see the stream when holding the colander (colander fills view) in s2.
- s3 (L9, iceberg, colander right): water 208, grasp 439, held with stream at colander LEFT rim 540-1000, released colander beside stream ~1000. held_beside_stream.
- s18 (L9, colander LEFT of sink): water 138, grasp 298, around stream 400-640 then raised & tilted next to faucet 660-1100, stream beside colander. held_beside_stream (medium).
- s26 (L9, wire colander LEFT of sink): lever touched not turned 186-520, water 550; hovered over colander 720-1100, never closed. stall_pre_grasp (root stage 1), secondary stall_at_fixture.
- L9: 4/5 fail. All three L9 held-beside cases: stream ends up at colander's rim. s13 success held colander centre under spout.
- s6 (L5, romaine, Colander008 right): water 170; closed at colander rim 406, slipped (width 0.003), empty carry to the stream 424-540; back at colander 556-1100 open-fingered, no re-grasp. missed_grasp_continued_empty.
- s14 (L4, iceberg, Colander008 right): water 190; 3 closes on colander BOWL WALL (right of loop handle) 389/409, 659 slip; wander 690-1100. missed_grasp.
- grasp_compare_wrist.png: successes mostly have the loop handle / wire rim between the fingers at closing.
- s19 (L6, romaine, Colander008 right): water 357; one close on colander bowl wall ~420-444 slips; fingers stay CLOSED 440-1100, hovering above colander looking at toaster oven. missed_grasp + stall_pre_grasp.
- s28 (L7, wire colander LEFT, long wire handle): water 267, handle grasped ~490 (slow squeeze, no close_event), colander hit spout 540-580 -> spout swivelled RIGHT ~90deg, stream displaced; held beside it 580-1100.
## KEY MECHANISM (held_beside_stream): colander/gripper bumps the SWIVEL SPOUT, spout rotates away, stream moves to the colander rim.
  Seen in s2, s3, s18 (L9 pull-down) and s28 (L7 gooseneck). Successes s13 (L9) and s0/s5/s12/s29 (L7): spout stays forward.
  Packet has NO spout joint -> invisible in packet. Category label used: "other: held_beside_stream - the colander bumped the swivel spout ..."
- s1 (success, L5): lever stall 174-656 (~480 steps, handle pushed slightly negative), water 722, success 1048 (52 steps spare). Near stage-0 timeout.
- s15 (success, L10): lever contact 138-499, pushed slightly negative until ~400; water 493; stage 1 normal, success 784.
- s11 (success, L2): water 439; colander grasped 601, slipped at ~703 (width 0.038->0.003) into basin, lettuce rolled out UNDER the stream -> success 742 (lucky).
- s23 (success, L1): water 386 (many lever touches), slow descent to colander 530-664, grasp 666, success 782.
- s16 (success, L6): lever left at exactly 0.40 (threshold, no stream) 288-414, nudged at 414 -> water 417; stage 1 fast, success 654.
- COORDINATOR: 6.28 = wrap-then-abs tooling bug, effective = min(v, 2pi-v); trust water_on. gripper_touch_intervals = finger contacts only (arm/colander contact with spout not recorded). Updated s1, s8, s10, s15, s2, s3, s18, s28 JSON notes accordingly.
- Successes s0 s5 s7 s9 s12 s13 s17 s20 s21 s22 s24 s25 s27 s29 written (light: packet + brief sheet). All: grasp colander rim/handle, carry, hold under spout outlet with spout still forward.
- Colander side: successes 7 left / 12 right-ish; failures mixed -> not side-dependent.
- Colander008 (loop handle, perforated): s6 s14 s19 = all 3 slipped colander grasps; 6 Colander008 successes. Visual compare shows no clear difference at closing (colander008_grasp_compare.png).
- Removed ran_out_of_time secondaries from s2, s8, s26 (not time-limited).
- Stage-0 stalls >100 steps also in successes: s1 (~480, success w/ 52 spare), s15 (~260), s16 (130 at 0.40 threshold); and failure s26 (~335).
