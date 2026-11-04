# Packet / trace tooling issues reported by the diagnosis agents (fix after all agents finish)

- Sink handle value wraps: `stage_trace.py` fixture value = abs(handle_joint), but handle_joint is wrapped mod 2*pi, so a
  handle a hair below 0 reads ~6.2832. Use min(a, 2*pi - a). Seen in RinseSinkBasin s2 s11 s13 s20, WashLettuce s1 s8 s10
  s15 s18. (RinseSinkBasin agent)
- Touch intervals are gripper-finger contacts only; arm-link contacts that move a fixture (spout) are not recorded.
  RinseSinkBasin s15 s22 s28. (RinseSinkBasin agent)
- No spout-angle timeline for the sink: RinseSinkBasin's progress records washed_loc / spout_ori only at the end in the
  packet; the agent rebuilt it from `_steps.npz` (matches washed_loc 30/30). WashLettuce: the spout swivel being pushed
  ~90 deg by the colander is invisible in the packet. Add spout joint to the sink state.
- `objects` is {} for tasks whose stages are fixture/process only (make_packets builds object entries only for pnp
  stages): WashLettuce has no lettuce trace. Track process-stage objects too.
- Non-task containers are not tracked (WashLettuce colander: the policy carries the whole colander), so colander grasps
  show as `on_nothing` / "touching lettuce"; slow squeezes on thin wire handles log no close event (width drop < 0.02 in
  10 steps): WashLettuce s0 s7 s25 s27 s28. (WashLettuce agent)
- `close_events.on_nothing` false positives: touching is checked only in [close, close+15], but contact can start 15-20
  steps into a slow close, so the real grasp reads on_nothing=true (SetUpCuttingStation s5 225, s3 410, s25 252). Use the
  window until the width settles (or +30).
- Wide / slow closes missed: a 0.080 -> 0.060 close (steak, s25) has no close event although the hold is recorded; the
  ">= 0.02 drop within 10 steps" rule misses it (also WashLettuce wire handles). (SetUpCuttingStation agent)
- Tilt is relative to the world z axis, but some models' rest pose is not "upright": knife_9 reads 24.5 deg lying flat,
  fish_9 and a drumstick are "tipped" from step 0. Measure tilt relative to the start orientation.
- Object displacement without a finger touch is not reported (knife moved 4.4 cm in s13, 16 cm in s16 when the steak hit
  it). Add "moved > 2 cm while not touched" events.
- dist_to_target for receptacle targets is horizontal only, so a knife in the drawer below the board reads 0.08-0.3 m.
- One-step flickers of a pnp condition give "an object left the target" undo events while the object lies still
  (SetUpCuttingStation s4 s9 s14 s19 (14x) s23 s26). Debounce (ignore drops < ~3 steps) or label them as flicker.
- PreSoakPan: `_task.json` stage definitions and `objects[o].target_of_stage` say stage 0 = pan, 1 = sponge, but our
  checks are order-free (stage 0 = first object in, 1 = both in). Fix the labels for all order-free multi-object stages.
- Drawer packet value is the official normalized value, not a physical fraction (see below); label it as such.

# Official-criterion facts verified by me (keep the official behaviour; report, don't change)
- Drawer `get_door_state` (robocasa/models/fixtures/cabinets.py:1061) normalizes the slide joint by 0.55 * depth while the
  joint range is the full depth (cabinets.py:960, range [-2y, 0]), so fully open = 1/0.55 = 1.82 (observed max 1.82 in
  DeliverStraw). OpenDrawer's ">= 0.95" therefore means >= 52 % of the physical travel; DeliverStraw s0's 0.853 is ~47 %.
- DeliverStraw composite `_check_success` (composite/serving_beverages/deliver_straw.py:76) does not check the drawer and
  uses check_obj_in_receptacle(straw, glass_cup, th=0.5): straw-cup centre distance 0.5 m (~10x the cup radius). s15 held
  it 795-836 with the straw lying across the rim; only gripper_obj_far blocked a false success. (DeliverStraw agent)
- `obj_inside_of` (robocasa/utils/object_utils.py:14) compares dot(u, p) against dot(u, p0) - th with u = the fixture's
  un-normalized edge vector, so the real tolerance along each axis is th / |edge| metres: 0.05 / 0.5 m = 10 cm across a
  sink, 25-50 cm vertically for a 10-20 cm deep region. Explains the PreSoakPan s16/s19 sponge-on-the-counter "inside".
  Applies to every sink / cabinet / microwave / fridge / dishwasher "inside" check. (PreSoakPan agent; code confirmed)
- ScrubCuttingBoard's scrub counter (composite/sanitizing_cutting_board/scrub_cutting_board.py:82,102) counts a board
  contact only if OU.check_obj_grasped (object_utils.py:665): all(finger joint < 0.035). finger_joint2 is always <= 0, so in
  practice only finger_joint1 < 0.035 is tested (open = 0.04): a wide sponge, or fingers shifted sideways while pressing,
  reads "not grasped" and its contacts are not counted. Likely decided ScrubCuttingBoard s12 (36 cm stroke, 1 contact
  counted), contributed in s8 s10 s18 s19; sponge_5 sits 3 mm from the threshold (1/5 success). (ScrubCuttingBoard agent;
  code confirmed)

# More tooling issues
- Pick stages (PickUpSponge, PickUpSpatula) are "held and lifted" now, not latched, so `first_incomplete_stage` = 0 in
  every success once the object is released (ScrubCuttingBoard). Latch pick stages after their first completion, as the
  online tracker does.
- `objects` is {} in ScrubCuttingBoard too (process stage): no sponge holds / release. close_events lists no close in 14
  of 28 episodes where the sponge was grasped (slow-close detection, see above).
- StoreLeftoversInBowl: `fridge.check_rack_contact` (models/fixtures/fridge.py:335) is not a contact test: it keeps the
  shelf regions whose z band (bottom .. 90 % of the region height) contains the object's centre and then runs
  obj_inside_of against those region boxes. The agent reports that Fridge 066's shelf regions leave out the front ~10-17 cm
  of the physical shelves (s25 s26 s29 fail with the bowl standing upright on a shelf) and Fridge 083's top shelf is
  dropped by the height filter (s21) - inferred from saved positions, region geometry not re-measured by me.
- `check_obj_in_receptacle` (object_utils.py:587) = contact AND xy centre distance < 0.7 * receptacle radius (or th):
  food on the floor touching a fallen bowl counts as "in the bowl" (StoreLeftoversInBowl s0 s12 s19 s20 s22 s27). The
  official `check_obj_upright` exists (object_utils.py:603) but no composite here uses it: a mug on its side counts as
  placed (PrepareCoffee s5).
- PrepareCoffee: the button press counts with no mug present (s4, s26), and the gripper-far clauses delay stage 0's
  recorded completion by up to 560 steps. (PrepareCoffee agent)
- `holds` keep running while an object rests on the open hand, so `release_step` marks the fall (StoreLeftoversInBowl s4
  s8 s20); `release_step` also fires on slips. Fixture "which part was touched" (lever vs spout head, which button) is not
  recorded. (StoreLeftoversInBowl, PrepareCoffee, PreSoakPan agents)
- Shared scratchpad: agents' helper scripts with the same names overwrote each other. Next time give each agent its own
  scratch dir.
- StirVegetables stir counter (composite/sauteing_vegetables/stir_vegetables.py:114-159, code confirmed): success_time
  += 1 on every step where the spatula is "grasped" (check_obj_grasped, the one-finger test above), the pot is at the task
  burner (0.15 m), both vegetables are "in the pot" (contact + xy) and _detect_stirring holds; success = 5 such steps in
  total (0.25 s). In _detect_stirring `contact_with_spatula` is sticky across the loop, so only veg1 has to touch the
  spatula; each vegetable only has to move > 0.05 cm = 0.5 mm in xy since the previous step. Shaking or tipping the pot
  satisfies it (s27, s29 scored successful while the pot was tipped / lifted). The spatula pickup check fires when closed
  fingers touch a spatula resting in the pot (s9, s0). (StirVegetables agent)
- No pot pose, spatula-vegetable contact or per-step stir counter in the StirVegetables packet; pot displacement (13 of
  16 stage-3 failures) was read from video only.
- LoadDishwasher (agent-reported, not code-checked by me): any rack contact counts as "on the rack" (s23's bowl lying on
  the door, touching the rack, counted 904-1168); dishes on their side count (13 episodes); a cup nested in the bowl does
  not count; the release clause latches after the first release (s4). Packet: dist_to_target is measured to the tub
  interior while the rack is slid out; dishwasher touch intervals include rack contacts; the stage-2 fixture's `done`
  means only "door closed" (true in s2 s15 s16 s28 with loading incomplete) - label it.
- BUG make_packets.py:89-91 undo reason: "an object left the target" is chosen whenever the at-target count changes,
  including when an object enters at the same step (StackBowlsCabinet s26 step 255). Fix: left = any object at target at
  e_ that is not at e_+1. (StackBowlsCabinet agent; code confirmed)
- Contacts by a held object or the arm are not recorded (StackBowlsCabinet s0 knock-over at 175, s3 18 cm push).
- StackBowlsCabinet (agent-reported): "smaller on top" is not enforced (either bowl inside the other counts), no upright
  test, a stack perched on the cabinet lip counts as inside (s23; consistent with the obj_inside_of tolerance).
- SteamInMicrowave (agent-reported): START latches with the door up to 90 % open and stays on after closing (s25); the
  vegetable-in-bowl check counts a bowl on its edge (s16) or bowl + corn on the floor (s10); s22 possible false negative
  of bowl-inside. Packet lacks a start-button contact flag / gripper-to-button distance, so wrong-key calls rest on the
  wrist video only.
- GetToastedBread (agent-reported): the toaster fixture value is the max over both slot pairs and `done_intervals` means
  "any pair on", so toasting the empty pair (s6 s7 s14) shows as done; no field for the bread's slot pair / height (agent
  took them from `_steps.npz`); s27 lever stuck at 0.999 after its cycle (likely sim artefact) so stage 0 could never
  complete. `holds` counts fist pushes as grasps (s16 s21). PackIdenticalLunches: no field for which tupperware holds an
  item (tupperwares not tracked); `lift_m` is relative to the shelf, real carries read 0.02-0.05 m.
- SearingMeat (agent-reported): s5 false success - pan placed upside down, drumstick on its bottom (no upright test).
  s28: check_obj_location_on_stove returns the first lit burner within 0.15 m in a fixed order, so with rear_right on the
  named 'center' burner can never count. Near misses s3 (pan 1.2 mm past 0.15 m), s29 (shrimp on the rim). close_events
  miss pan grasps whose width only drops to ~0.06 (s13 s20 s24).
