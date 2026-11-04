# Stage-level failure diagnosis: all 16 composite-seen tasks (480 episodes)

GR00T N1.5, official RoboCasa365 Composite-Seen checkpoint-60000, frozen scenes, policy_seed 0 (`../run/`). Stage
completion and timing come from the official stage conditions evaluated every control step (`../analysis/`,
`../packets/`). The physical cause of every failure comes from a per-task agent that watched the video at the events,
following `AGENT_BRIEF.md` (`<Task>/<tag>.json`, `<Task>/_summary.md`). Units: 1 step = 0.05 s.
Aggregation: `aggregate.py` -> `_aggregate.json`.
Tooling and official-criterion issues found along the way: `_tooling_issues.md`.

## Outcome
- **185 / 480 successes (38.5 %); 295 failures.**
- The root-cause stage equals the first incomplete stage in 258 of 295 failures. The other 37 split as follows:
  - 19 lie in an earlier stage: it used up the time or was only nearly met.
  - 18 lie in a later stage. Example: StoreLeftoversInBowl, where the falling bowl undoes "food in bowl".
- `ran_out_of_time` is never the primary cause. It appears as a secondary tag 32 times, almost always behind a stall.

## Failure modes (primary category of each failure, grouped by what a fix would have to address)
| mode | n | % | categories |
|---|---|---|---|
| fixture actuation | 72 | 24 | actuation_insufficient 25, wrong_fixture_element 20, stall_at_fixture 17, approach-retreat 5, missed button 5 |
| grasp acquisition | 66 | 22 | missed_grasp 47, stall_pre_grasp 11, missed_grasp_continued_empty 8 |
| placement | 47 | 16 | placed_off_target 31, placed_then_fell_or_slid 13, release_timing 3 |
| grasp retention | 29 | 10 | dropped_in_transit 19, slipped in hand (spatula, mug pitched in the grip) 10 |
| task level | 28 | 9 | skipped_or_wrong_stage 12, wander_away 11, wrong destination 4, freeze 1 |
| stage not terminated | 21 | 7 | holds at the target without releasing 18, no withdraw after a press 3 |
| process skill | 18 | 6 | scrub too little / above the board, stir misaimed, lettuce held beside the stream |
| disturbance | 12 | 4 | knocked a placed object / the pot, undid an earlier stage |
| criterion / sim | 2 | 1 | ScrubCuttingBoard s12 (grasp test), GetToastedBread s27 (lever jammed) |

Per task (failures by mode, root-cause stages):

| task | fail | fixture | acq | place | retain | task | no-term | process | disturb | crit | root-cause stage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DeliverStraw | 28 | 14 | 2 | 3 | 2 | 2 | 5 | | | | s0 16, s1 12 |
| GetToastedBread | 30 | 15 | 4 | | | 10 | | | | 1 | s0 22, s1 8 |
| KettleBoiling | 10 | 2 | 7 | | | | | | 1 | | s0 8, s1 2 |
| LoadDishwasher | 18 | 3 | 4 | 5 | | 1 | 5 | | | | s0 6, s1 8, s2 4 |
| PackIdenticalLunches | 26 | | 11 | 6 | 3 | 6 | | | | | s0 10, s1 8, s2 6, s3 2 |
| PreSoakPan | 12 | 9 | 2 | | | 1 | | | | | s1 2, s2 10 |
| PrepareCoffee | 24 | 5 | 5 | 5 | 6 | | 3 | | | | s0 16, s1 8 |
| RinseSinkBasin | 13 | 9 | 1 | | | 3 | | | | | s0 4, s1 9 |
| ScrubCuttingBoard | 17 | | 3 | | 1 | | 5 | 7 | | 1 | s0 3, s1 14 |
| SearingMeat | 21 | 2 | 6 | 9 | 4 | | | | | | s0 17, s1 2, s2 2 |
| SetUpCuttingStation | 16 | | 7 | 2 | 4 | 3 | | | | | s0 12, s1 4 |
| StackBowlsCabinet | 9 | | 3 | 2 | 1 | | 2 | | 1 | | s0 6, s1 3 |
| SteamInMicrowave | 20 | 11 | 2 | 2 | 1 | 1 | | | 3 | | s0 2, s1 6, s2 6, s3 6 |
| StirVegetables | 21 | | 4 | | 5 | 1 | | 7 | 4 | | s1 1, s2 4, s3 16 |
| StoreLeftoversInBowl | 19 | | | 13 | 2 | | 1 | | 3 | | s0 1, s2 18 |
| WashLettuce | 11 | 2 | 5 | | | | | 4 | | | s0 3, s1 8 |

## Mechanisms that recur across tasks
1. **Unstable grasps on thin, flat or small objects, and grasps at the tip or rim** (acquisition + retention: 95 failures, 32 %).
   - Affected objects: knife handle end, straw, spatula end, flat steak, lemon wedge, mug rim, pan handle tip (0 of 12 tip grasps succeeded).
   - The object either is never lifted, or slides out 20-45 steps after lifting, or rotates in the fingers when it meets the target (mug against the tray).
   - Some object models fail almost always: mug_8 0/8, Colander008, pan_5 tip grasps, sponge_5 1/5.
2. **No error awareness and almost no retry.**
   - The policy made no recovery attempt at the fatal error in 169 of 295 failures (57 %). It tried and failed in 105.
   - After a missed grasp it often carries on empty-handed: 11 PackIdenticalLunches episodes "place" nothing. In KettleBoiling and SearingMeat it turns the knob with no kettle or pan.
   - A dropped object is never picked up again.
   - Spilled food goes unnoticed: SteamInMicrowave closes the door and presses start with the vegetable on the floor.
3. **Fixtures moved part of the way, then abandoned**, each short of its official threshold:
   - faucet handle: 0.16-0.35 rad vs 0.40;
   - toaster lever: 0.19-0.64 vs 0.9;
   - drawer: 0.38-0.85 normalized vs 0.95;
   - microwave door: closedness 0.51-0.94 vs closed;
   - sink spout: swept to one side only.
   
   Successes typically finish within 20-100 steps of first contact. Failures dwell, let go and leave.
4. **Wrong element.** Cabinet door instead of drawer (DeliverStraw 7), the empty pair on 4-slot toasters, front/rear knob swaps, the neighbouring keypad key (SteamInMicrowave 9 of 20 failures hit a wrong key at some point), and the round spout head instead of the lever.
5. **The stage is not ended.** The policy:
   - holds the object at the target without releasing (LoadDishwasher, DeliverStraw);
   - keeps scrubbing long after the counter is met (ScrubCuttingBoard s7, s24);
   - leaves open fingers on the placed dish;
   - stays at the button after pressing it.
   
   The official "gripper >= 25 cm away" clause turns these into failures.
6. **Release short of the target near fixture edges.** Fridge shelf (StoreLeftoversInBowl: 5-25 cm less deep than in successes), dishwasher rack edge, cabinet lip, coffee tray (0.043-0.072 m vs <= 0.036 m in successes).
7. **Strong scene / model dependence.**
   - PreSoakPan: L1 0/3, L9 3/5, all other layouts 13/13.
   - RinseSinkBasin: 12 of 13 failures on layouts 8-10.
   - PackIdenticalLunches: layout 5 3/5, others 1/25.
   - StackBowlsCabinet: off-spot bowls 0/4.
   - StoreLeftoversInBowl: L9 fails 10/12.
   
   Faucet / fridge / toaster model, layout and style change together, so the factor cannot be separated from these runs.
8. **Task-level errors** (9 %).
   - Bread taken before toasting (GetToastedBread 10). After the pop-up, no episode ever picked the bread (8/8).
   - Knife abandoned for the meat (SetUpCuttingStation 7).
   - Two of a kind in one tupperware (PackIdenticalLunches 6).
   - The wrong counter.

## Official-criterion findings (kept as official; listed so results can be read with them)
- **Successes that look wrong:**
  - SearingMeat s5: pan upside down, tilt 179.4 deg at the end, drumstick on its bottom.
  - StirVegetables s27, s29: scored while the pot was tipped or lifted. The stir counter needs 5 steps of veg movement > 0.5 mm, and only veg1 has to touch the spatula.
- **Failures that look like successes or near misses:**
  - StoreLeftoversInBowl s21, s25, s26, s29: bowl standing on a fridge shelf, outside the shelf regions.
  - PrepareCoffee s29: 4.29 cm vs 4 cm.
  - RinseSinkBasin s29: 0.004 rad short of "left".
  - SearingMeat s3: 1.2 mm past 0.15 m.
  - ScrubCuttingBoard s12: the one-finger grasp test.
  - GetToastedBread s27: lever jammed in sim.
- **Code facts (confirmed in source):**
  - `obj_inside_of` tolerance = 0.05 / edge length.
  - The drawer value is normalized by 0.55 x depth, so 0.95 = 52 % of travel.
  - `check_obj_grasped` effectively tests one finger.
  - DeliverStraw success ignores the drawer and uses a 0.5 m straw-cup threshold.
  - `check_obj_in_receptacle` = contact + xy distance.
  - No composite here checks upright.

## Audit
I checked these claims against the agents' sheets or the packet data, and all were consistent:
- WashLettuce s2: the colander is carried.
- SearingMeat s5: upside-down pan, confirmed by tilt.
- StoreLeftoversInBowl L10: bowls on the shelves.
- StirVegetables s27: pot tipped at the end.
- GetToastedBread s6: the empty toaster pair was pressed.
- DeliverStraw s6: a cabinet door was opened instead of the drawer.
- SteamInMicrowave s11: the press was off START.
- PackIdenticalLunches s24.

PackIdenticalLunches s24 corrected my own earlier note: the hand hovered at the empty tupperware with open fingers. It never grasped it.

Medium confidence is recorded for 109 failures and low for 4; their `uncertain_about` fields say why.

## What this means for the oracle-detection fixes (next phase)
Candidates for a simple retry or resample from the saved state at the detected failure:
- grasp acquisition (66) and retention (29): re-attempt the grasp;
- partial fixture actuation (25) and stalls at a fixture (22): resample, or continue the push;
- stage not terminated (21): a release / withdraw step.

Needs guidance beyond a retry (where to act, which element, which stage):
- wrong element (20);
- task-level errors (28);
- placement short of the target near edges (47), partly.
