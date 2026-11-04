# PackIdenticalLunches: stage-level diagnosis (GR00T N1.5, 30 episodes, 4 succeeded, 26 failed)

Task: carry two vegetables and two meats, one at a time, from the open fridge to two tupperwares on a counter. The robot drives between the fridge and the counter on every trip. Each tupperware must end with exactly one vegetable and one meat. Horizon 2600 steps.

- Stages 0-2 = "at least 1 / 2 / 3 items in some tupperware".
- Stage 3 = the correct composition.
- The item types vary by episode: vegetables include onion, lemon wedge, garlic, broccoli, pepper and others; meats are drumstick, steak or fish.

Files: per-episode `PackIdenticalLunches_s<N>_single.json`, contact sheets in `frames/`, running notes in `_notes.md`. `frames/compare_tupperware_choice.png` shows the wrist view at the moment of the decisive placement in 6 failures and 4 correct cases.

## Counts

| root-cause stage | n | episodes |
|---|---|---|
| 0 | 10 | s0, s14, s16, s17, s21, s23, s25, s26, s27, s28 |
| 1 | 8 | s2, s4, s7, s15, s20, s22, s24, s29 |
| 2 | 6 | s1, s5, s11, s12, s13, s19 |
| 3 | 2 | s6, s10 |

Root cause differs from the first incomplete stage in 9 episodes:
- An item was dropped in stage 0, so 4 items were never available: s0, s17.
- Stage 0 took 1519 steps: s21.
- A wrong-tupperware placement in an earlier stage made stage 3 impossible: s1, s2, s19, s20, s29.
- The episode timed out while the third item had just been released: s5.

| primary category | n | episodes |
|---|---|---|
| placed_off_target (all of them: the wrong tupperware, giving two of a kind) | 6 | s1, s2, s6, s19, s20, s29 |
| missed_grasp | 5 | s4, s12, s21, s26, s27 |
| missed_grasp_continued_empty | 5 | s11, s13, s14, s16, s22 |
| dropped_in_transit | 3 | s0, s17, s28 |
| other: wrong destination (a counter with no tupperware) | 3 | s7, s15, s23 |
| wander_away | 2 | s5, s24 |
| stall_pre_grasp | 1 | s25 |
| other: froze in the tucked carry pose holding the last item | 1 | s10 |

Grouped by failure type:
- Grasp-related: 11 (missed_grasp, missed_grasp_continued_empty, s25's stall)
- Composition: 6, plus 2 more as a secondary category (s5, s17)
- Drops: 3 primary, but 12 items in 11 episodes ended off target after being carried (see mechanism 3)
- Navigation or destination: 5 (s7, s15, s23, s24, s5)

## Recurring mechanisms

### 1. A failed grasp at the fridge, then the policy acts as if it holds the item (11 episodes, the largest group)

**What fails at the grasp.** The fingers close on the item and it squeezes out, or they close on air 5-10 cm above it. Object types:

| object | episodes and steps |
|---|---|
| lemon/lime wedges (squeeze out sideways) | s27 @51, s12 @1025, s13 @1034/1688/2472 |
| thin steak slices lying flat (fingers close past them) | s25 @2275, s26 @128/151/644 |
| thick steaks | s4 @591, s22 @1789 |
| fish | s11 @509/1715 |
| drumstick closed on its bone tip | s21 @89 |
| avocado | s14 @1428 |
| closes on air above mushroom/steak through the glass shelf | s16 @89/777 |

**What happens next.** One of two things, often both in the same episode.

**(a) Empty trip.** The fingers stay closed and the robot drives to the counter, makes an empty "place" over a tupperware, and drives back.
- 13 episodes contain at least one empty place: s0, s1, s5, s11, s13, s14, s16, s17, s21, s22, s25, s26, s28.
- Each one costs ~300-500 steps.
- s14, s13 and s28 made three empty places each.

**(b) Closed-finger stall at the fridge.** The arm hovers with closed fingers for hundreds to thousands of steps:
- s27: 72-2600
- s12: 1048-2600
- s21: 112-1144 and 1784-2600
- s25: 720-1528 and 1840-2176

Often it ends in a tucked pose, with the wrist camera looking back at the robot's own body and nothing moving: s4 700-2600, s16 960-2600, s22 1830-2600, s26 1350-2250.

**The same tucked pose while holding an item.**
- s10 froze ~700 steps holding the last pepper (1830-2510), never drove off, and the pepper slipped to the floor.
- s15 carried a steak in the tucked pose for 1000 steps.

The pose looks like the carry/navigation posture with the base not moving. That is my interpretation, not verified.

**Retries are rare and weak.** It retried the same flat item (s13, s26) and never switched approach. Switching to a different item after a failure happened only in s11 (switched to broccoli) and s21 (switched to the tomato), and both then succeeded once.

### 2. Wrong tupperware: two of a kind in one tupperware (6 primary + 2 secondary)

| episode | decisive placement | result |
|---|---|---|
| s2 | 2nd meat into the meat tupperware, step 503 | |
| s20 | 2nd garlic onto the 1st, ~950 | |
| s29 | 2nd broccoli next to the 1st, 821 | |
| s1 | 3rd item, a veg, into the full meat+veg tupperware, 1146 | |
| s19 | 2nd pepper next to the 1st, 1545 | sorted by kind: 2 meat / 2 veg |
| s6 | 4th item, a lime, into the full fish+lime tupperware, 1257 | |
| s5 | 2nd meat with the 1st, 2586 | |
| s17 | 2nd eggplant into the full tupperware, 1626 | |

**These are not time failures.** s1, s2, s6, s19 and s20 had all four items in tupperwares by 1146-2267.

**The policy does not appear to use the contents.** The compare sheet shows that at the decisive moment both tupperwares and their contents are clearly visible in the wrist camera, in the failures as well as the successes. The choice looks unrelated to what is already inside.

**It matters most when the second item is the same kind as the first.** Then it must go to the empty tupperware:
- right in 4/7 (s6, s8, s11, s12)
- wrong in 3/7 (s2, s20, s29)

**The 3rd/4th placement must go to the tupperware lacking that kind.** Right in s3, s6 (3rd), s8, s9, s10, s18; wrong in s1, s5, s6 (4th), s17, s19.

No mechanism for the choice is visible. The successes look like getting every choice right by chance.

### 3. Items lost in transit from tip or edge grasps (3 primary, in 11 episodes overall)

Items held by a tip or edge fall out on the way:
- **Floor (8 items):** s0 onion ~170, s7 wedge ~850 and steak ~1673, s10 pepper 2513, s14 steak 157, s17 T-bone 207, s23 tomato 427, s28 potato ~245
- **Robot's own base:** s15 steak ~1640, s25 garlic ~205
- **Robot's own arm:** s0 drumstick ~1640-1687
- **Bottom of the fridge:** s29 drumstick 1055

The policy never noticed a dropped item and never tried to pick one up from the floor.
- A drop of the first item makes the task impossible: s0, s17.
- The drop is followed by an empty "place" at the counter: s0, s14, s25, s28.

### 4. Destination and navigation errors (5)

**Layout 10 has two tupperware sites in this set:** the microwave corner left of the fridge (s7, s15) and the toaster counter right of it (s17, s23, s29). The policy goes to the wrong one:
- s7 and s15 placed the first item correctly on the left. They then carried item 2 to the right counter, where there is no tupperware, held it there 130-1000 steps, and it fell.
- s23 carried its first item to the left corner while the tupperwares were on the right (inferred from positions), and later took the second item to the same wrong place.

**Other navigation failures:**
- s24 placed one item and then never drove back to the fridge: 2270 steps at the counter with open fingers.
- s5 idled/drifted 770 steps after stage 1.

### Scene dependence (strong)

| layout | fridge-to-tupperware distance (mean start dist) | items placed | successes |
|---|---|---|---|
| 5 | 0.64 m (tupperwares right beside the fridge) | 20/20 | 3/5 (s8, s9, s18) |
| all others | 0.86-1.57 m | 42/100 | 1/25 (s3, layout 2) |

- Both layout-5 failures (s2, s6) are pure composition errors.
- Layout 10 (1.57 m, two sites): 0/6, 8/24 items.
- Layout 7: 0/6, 8/24.

Object types also matter:
- Steak episodes: 1/12 success (flat slices and thick steaks are hard to pinch).
- Every lemon-wedge episode (s7, s12, s13, s27) failed, and in s12, s13 and s27 the wedge grasp itself failed.
- Layout and objects are both randomised, so I cannot separate their effects. Layout 5 contains fish, steak and drumstick episodes alike.

## What successful episodes do differently at the critical moments

- **Every item lifted on the first attempt.** In all 16 placements of the 4 successes: no missed grasp, no drop, no empty trip.
- **Content-correct tupperware choices:** pair first, then the other tupperware (s3, s9), or alternate (s8, s18).
- **Short trips.** 3 of 4 are in layout 5, with the least driving.
- **Stage times** of 200-528 steps.
- **The only recovered error:** s8 released a drumstick on the rim at 603 (0.15 m from the centre) and pushed it in at 650.

## Official-criterion and packet issues

1. **A stacked item does not count as "in".** `check_obj_in_receptacle` requires contact with the tupperware itself.
   - s20: the second garlic rested on the first and counted only 73 steps later, when it tipped (stage 1 at 1027 vs release ~950).
   - s0: a drumstick lying on an onion loses tupperware contact intermittently (128 at_target intervals, ~140 stage-1 undo events).
   - A correct pair could therefore fail the final check depending on how the meat rests.
2. **Contact flicker appears as undo events.** Hundreds of 1-3-step `undone_events` labelled "an object left the target" are contact flicker of items lying in the tupperware, not removals (s0, s1, s3, s6, s7, s21, s22). The packet labels cannot distinguish the two.
3. **No upright test.** Items on their side count (s3 mushroom at 104 deg, s7 wedge at 115 deg). Harmless here.
4. **Gripper-distance clause in s5.** Stage 2 is false at the end only because the release came 14 steps before the timeout.
5. **No composition field.** The packet has no field for which tupperware an item is in, and the tupperwares are not tracked objects (no contacts or holds), even though composition is the main stage-3 failure. I inferred membership from item end positions and checked it in the video.
   - This is also why the s24 note ("grabbed the empty tupperware") cannot be checked in the packet. In the video, the fingers stay open 336-2600 and the tupperware does not move, so I could not confirm a grasp.
6. **Measurement traps:**
   - `lift_m` is relative to the start height. Items carried from a fridge shelf (z 1.36-1.43) down to the counter (0.95) show lifts of 0.016-0.05 m, e.g. s3 veg1 0.02, s4 veg1 0.016, s15 meat0 0.037, s28 veg0 0.017. A small lift does not mean a failed grasp in this task.
   - `close_events` misses grasps of wide items whose finger width never drops below the open threshold (s22 broccoli 66-260, s3 mushroom 84-259).
   - Items that end on the robot's own body (s0 z 1.11, s15 z 0.41, s25 z 0.40) are not flagged and get meaningless end distances.
   - In s20, `holds` ends at 182 while finger contact (and the carry) continues to 285.

## Uncertainty

- **s5:** I classified the 770-step idle (1148-1920) as wandering. It could also be read as a stall at the counter.
- **s23:** where the tupperwares stand is inferred from layout-10 positions; I never saw them in that video.
- **Mechanism 2:** I see no visual cue that separates right from wrong choices. "Looks random" rests on 16 placements.
- **The tucked-pose interpretation** (navigation posture without base motion) is a guess from the camera views.
