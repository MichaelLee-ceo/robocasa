# StackBowlsCabinet: stage-level diagnosis (GR00T N1.5, 30 episodes, 21 succeeded, 9 failed)

Task: put the small bowl (bowl1, scaled x0.7) and the large bowl (bowl2, scaled x0.95) stacked, small on top, into an open upper cabinet. Horizon 1400 steps.
- Stage 0: one bowl inside the cabinet and not touched.
- Stage 1: both bowls inside, nested, gripper at least 25 cm away.

Files:
- Per-episode results: `StackBowlsCabinet_s<N>_single.json`. Successes carry an extra `strategy` field.
- Running notes: `_notes.md`. Contact sheets: `frames/`.
- Cross-episode comparison sheets:
  - `cmp_nest_release_wrist.png`: wrist view at the nest release.
  - `cmp_pair_grasp.png`: grasp of the nested pair.
  - `cmp_first_close_wrist.png`: first close on the small bowl.
  - `cmp_L1_release.png`: release in the cabinet, layout 1.
  - `cmp_L8_place.png`: start and placement in layout 8.

Bowl sizes come from the mesh extents × the registry scale of 2.0 × the object scale. bowl1 is 0.16-0.18 m across and bowl2 0.22-0.25 m, so bowl1 is always the smaller bowl, as the instruction assumes.

## Counts

| root-cause stage | episodes |
|---|---|
| 0 (first bowl / stack into cabinet) | 6: s0, s3, s13, s20, s22, s23 |
| 1 (both bowls nested in cabinet) | 3: s1, s5, s17 |

| primary category | episodes |
|---|---|
| missed_grasp | 3: s5, s17 (lone second bowl after a failed nest), s20 (first grasp of the small bowl) |
| stall_holding_at_target | 2: s3, s13 (held the small bowl at the large bowl's rim, never nested; far-island layout) |
| dropped_in_transit | 1: s1 (pair slipped out during the lift) |
| placed_off_target | 1: s22 (stack put on the wrong upper storage fixture) |
| placed_then_fell_or_slid | 1: s23 (stack released on the cabinet lip, fell out) |
| other: knocked the large bowl upside down while nesting | 1: s0 |

**The stage labels hide where the failures happen.** In the packet, stage 0 is "a bowl in the cabinet", but the policy's first sub-task is nesting the small bowl into the large one on the counter. By physical sub-task:

| sub-task where it went wrong | episodes |
|---|---|
| nest on the counter | 6: s0, s1, s3, s5, s13, s17 |
| first grasp | 1: s20 |
| carry / place the pair in the cabinet | 2: s22, s23 |

How the nest failures are split between stages:
- All three stage-1 failures (s1, s5, s17) start with a failed nest during stage 0's window (steps 160-194).
- The stage-1 label only reflects that one bowl later reached the cabinet alone.

**Strategy.**
- All 21 successes used nest-then-carry. s16 first moved the large bowl next to the small one, then nested and carried.
- 8 of 9 failures attempted nest-then-carry. s20 missed its first grasp and went one-by-one.
- Every episode that ended up one-by-one failed (0/4: s1, s5, s17, s20).
- The policy only falls back to one-by-one after an error, and it cannot then finish a lone bowl:
  - s5 and s17 could not grasp the second bowl.
  - s20 dropped the large bowl.

Checklist from the task notes:

| failure type | episodes |
|---|---|
| bowls side by side in the cabinet | none |
| wrong nesting (large inside small) | none |
| small bowl fell out of the large one while carrying | none; in s1 the whole unseated pair slipped from the fingers |
| on the cabinet edge | s23 (lip, fell); s22 (adjacent shelf, 0.17-0.21 m outside) |
| bowl dropped | s1 (pair, ~485); s20 (large bowl, ~453) |
| stall at the cabinet | none; the stalls were at the bowls on the counter (s3, s13, s17, s20) |
| ran out of time | none; every failure was a dead end, not a slow but progressing episode |

## Recurring failure mechanisms

### 1. Nesting the small bowl into the large one fails, and the policy cannot recover (6 of 9)

Successes (21/21):
- They release the small bowl directly over the large bowl.
- In the wrist view the held bowl overlaps the large bowl's interior, with the large rim visible around it (`cmp_nest_release_wrist.png`, bottom row: s2, s9, s24, s28).

Failures (`cmp_nest_release_wrist.png`, top row):
- The large bowl is barely or not visible at release, so the small bowl was offset.
- s5 (179), s13 (423) and s17 (160): the small bowl was released beside the large bowl.
- s1 (194): it was left leaning on the large bowl's rim (tipped 204-338).
- s3 (360-540): it was held at the rim for 180 steps and never released.
- s0: the hanging small bowl hit the large bowl's rim at ~168, and the large bowl flipped upside down (175-232).

What came next:
- **Pair grasp on an unseated stack:**
  - s1: closed across both rims at 438 and lifted 0.3 m.
  - The pinch slipped at ~485 and both bowls fell back onto the counter.
  - Re-grasps then pushed the small bowl off the counter onto the robot's own base (z 0.42 m, ~720).
- **One-by-one fallback, second bowl ungraspable:**
  - s17: carried the small bowl in alone (stage 0 at 305). It then spent 403-1316 side-on at the large bowl's outer wall near the backsplash: 4 closes and 10 contacts, max lift 8 mm.
  - s5: carried the large bowl in alone (stage 0 at 718). Then 3 closes on the small bowl's rim (879, 1145, 1381) lifted nothing. Twice it went on to the cabinet empty-handed.
- **Inverted bowl (s0):** about 10 grasps on the dome (max lift 5 mm) and 6 closes on air. It never tried to right the bowl.

The policy never re-placed the small bowl properly into the large one after a failed nest.

### 2. Bowls spawned away from the canonical spot (4 of 9 failures, 0 of 21 successes)

- All 21 successes start with both bowls 0.45-0.49 m from the cabinet interior.
- Each of the four episodes with a bowl farther than 0.5 m failed:
  - s3 and s13: 2.4-2.8 m; layout 9, bowls on a kitchen island.
  - s22: 0.56 and 0.86 m.
  - s20: 0.57 m.
- 4/9 vs 0/21 (Fisher p ≈ 0.005). n is small and the layouts differ, so treat this as a strong hypothesis.

What happened in each:
- **s3 and s13:**
  - The base drove to the island in about 200-300 steps.
  - Nesting failed: s3 held the small bowl at the rim, then held it for 850 more steps while the arm wandered. s13 released it beside the bowl, lifted it 0.53 m and dropped it at 617, then hovered at the rim 806-1360.
  - Neither ever drove back toward the cabinet.
- **s22:**
  - The target is the green cabinet over the sink. In s4, s10 and s17 (also layout 8) the target is the wooden upper cabinet next to it.
  - The nest and pair lift were clean (391-536).
  - The stack was put on the adjacent wooden shelf at 539, 0.17-0.21 m outside the interior. The robot then wandered.
- **s20:**
  - The small bowl sat nearer the robot.
  - The first close at 168 had the gripper centred over the bowl rather than over the rim (`cmp_first_close_wrist.png`), and it missed.
  - It switched to the large bowl, which slipped out at ~453. A second miss at 609 was followed by hovering to the end.

### 3. Pair placed at the cabinet's front lip (s23)

- The nested pair was carried with the inner bowl tilted more than 45 degrees (from step 380).
- It was released at 456, lower and nearer the lip than in the layout-1 successes s7 and s27 (`cmp_L1_release.png`).
- Stage 0 held 436-463. The stack then tipped out and fell 0.45 m onto the counter, upside down (464-484).
- The robot righted the large bowl (~780) but did not re-nest in the remaining ~600 steps.

## What successful episodes do differently

- **Small-bowl grasp:** closes once on the rim at step 57-108 and lifts 0.10-0.24 m. The bowl sits off to one side of the wrist image, with the rim across the finger gap.
- **Nest release:** directly over the large bowl at 127-228. The bowl seats, sometimes after pressing on the large bowl's rim (s6, s21, s24, s25).
- **Pair grasp:** one close on the double rim 15-60 steps later lifts both bowls 0.52-0.62 m.
  - 18/21 lifted the pair on the first pair grasp.
  - s4 re-closed after a miss at 225/245, losing ~165 steps.
  - s14 aborted a 7 cm lift (264-294) and re-grasped at 354.
  - s21 re-closed 245 → 266.
- **Carry:** the inner bowl tilts more than 45 degrees during the carry in 13/21 successes, but it stays nested. Contact plus centre distance is enough for the check, so this is tolerated.
- **Placement:** the stack is pushed well inside the cabinet, and the gripper retreats 10-30 steps after release.
- **Slow successes:**
  - s4: stage 0 at 503, pair re-grasp.
  - s14: 453, aborted lift.
  - s16: 427, moved the large bowl first.
  - s21: 395.
  - All recovered errors; none came near the horizon. Every success finished by step 518.

## Official-criterion issues

1. **Nesting direction and upright are not checked.** `_check_success` accepts bowl2-in-bowl1 or bowl1-in-bowl2, so "smaller on top" is not enforced, and there is no upright test. Neither case occurred here: every nest was small-in-large.
2. **The "inside" tolerance is loose.** `obj_inside_of` uses th/|edge| per axis, often 10+ cm.
   - s23: the stack perched on the front lip counted as inside (stage 0 held 436-463) just before it fell out.
   - A stack resting on the lip with the gripper away could score.
3. **Stage 0 fires before the release under nest-then-carry.** The inner bowl is inside the cabinet and untouched while the pair is still held.
   - Seen in s8, s10, s12, s15, s16, s19 and s26, 22-42 steps before the release.
   - So the typical "stage 1 = 28 steps" mostly measures the gripper retreating, not stacking.
   - The nest sub-task, where 6/9 failures originate, has no stage of its own. A "nested on the counter" sub-stage would make this task's stage analysis meaningful.
4. **Tilted nests count.** s15 ends with the inner bowl tilted 44 degrees inside the outer one and counts as nested. That seems acceptable.

## Packet and tooling issues (possible bugs)

1. **`undone_events` reason mislabel (bug).**
   - `make_packets.py` labels a drop "an object left the target" whenever the number of at-target objects changes, including when an object enters.
   - s26 step 255: bowl1 stayed inside (at_target 249-304) and bowl2 entered, yet the packet says "left". The real cause was the gripper touching bowl1 (255-270).
   - Fix: check that every object at the target at e_ is still there at e_+1.
2. **Contacts by a held object or arm link are not recorded.**
   - s0: the large bowl was flipped at 175, but bowl2 has no gripper touch until 288.
   - s3: bowl2 was pushed 18 cm with an empty touch list.
   - So the packet cannot show knock-overs caused by a carried bowl.
3. **`holds` overstates grasps.** s17 has 10 "holds" on bowl2 with lift ≤ 8 mm; s0 has 9 with ≤ 5 mm. Most are side contacts.
4. **The official labeler breaks on failures.** It reports "expected 2 pick stages" for s0, s13 and s20. Context only.

Replay and online logs agree in all 30 episodes.

## Uncertainty

- **Medium confidence on the "why" of the missed grasps.**
  - s20: the centred approach is clear, but why it failed is not.
  - s5: a width of 0.014 after the close; the shallow small bowl_5 might leave too little rim to pinch.
  - s17: side-on next to the backsplash; it is unclear whether the backsplash physically blocks a top-down grasp.
- **Exact landing spot of the small bowl at the failed nests** (s1, s5, s13, s17): the agent view is often blocked by the arm.
- **s22:** whether the stack rests on a separate wooden cabinet or on a ledge of the green one.
- **s0:** which part (the held bowl, the fingers' outside, or the wrist) hit the large bowl.
- **Scene dependence (mechanism 2)** rests on 4 episodes.
- **Housekeeping:** one stray sheet (`frames/s0_close.png`) was written by another agent through a shared scratchpad helper and has been deleted. `s0_overview.png` and `s0_flip.png` were regenerated with a private helper.
