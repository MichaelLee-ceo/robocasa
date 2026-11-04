# LoadDishwasher: stage-level diagnosis (GR00T N1.5, 30 episodes, 12 succeeded, 18 failed)

Task: put the cup and the bowl from the counter onto the top rack of the dishwasher, in either order (stage 0 = one
dish on the rack, stage 1 = both), then close the door (stage 2). Horizon 1200 steps.

The scene starts with the door open and the top rack slid fully out over the door. A successful close has three
steps: push the rack back into the tub, go down to the outer edge of the door, then get the fingers under the door and
lift. The lift takes 30-60 steps (0.1 to 1.0 closedness).

Files:
- Per-episode results: `LoadDishwasher_s<N>_single.json`.
- Contact sheets: `frames/`.
- Running notes: `_notes.md`.
- Useful comparison sheets:
  - `cmp_door_lift.png`: door lift in successes vs s9/s6.
  - `cmp_L10_release.png`: s15 vs s7/s29 release.
  - `succ_C_overview.png`: s23 vs s27, same kitchen and same objects.

## Counts

| root-cause stage | episodes |
|---|---|
| 0 (first dish) | 6: s3, s5, s6, s14, s15, s20 |
| 1 (second dish) | 8: s2, s8, s16, s17, s19, s23, s25, s28 |
| 2 (close the door) | 4: s1, s4, s9, s11 |

The failed stage (first incomplete stage) differs from the root-cause stage in five episodes:
- s3 and s6 failed at stage 2, but stage 0 was slow: 546 and 703 steps (median 298).
- s14 and s20 failed at stage 1, but stage 0 was slow: 922 and 645 steps.
- s5 failed at stage 1, but stage 0 only completed at 1073.

| primary category | episodes |
|---|---|
| stall_holding_at_target (holding over the rack, or parked at the dish just placed) | 5: s3, s4, s14, s17, s20 |
| missed_grasp (bowl) | 4: s5, s6, s16, s25 |
| placed_then_fell_or_slid (released at the rack edge) | 3: s15, s23, s28 |
| placed_off_target (cup dropped into the bowl) | 2: s2, s8 |
| stall_at_fixture (door) | 2: s9, s11 |
| wrong_fixture_element (neighbouring cabinet handle) | 1: s1 |
| wander_away | 1: s19 |

Recurring secondary categories:
- ran_out_of_time: s3, s5, s6, s14, s19, s20, s23, s25
- missed_grasp_continued_empty (empty trip to the rack after a missed bowl grasp): s5, s6, s25

## Recurring failure mechanisms

### 1. Freezing at the dish: holding it over the rack, or parking at it after release (5 primary; time loss in 3 more)

There are two forms.

**Holding a gripped dish just above the rack without opening:**

| episode | hold | outcome |
|---|---|---|
| s14 | cup, 260-917 (~650 steps) | then fell on its side |
| s3 | bowl, 212-543 (~330 steps) | |
| s1 | cup, 420-720 (~300 steps) | |
| s23 | bowl, ~700-903 | |
| s22 (success) | bowl, ~200-456 | |

In successes the whole grasp-to-release time is 70-310 steps (median ~120).

**Releasing, then keeping the open fingers on the placed dish instead of moving on:**

| episode | parked at | for | what it never did |
|---|---|---|---|
| s4 | bowl, 866-1200 | 334 steps | never went to the door |
| s17 | bowl, 310-1200 | ~890 steps | never touched the cup |
| s20 | cup, 373-980 | ~600 steps | bowl was still over the rack at the timeout |

The parking form also delays the official stage completion, because the stage requires the gripper to be off the
dish. Stage 0 first held only at 645 (s20) and 656 (s17).

Nothing in the scene explains the freeze: the dishes look correctly placed and nothing obstructs them. s17 and s22 share
the same view (layout 8, robot left of the dishwasher). s22 hesitated for 250 steps and then went on.

### 2. Releasing at the rack edge: the dish falls onto the open door (3 primary)

| episode | dish | where released | where it ended |
|---|---|---|---|
| s15 | cup (299) and bowl (582) | side edge of the rack | Cup fell 0.45 m to the door level; the bowl slid onto it. The policy pushed the rack in and closed the door over both. |
| s28 | cup (595) | lowered to the rack's front rail | Fell 0.36 m onto the open door. The door was closed over it at 740, and it ended near the hinge. |
| s23 | bowl, carried tilted (903) | front edge; came to rest on the open door, leaning on the front wires | Released 0.335 m from the tub box, vs 0.09-0.255 in successful releases. The policy pushed the rack in without it. The rising door slid it down to the hinge (stage 1 undone at 1169, end z 0.14). |

What successes do differently:
- s7 (same kitchen and side as s15) approaches along the same side rail but releases over the rack interior.
- s27 (same kitchen and objects as s23) puts the bowl first, deep in the rack.
- Successful releases fall at most 0.18 m; edge falls are 0.35-0.45 m.

In every case the policy did not notice and carried on with the rack and the door.

### 3. Second dish dropped into the first (2 primary)

In s2 and s8 the bowl went first. The cup was then released directly above it and ended inside the bowl (s2 on its
side, s8 upright), touching only the bowl and not the rack. s2 then closed the door correctly (639); s8 poked at the
nested pair until the timeout.

In the other bowl-first episodes where the cup reached the rack, it went beside the bowl: s0, s1, s3, s9, s13, s22, s27
and s29. Five of these succeeded; s1, s3 and s9 failed later, at the door.

### 4. Bowl grasp failures (4 primary, plus s19's late slip)

The policy pinches the bowl's rim:

| episode | bowl | what happened |
|---|---|---|
| s5 | wooden bowl_7 | Missed at 209 and 415, with two empty trips to the rack. Lifted only at 943. |
| s6 | black bowl_5 | Missed at 175, then an empty trip to the rack. |
| s16 | flat bowl_9 | Rim pinch with no grip. The bowl was dragged off the counter edge onto the floor (0.88 m), then the door was closed without it. |
| s25 | black bowl_5 | Slipped at 536 after a 1.9 cm lift. Two more tries; it was gripped only at 1131. |

**Scene dependence.** All three layout-6 episodes (s16, s19, s25) failed on the bowl. The black bowl_5 succeeded in
s21, s22 and s24, so the bowl model alone does not explain it.

### 5. Door-closing failures, once both dishes are on the rack (4 primary, 2 contributing)

- **s9 and s11.** The gripper parked at the door's outer tip or corner for 290-440 steps without getting under it.
  The door stayed at 0.00-0.08. The rack had been pushed in correctly.
- **s1.** The gripper went to the bar handle of the cabinet left of the dishwasher (832-1200) and closed on air there
  three times.
- **s4.** Never left the bowl (mechanism 1).
- **s6.** One lift reached 0.26 and slipped back, about 50 steps before the timeout.
- **s3.** The lift started only at ~1100 and reached 0.165.

The door-top stall also occurs in successes: s10 spent 640-760 there and s26 520-900, at the control strip, before
going under the door. It is a common policy mode that recovers only sometimes.

**Scene dependence.** Both layout-5 episodes with the robot right of the dishwasher (s6, s9) failed at the door. This
is the only kitchen with a black door, and n = 2.

## What successful episodes do differently at the critical moments

- **Release:**
  - They open within ~70-310 steps of the grasp.
  - The release is over the rack interior, 0.09-0.25 m from the tub box.
  - The drop is ≤ 0.18 m, and the arm leaves the dish within ~10-40 steps.
- **Second dish:** it goes beside the first, never into it.
- **Door:**
  1. They push the rack in.
  2. They go low beside the door's outer edge (the wrist shows the floor tiles).
  3. The wrist view fills with the grey door underside, and the door rises from 0.1 to 1.0 in 30-60 steps.

  Stage 2 takes 114-275 steps, except s26 (570).

## Official-criterion issues and packet notes

1. **No upright test.** Dishes lying on their side count as placed: s0, s6, s7, s9, s12, s14, s19, s21, s22, s24, s25,
   s26, s27 (tilt 46-90 deg).
2. **Rack contact means any contact with the rack body.**
   - s23: the bowl resting on the open door counted as on the rack for 264 steps (904-1168), because it leaned on the
     front wires. The task notes say a dish on the door should not count.
   - s16: a falling bowl counted for 1 step (533).
3. **A dish nested in another dish on the rack does not count** (s2, s8). This is correct by the letter of the
   criterion, but a human would call it loaded.
4. **The release part of stages 0/1 latches after the first release.** In s4, stage 1 holds 866-1200 while the
   gripper touches the bowl almost continuously. The definition in `_task.json` is stricter than what is measured.
5. **The door fixture's `done` is not stage 2.** The door was closed while the loading was incomplete in s2, s15, s16
   and s28.
6. **`dist_to_target` is misleading for this task.**
   - It measures to the tub interior box, but the rack starts slid out. A dish on the rack reads 0.1-0.33 m "away"
     while at target, and drops to 0.0 only once the door pushes the rack in.
   - It does not separate edge releases from good ones (s15 0.08-0.13, s28 0.18).
7. **Other packet fields that need care:**
   - `fixtures.stage2.gripper_touch_intervals` includes contacts with the rack (the rack is part of the dishwasher),
     so it is not "door touched" (s8, s17, s20).
   - Forearm pushes on the rack are invisible (s6).
   - `holds` lists zero-lift finger rests as holds (s17 has ten).
   - `lift_m` is small even for good transports, because dishes are carried down into the rack.
8. **s22 `replay_matches_online_log=false`** is a 1-step contact flicker at the logged step 464, not a real
   disagreement.

I found no case where the packet's stage intervals contradict the video in a way that suggests a tooling bug. The
mismatches above are definitional.

## Uncertainty

- **Door-stall geometry (s9, s11).** It is medium confidence: I cannot resolve from 256-px frames whether the fingers
  press the door's lip, sit just past it, or the arm is at its reach limit.
- **s6 root cause.** I assigned it to stage 0 (time). Since the same kitchen in s9 never lifted the door above 0.08,
  it could equally be a stage-2 door failure.
- **s3** loses time in three places. The door lift was slow when it started (0.08 in ~80 steps), so it might not have
  finished even with more time.
- **Assigning the parking freezes to stages:** s4 to stage 2, s17 to stage 1, s20 to stage 0, by where the time went.
  It is one mechanism.
- **Robot side.** Which side of the dishwasher the robot is on was read from the agent view, not from the packet.
- **Process note.** Early in this run, helper scripts with the same name in the shared scratchpad collided with other
  agents' scripts. All sheets used here were made or re-made with my private helper and checked.
