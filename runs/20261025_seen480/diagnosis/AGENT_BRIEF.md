# Stage-level failure diagnosis of GR00T N1.5 rollouts — brief for the diagnosis agents

## Goal
A frozen vision-language-action policy (GR00T N1.5, official RoboCasa365 Composite-Seen checkpoint) was run on 16
long-horizon kitchen tasks, 30 fixed scenes each. Every task is a fixed sequence of **stages** (official sub-tasks,
e.g. "pick the kettle and place it on a burner", then "turn on that burner"). For every episode of your task, find out
**what happened stage by stage, and for failures, which stage really failed, when, and why** — the physical cause
(e.g. "placed the mug 8 cm off the spout after holding it at the right spot for 5 s", "pulled the untoasted bread out
instead of pressing the lever", "hovered over the kettle handle for 24 s, so the next stage ran out of time").
The program cannot do this causal reasoning; you do it from the video, using the program's measurements as evidence.

## Files (all paths relative to /home/michaellee/mclee/robocasa/runs/20261025_seen480)
- `packets/<Task>/_task.json`: instruction, stages, plain-language stage definitions (what counts as complete and what
  does not), horizon, and typical steps per stage in this task's successful episodes (median, p90).
- `packets/<Task>/<tag>.json`: one episode (see "Packet fields").
- `run/<Task>/<tag>.mp4`: the rollout video, one frame every 4 steps, each frame = [left agent-view camera | wrist camera].
- `frames.py`: contact sheet at chosen steps. Run with the project env:
  `source ~/miniconda3/etc/profile.d/conda.sh && conda activate /home/michaellee/mclee/robocasa/envs/groot && python frames.py run/<Task>/<tag>.mp4 diagnosis/<Task>/frames/<name>.png <step> <step> ...`
  Give it 6–12 steps per sheet (2 tiles per row; each tile is labelled with its step). Then look at the PNG with Read.
  Options: `--view left|wrist` (one camera), `--zoom X0 Y0 X1 Y1` (crop to those pixels of the 256x256 view, enlarged;
  use it for small details such as a burner flame, a knob, the gripper fingers on a handle), and
  `python frames.py --compare <out.png> run/<Task>/<tagA>.mp4:<step> run/<Task>/<tagB>.mp4:<step> ...` to put the same
  moment of several episodes side by side (e.g. first contact with the object in failed vs successful episodes).

## Units
1 step = 1 control step = 0.05 s (20 Hz). The policy outputs a chunk of 16 steps (0.8 s) and sees the cameras once per
chunk. Video frame i shows the state after step 4·(i+1). All step numbers in packets and in your output are control steps.

## Packet fields and how far to trust them
- `success`, `success_step`: trust `success`. The rollout stops in the chunk where the task first succeeds; a
  stage condition can drop again in that last chunk, so `complete_at_end` may be false in a successful episode.
- `online_stage_log` / `replay_matches_online_log`: the stage completions the rollout itself logged live (sequential,
  checked every 16 steps) and whether the replay agrees at those steps. Stages judged by the env's own accumulated
  counters (stir time, scrub contacts, wash time, rinsed directions, toaster timers) depend on contacts that a replay of
  saved states does not always reproduce; if the two disagree, trust the online log and `success`.
- `scene`: kitchen layout / style ids and object models. Group episodes by them when you look for scene-dependent
  failures (e.g. one kettle model whose handle lies along the finger-closing axis).
- `stages[k]`: `first_complete_step` and `complete_intervals` = when the **official completion condition** of stage k
  held (evaluated every step, independent of order). This is an exact measurement of the official criterion, so trust
  it as "the official condition held / did not hold". The criterion itself can be off from what you see (known: a small
  object lying on the counter within ~5 cm of a sink/cabinet region counts as "inside"; the drawer needs >= 95 % open):
  report such mismatches, do not silently override them. The official conditions also have no upright test (an object
  lying on its side can count as placed; see the tilt fields). Pick-and-place conditions include "the gripper is
  >= 25 cm away (or no longer touching)", so `first_complete_step` can lag the real placement (use the hold's
  `release_step` for when it was put down), and a stage looks undone whenever the gripper comes back near the object
  (e.g. to turn a knob next to the kettle). `undone_events` gives the reason for every drop of the condition: "objects
  still at target - only the release part broke" is not a real undo; "an object left the target" is.
  `complete_at_end=false` with an earlier interval and a real reason = the stage was **undone** (e.g. an object slid
  out after being placed). `steps_since_previous_stage` vs
  `typical_in_successes` tells you whether a stage was abnormally slow. `first_incomplete_stage` = first stage (task
  order) whose condition does not hold at the end.
- `objects[o]`: the objects moved by pick-and-place stages. `holds` = intervals where the gripper touched the object
  while it was grasped or lifted >= 2 cm and that either lifted it >= 1 cm or kept the grasp >= 3 steps (shorter
  touches are only counted in `brief_contacts_without_lift`; grasp_step .. release_step; release_step null = still
  held at the end; a negative height drop after release means it was picked up again within 2 s),
  with lift height, whether it was at the stage's target at release, distance to target at release and 2 s later,
  and the height drop after release (a large drop = it fell). `at_target_intervals`, `gripper_touch_intervals`,
  `min_eef_dist_to_centre_m` (closest approach to the object's centre: 0.15-0.2 m is normal for a handle grasp on a
  large object), `tilt_deg_*` and `tipped_over_intervals` (object axis > 45 degrees from upright).
- `gripper`: finger width every 8 steps (sum of both finger joints; ~0.08 = open, ~0 = closed on nothing) and
  `close_events`: every closing of the fingers with the width after it, `on_nothing` (touched none of the task objects = closed on air or on something else), the objects
  touched and whether they were then lifted. Grasps that never touched the object show up only here. Distances: receptacle targets = horizontal centre distance; fixture
  interiors (sink, cabinet, microwave, fridge, dishwasher) = distance from the object centre to the interior box (0 inside);
  burner = distance to the burner centre (on-burner threshold 0.15 m); coffee machine = distance to the pour site (0.04 m).
- `fixtures`: the official fixture state the stage acts on, sampled every 16 steps (`value_every_16_steps`), with
  `done_intervals`, gripper `gripper_touch_intervals` and the full state every 32 steps (`state_every_32_steps`; for the
  stove: every knob's angle plus the burner nearest the kettle / pan and its distance, and SearingMeat's task knob).
  The stove's `done_intervals` mean "some burner is on" (KettleBoiling) or "the task knob's burner is on" (SearingMeat),
  not "the burner under the kettle is on": compare the knob angles with the nearest burner yourself. Values: stove = largest knob angle from off in rad (on at
  >= 0.35; for SearingMeat only the task's knob); drawer = opening fraction (done at >= 0.95); microwave / dishwasher door
  stage = closedness (1 = closed); microwave / coffee machine on = 0/1; sink = handle joint |angle| (water on per
  `state_end.water_on`); toaster = highest lever position.
- `process_progress_end`: the task's own counters (scrub contacts/sweep, stir time, wash time, rinsed spout directions).
- `official_labeler`: RoboCasa's official subtask labeler, built for successful demos. On failures its segment
  boundaries are often meaningless (it still outputs a "place" end when an object is dropped). Context only.

## Procedure for each episode
1. Read the packet. Note the first incomplete stage, undone intervals, abnormally slow stages, and every hold/release
   and fixture change — these are the moments to look at.
2. Overview sheet: ~12 steps spread over the whole episode.
3. **Dense sheets at the events** (always for failures; for successes where something is odd): around each
   grasp/release of the relevant object, each completion/undo, and across the failed stage's window. Sparse frames
   miss things (a 12-frame overview of a 2600-step episode is one frame every 11 s — grasps and drops vanish), so do not
   conclude "never grasped" or "never touched" from an overview: check the packet's holds/touch intervals and zoom in.
4. Decide the **root-cause stage**: usually the first incomplete stage, but not when (a) an earlier stage used up the
   time (abnormally slow vs typical) and the failed stage was simply reached too late, (b) an earlier stage was undone,
   or (c) the official condition of an earlier stage was only nearly met and the robot behaved as if it was done.
5. Describe what physically happened, with step numbers, and the causal chain to the failure. Note every attempt the
   policy made to fix things by itself and whether it worked. Note anything where the packet and the video disagree.

## Categories (pick one primary for the root cause; add secondary ones if relevant; use "other: <description>" if none fits)
- `stall_pre_grasp`: hovers at/near the object without grasping.
- `stall_holding_at_target`: holds the object at/near the target and does not release (or keeps re-positioning).
- `stall_at_fixture`: at the handle/knob/button/lever but does not actuate it.
- `stall_approach_retreat`: repeatedly approaches and backs off.
- `wander_away`: leaves the task area / looks around without purpose.
- `missed_grasp`: closes on air or slips off at grasp (a real attempt: the fingers closed at the object). If the
  policy then hovers, add `stall_pre_grasp` as secondary; use `stall_pre_grasp` as primary only when it never closes
  at the object.
- `missed_grasp_continued_empty`: after a failed grasp it carries on with the stage's motion empty-handed as if it
  held the object (e.g. an empty "place" over the stove, then turning knobs).
- `dropped_in_transit`: grasped and lifted, lost before reaching the target (say where it landed).
- `placed_off_target`: released at the wrong place (say where and how far, e.g. counter in front of the machine).
- `release_timing`: was at the right place but released only after drifting off (or released too early).
- `placed_then_fell_or_slid`: released on target or at its edge, then it fell / slid / tipped out.
- `actuation_insufficient`: moved the fixture but not far enough (drawer 85 %, knob not to on, door not closed).
- `wrong_fixture_element`: actuated the wrong knob/burner/button/door.
- `wrong_object`: manipulated an object that is not the stage's object.
- `skipped_or_wrong_stage`: worked on a different stage than the current one (e.g. took the bread out before toasting).
- `undid_previous_stage`: disturbed or reversed an already completed stage.
- `disturbed_placed_object`: knocked, dragged or re-grasped an object it had just placed (e.g. fingers catch the
  handle while retreating and tip the kettle off the burner).
- `ran_out_of_time`: the failed stage was progressing normally but started too late — then the root cause is the
  earlier stage's category (e.g. `stall_pre_grasp` in stage 0) and `ran_out_of_time` is secondary.
- `official_criterion_issue`: the official condition disagrees with what the video shows (near-miss threshold,
  false positive). Secondary in most cases.

## Output (write only inside diagnosis/<Task>/)
One file per episode, `diagnosis/<Task>/<tag>.json`:
```json
{"tag": "...", "success": false,
 "failed_stage": 1, "root_cause_stage": 0, "root_cause_steps": [144, 628],
 "category": "stall_pre_grasp", "secondary": ["ran_out_of_time"],
 "what_happened": "step-referenced narrative of the whole episode, stage by stage (3-8 sentences)",
 "causal_chain": "e.g. hovered over the kettle handle 144-628 (24 s, typical stage 0 is 230 steps) -> kettle placed at 768 -> 232 steps left -> knob being turned at timeout (1000)",
 "self_recovery": "attempts the policy made on its own and their outcome, or 'none'",
 "official_criterion_notes": "mismatches between the official condition / packet and the video, or ''",
 "evidence": {"sheets": ["frames/....png"], "packet_facts": ["holds: grasp 696 release 760 ..."]},
 "confidence": "high | medium | low", "uncertain_about": ""}
```
`root_cause_steps` is a window: from the first event of the failure (e.g. the missed grasp) to the step where the
episode became unrecoverable or the stage's window ended.
For successful episodes fill `failed_stage`/`root_cause_*`/`category` with null and use `what_happened` and
`self_recovery` to record any error that was recovered from and any abnormally slow stage (with steps).

After all episodes, write `diagnosis/<Task>/_summary.md` (for the project lead): counts per root-cause stage and
category; the 2–4 recurring failure mechanisms with representative tags and steps; what successful episodes do
differently at the critical moments; official-criterion issues found; anything surprising. Be concrete and honest
about uncertainty. Do not modify any other file, do not run simulations or GPU jobs.
