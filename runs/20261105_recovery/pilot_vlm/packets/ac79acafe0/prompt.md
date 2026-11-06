A robot is doing a kitchen task in simulation. Task instruction: "Start the toaster. Once the lever pops up, take the bread to the plate on the dining counter."
The task has these stages, in order:
  1. Push down the lever of the toaster to turn it on.
  2. Place the toasted item on a plate.
You see 4 moments from 59.2 s to 64.0 s after the start, each from 3 cameras.

Answer with one JSON object and nothing else:
{"stages_done": [one true/false per stage, in order, judged at the LAST moment],
 "event": one of ['none', 'missed_grasp', 'dropped', 'fixture_short', 'wrong_element', 'no_progress', 'other'],
 "event_detail": "which object or fixture, and what happened (empty if none)",
 "evidence": "what you see in the frames that supports your answer",
 "confidence": a number from 0 to 1}
Event definitions (pick the one that best describes the clip; "none" if the robot is making normal progress or is done):
- missed_grasp: the gripper closed on nothing at or near the target object, and the robot moves on without it.
- dropped: an object the robot was holding fell or slipped out of the gripper.
- fixture_short: a drawer, door, lever, knob or handle was moved only part of the way, then released or left.
- wrong_element: the robot opens, pushes or turns a fixture part that is not the one the current stage needs.
- no_progress: the robot is not advancing the current stage (wandering, hovering, repeating the same motion).
- other: a failure that fits none of the above (describe it).

The frames are in panel.png next to this file: rows are moments (oldest at the top, time on the left), columns are cameras. Write your answer to answer.json in this folder.
