# Coordinator runs and the retrace checks (2026-11-06/07)

Code: `coordinator/` (top level). This folder holds records only.

| file / folder | what |
|---|---|
| `oracle_retrace/`, `oracle_retrace.jsonl`, `*.log` | coordinator `oracle_retrace` runs with the earlier (v2) retrace: p0-3 (172, before the 4 rare-path fixes, partly other noise-stream names), p4-7 (24, matched to loop.py) |
| `skeleton_v0_oracle_retrace/` | the first coordinator skeleton's runs (archived) |
| `equivalence*.jsonl` | the coordinator proven equal to loop.py (check_equivalence.py, kept in git at 97a6269) |
| `retrace_state_final.jsonl` | `coordinator/tests/check_retrace_state.py` with the final retrace: the first retrace at each of the 47 usable phase 0 failure points |
| `retrace_state_v2.jsonl`, `_v3`, `_vC`, `_final_skip` | the same check with the versions it replaced (table below) |
| `retrace_fallback/` | `coordinator/tests/check_retrace_fallback.py`: fallbacks where the retrace is blocked (PackIdenticalLunches s4), frames |
| `unstick_test/`, `unstick_final/` | s4 in the coordinator loop with the VLM picking the unstick sentence (Qwen3.8-27B, local): version C, and the final version |
| `vlm_check/` | `coordinator/scripts/vlm_check.py` outputs |

## The retrace (coordinator/motion/retrace.py)
What GR00T moves, and what the rewind restores (an audit of every robot quantity that changes from step to step):

| GR00T's action | what it moves | the trace records and the retrace restores |
|---|---|---|
| end-effector position / rotation (6, deltas, through the OSC) | the arm's 7 joints | the joints and their speeds at every step; at the end the OSC goal's offset from the hand |
| gripper_close | the fingers (through the command ramp) | the command at every step: it opens where it had closed |
| base_motion x, y, yaw | the base | the base pose at every step |
| base_motion torso | the torso | the torso height at every step (the first trace missed it; its controller's goal follows any push) |
| control_mode | how the OSC goal is updated | the OSC goal's offset at the end |

The final version: the trace is a stack popped one entry per step ("怎麼去就怎麼回來"); the arm follows the popped joints
with the recorded speeds reversed as the speed reference; the base gets the command that takes it to the popped pose;
the playback waits only when the robot lags far (arm 0.25 rad, base 15 cm or 0.3 rad), and a lag that does not shrink
over 64 steps of waiting is "blocked"; on arrival the arm is held until at rest and the OSC goal is restored.

| 47 points | v2 (loop.py's) | v3 | C | **final** |
|---|---|---|---|---|
| back at the stage start | 44/45 | 44/47 | 46/47 | 45/47 |
| duration / forward duration, median (p90) | | | 1.02 (1.41), half of it waiting for the arm | **1.08 (1.17)**, waiting 2 % |
| hand off the recorded path, median | 0.42 cm | 0.31 | 0.33 | **0.18** |
| arm lag behind its entry, max (median over points) | 0.11 rad | 0.07 | 0.09 | **0.03** |
| restart: hand vs the stage start, median / p90 | 1.3 / 7.7 mm | 0.5 / 1.0 | 0.6 / 1.6 | **0.53 / 1.31** |
| restart: torso, max | 5.8 cm | 0.1 mm | 0.1 mm | 0.1 mm |
| restart: arm speed, p90 | 0.11 rad/s | 0.005 | 0.014 | 0.003 |
| restart: OSC goal offset error | 2.9 cm | 0 | 0 | 0 |

What changed and why:
- v3 / C restored the torso, the OSC goal and rest (C: looser tracking so that contact-pressed arms do not stall).
- C's plan skipped "standing still" and "loops", but its loop rule also jumped ahead on any stretch within 0.05 rad,
  so half the entries were skipped and the arm, asked to move twice as fast, lagged and waited as long. The final plan
  pops every entry; with the speed reference the arm keeps up (lag about 0.03 rad).
- The base had a 3 cm / step cap below what GR00T drives (4.6 cm / step): it fell up to 42 cm behind
  (PackIdenticalLunches s27). The final base command has no cap but the command range.
- Tried and dropped: feeding back the recorded arm torques (worse tracking), an integral term (oscillation).

Not back at the start (final):
- **SteamInMicrowave s8** (15 mm, joint 0.107 rad off): the reversed motion pulls the microwave door open 0.22 rad by a
  handle the forward motion had pushed against; the finger then presses on the handle. A rewind meets contacts the
  other way round.
- **PackIdenticalLunches s4**: blocked (below).

Not reproducible by commands: the gripper's command ramp at the episode start (0 after reset; gripper_close is
binary): fingers about 19.5 mm apart from the reset state at stage-0 restarts.

Objects (the final run, `objects` in each line): 170 objects that did not move in the stage: none moved 2 cm or more by
the retrace. 4 were held at the failure point: 2 put down where they were last picked up (s23 1.8 cm, s27 9 cm from
the stage start, having slid once before), 2 slipped out on the way back (s15 meat, s14 bowl) because the grasp was
already failing at the failure point (the finger gap was shrinking; in the plain run GR00T lost them within 7 and 15
chunks). Of 26 objects let go during the stage, 9 were picked up again by the retrace where they had been put down
and carried back (0.6-8.7 cm from where they were picked up); the others fell or were pushed and stay.

## When the retrace is blocked: unstick (PackIdenticalLunches s4)
GR00T had pushed the arm into the open fridge (contact at 1533 of 2145 steps of the stage, joint 1 at its torque
limit). Fallback (coordinator/recovery/unstick.py, the user's idea): GR00T gets one sentence for 8 chunks, the VLM
choosing it in GR00T's own wording; then the retrace resumes from the last entry at which the hand was 0.25 m clear.

| | blocked at | fallback | back at the stage start |
|---|---|---|---|
| by hand (retrace C) | entry 1309 | none / base backs up / task sentence / "Close the fridge door." | 0/1, 0/1, 1/4, **3/4** |
| VLM (27B) in the loop, retrace C (`unstick_test/`) | entry 1309 | the VLM wrote "Close the fridge door." | **4/4** (one needed two tries) |
| VLM (27B) in the loop, final retrace (`unstick_final/`) | entry 657 | the VLM wrote "Close the fridge door." twice | 0/4 |

Asked directly at the block (retrace C), the 27B wrote "Close the fridge door." with or without the wording rule;
Claude Opus wrote "Pick the steak from the fridge and place it in the tupperware on the nearby counter." (a next
step that carries the arm out; not tested). With the final retrace the robot gets further back and is blocked near
the stage start, where the path back runs into the fridge again; resuming 0.25 m away re-enters it and is blocked at
once. Open (the user, 2026-11-07: leave it for now): choosing where to resume.
