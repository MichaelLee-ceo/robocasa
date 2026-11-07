# The to-do list tracker, second version, offline on recorded GR00T runs (started 2026-11-08)

Changes from the first version (`../20261107_todo_offline`), all the user's decisions of 2026-11-08 after its results:
- **Window:** 10 chunk ends sampled uniformly from the start of the current attempt (the item's start; after a retry,
  the end of the rewind) to now, with the three cameras GR00T sees (30 images; the local server was restarted on GPU3
  with a 32-image limit; GPU0 was given back).
- **Hand-off:** "let go of it, no longer touching it" (the first version's plan prompt said "moved away", and the
  VLM blocked SteamInMicrowave's close-door 40 times while GR00T pressed start next to it).
- **Any order:** the plan marks placements without a fixed order (`any_order`, PackIdenticalLunches); every open item
  of such a run is judged at each check and whichever is done is ticked (memory.tick_out_of_order; the current item
  restarts its attempt there, so a retry does not undo the item). GR00T picks the placement order itself (trained on
  the whole-task sentence only), so the tracker follows it and never names an item to GR00T.
- **Grasp-and-release guard:** an any-order item is ticked only if the gripper closed and opened since the attempt
  began (the robot's own gripper commands): the VLM counted an identical object twice.
- Scoring by count (the i-th tick against the oracle's i-th stage); re-scoring the first batch this way gives the
  same numbers.

Smoke (5 runs, scratchpad): 30 images per call, 7.7 s median at 5 parallel; SteamInMicrowave s1 ticks at 17 / 45 /
60 (oracle 16 / 45 / 57); the guard blocked one double tick (PackIdenticalLunches s6). Seen there: GR00T dropped the
lime ON the fish in a tupperware (chunk 38, z 0.998 vs the floor 0.96); the VLM ticked it, the oracle (and the
official check) does not count it: a visual / official disagreement, the reverse of the tipped kettle.

Run: `coordinator/scripts/watch_eval.py --windows uniform` (setsid, 16 workers on GPU1, the 27B on GPU3).

## Stopped at 15/90 (2026-11-08 04:34, the user's decision)
All 15 finished runs are PackIdenticalLunches (the longest first). On the same 13 runs as the first batch: premature
ticks 6 (3 runs) vs 2, tick lag median 21 vs 4.5 chunks, completions never registered 10/29 vs 5/31, false stalls
0.83 vs 1.15 per run. The hand-off check blocked 29 correct ticks (median delay 30 chunks) for 1 premature one
prevented, and some hand-off conditions the VLM wrote were false by construction (s7: "the other lemon wedge and both
steaks remain on the counter"; they were in the fridge: no tick in the whole run); s22's plan had 2 items. The user
dropped the hand-off check; the evaluation continues without it in `../20261108_todo_nohandoff`.
