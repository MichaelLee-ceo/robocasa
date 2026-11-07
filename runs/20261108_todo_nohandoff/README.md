# The to-do list tracker without the hand-off check, offline on recorded GR00T runs (started 2026-11-08)

The user's decision (2026-11-08): no hand-off check; the tracker only checks whether the current item is done.
The hand-off check had blocked 69 correct ticks for 13 premature ones prevented in the first batch
(`../20261107_todo_offline`), and 29 for 1 in the 13 PackIdenticalLunches runs of the second (`../20261108_todo_uniform`,
stopped at 15/90 for this change), and the hand-off conditions the VLM wrote could be false by construction
(PackIdenticalLunches s7: "the other lemon wedge and both steaks remain on the counter" while they were in the fridge,
so that run never ticked).

The tracker now (coordinator/tracking/vlm.py):
- plan: the VLM splits the instruction into items (one skill each; placements without a fixed order marked any_order,
  one object into one container per item), each with its done condition ("... and the gripper has let go of it");
- every chunk: the open items (the current one, plus the not-yet-done items of its any-order run): done yes / no each,
  the progress level, advancing, earlier results intact; 10 chunk ends sampled uniformly over the current attempt,
  three cameras (30 images);
- tick: done at 2 checks in a row; any-order items also need a grasp-and-release since the attempt began.

Smoke: KettleBoiling s0 ticks at chunk 24 (oracle 22), tracker calls only; the 10 PackIdenticalLunches plans checked
(s0, 4, 5, 7, 11, 13, 19, 22, 25, 28) all have 4 any-order items (s22 had 2 items in the second batch).

Run: `coordinator/scripts/watch_eval.py --windows uniform` (setsid, 20 workers on GPU1; the 27B on GPU3 with 24
concurrent sequences, kept up by `runs/vllm_logs/guard_vllm.sh`).
