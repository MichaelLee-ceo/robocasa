# The final retrace in the coordinator loop (2026-11-07)

`oracle_retrace` (oracle tracking, undone + p90 timeout detection, retrace recovery) with the final retrace
(coordinator/motion/retrace.py: the trace rewound one entry per step at the forward speed, torso / OSC goal / rest
restored) and the unstick fallback for a blocked retrace (the VLM, local Qwen3.8-27B, picks GR00T's sentence). The 43
phase 0 failure points x policy seeds 0-3, the same points, seeds and noise-stream names as loop.py's retrace arm (v2,
`../20261105_recovery/loop/retrace`), so the runs pair one to one.

Results: `oracle_retrace_SUMMARY.md` (written by coordinator/scripts/watch_batch.py when the batch was complete).

| | control (GR00T alone) | v2 retrace (loop.py) | final retrace |
|---|---|---|---|
| failed stage fixed | 20/172 | 64/172 (37 %) | 66/172 (38 %) |
| task success | 7/172 | 38/172 (22 %) | 37/172 (22 %) |

- **The same outcomes as v2** (runs that differ: 17 / 15 fixed, 15 / 16 success; per point 8 better, 7 worse). The
  final retrace restores the robot's state far more exactly (hand 0.5 mm vs 1.3 mm median at the restart, torso and OSC
  goal restored, the rewind at the forward speed), and this does not change what GR00T then achieves: with about 17 %
  of re-attempts fixing the stage and 1-2 re-attempts fitting in the time left, the exactness of the restart is not
  what limits the result. By task: SteamInMicrowave fixed 30 -> 35, success 23 -> 26; KettleBoiling 12 -> 10 both;
  PackIdenticalLunches success 3 -> 1.
- Retraces: 402, 391 arrived (97 %); 8 blocked, 3 of them back after unstick ("Close the fridge door." 12 times, "Pick
  the fish from the fridge and place it in the tupperware." once); retrace length median 38 chunks, max 297.
- Run time: median 756 s, max 2015 s.

How it ran (incidents):
- First launch: 39 workers all rendering on GPU1; 79 runs failed at once with "Offscreen framebuffer is not complete"
  (too many render contexts on one GPU). Their error results were deleted and they were re-run with at most 19 on GPU1 and 13 on GPU2.
- 13 runs started at the moment of those failures ran for 6 hours at full CPU without finishing (a replica of one of
  them, SteamInMicrowave s1 seed 0, took 4.5 minutes; they rendered on NVIDIA, not in software; the cause was not found
  since the processes could not be inspected). The watchdog stopped them (`oracle_retrace_errors.jsonl`, "timeout")
  and re-ran them; all finished normally.
- `coordinator/scripts/watch_batch.py` (setsid, independent of the session) watched the batch: error results deleted
  and re-run (at most 3 tries), a run over 45 minutes stopped and re-run, the status in `oracle_retrace_STATUS.txt`,
  the valid-only `oracle_retrace.jsonl` and the summary written at the end.
