# The coordinator without the official time limit (2026-11-07)

The user's decision (2026-11-07): the method retries, so it needs more time than GR00T alone; what matters is the
success rate; GR00T alone fails by getting stuck, not by running out of time (in the phase 0 plain runs, 38 of 39
successes came before 75 % of the horizon, and failed runs spent a median 62-72 % of the horizon with no stage done),
so it keeps its official-horizon result as the baseline and only the method runs without the limit.

Setup: `oracle_retrace` (oracle tracking, undone + p90 timeout detection, the final retrace, the VLM unstick with the
local Qwen3.8-27B), the 43 phase 0 failure points x policy seeds 0-3 (the same points, seeds and noise streams as
`../20261107_final_retrace` and loop.py's v2 runs). `run.py --horizon-scale 5 --stuck-windows 4`: a run ends at the
official success (RoboCasa's check, unchanged), or when no sub-goal has been completed for 4 of the current
sub-goal's p90 windows of GR00T time (retraces not counted), or at 5x the official horizon (a safety cap, never
reached). The success time is recorded, so the official-horizon result of the same runs follows.

| 172 runs (paired) | fixed (the failed stage) | task success |
|---|---|---|
| GR00T alone, official horizon (loop.py control arm) | 20 (12 %) | 7 (4 %) |
| v2 retrace, official horizon (loop.py) | 64 (37 %) | 38 (22 %) |
| final retrace, official horizon (`../20261107_final_retrace`) | 66 (38 %) | 37 (22 %) |
| **final retrace, no time limit (this batch)** | **80 (47 %)** | **48 (28 %)** |

- Against the final retrace within the official horizon: success only here 15 runs, only there 4 (sign test p 0.02);
  against v2: fixed 27 / 11 (p 0.01), success 21 / 11 (p 0.11).
- Success within 1x / 1.5x / 2x / 3x the official horizon (this batch): 36 / 46 / 48 / 48. All the gain comes before 2x;
  124 runs ended "stuck", 48 at success, none at the cap. GR00T time used: median 1.4x the official horizon.
- By task (fixed / success, official horizon -> no limit): KettleBoiling 10 / 10 -> 18 / 17 (28 runs);
  SteamInMicrowave 35 / 26 -> 35 / 29 (68); PackIdenticalLunches 21 / 1 -> 27 / 2 (76): its failed stage gets fixed
  more often, but the later stages rarely all get done.
- Retraces: 605, 565 arrived; 42 blocked, 13 of them back after unstick. The VLM's unstick sentences vary with the
  scene: "Close the fridge door." (39), "Pick the steak from the counter and place it in the tupperware." (6), and other
  pick-and-place steps; 5 retraces ran to the 400-chunk cap.
- Run time: median 921 s, max 7464 s (PackIdenticalLunches s4: repeated retrace -> blocked -> unstick).
- The watchdog (coordinator/scripts/watch_batch.py, 36 workers: 20 on GPU1, 16 on GPU2) ran it with no errors;
  `oracle_retrace_SUMMARY.md` is its report.
