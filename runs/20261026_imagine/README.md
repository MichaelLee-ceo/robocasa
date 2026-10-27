# Stage-level "imagine, then choose" (oracle upper bound), KettleBoiling, 2026-10-26

> **Caveat (2026-10-27).** All runs below restored node states with `set_state()` only. That does not restore the robot
> controllers' internal state: the gripper command ramp, the base-mode arm goal, cached kinematics, and the warm start.
> So every imagined candidate started with whatever the previous candidate left in the controllers, and "execution
> equals imagination" did not hold exactly.
>
> `imagine.py` now carries each node's controller state, `hidden_state.capture()`, from the rollout that produced it.
> See `../20261026_critic_data/README.md` and `check_restore.py`.
>
> **KettleBoiling, re-run with the fix (`full_v2/`, 2026-10-27, search jobs only):**
> - 29/30 = 96.7 %, and 27/30 = 90.0 % without backtracking, both the same as before.
> - No scene changed outcome. The only failure is s3 again.
> - Policy calls: 8,253 vs 8,333.
>
> The two 4-stage tasks below have not been re-run yet, so their numbers stay provisional.

`imagine.py` (method in its docstring), `run.sh` (GPU1, approved by the user while GPU0/GPU3 were taken),
`analyze.py`. Results in `full/` (`results_w*.jsonl`; videos of every imagined candidate in `full/videos/<tag>/e<expansion>_c<candidate>.mp4`).
`smoke/` and `smoke2/` are the two smoke tests (greedy stage choice, then the search).

Setup: 30 frozen KettleBoiling scenes, horizon 1000 steps.
- Search: N = 8 candidates per expansion; L = 416 (S0) / 272 (S1) steps; at most 3 finishers kept per expansion; at most 40
  candidates per episode.
- Imagination: the simulator, restored to the node's state. Each candidate is the frozen policy running closed-loop with its
  own action noise until the stage's official condition holds, or until L steps pass.
- Commit: set the env to the chosen path's end state. Imagining is not counted in the episode's 1000 steps.
- Baseline: the plain policy with policy_seed 0 (from `runs/20261025_seen480`) plus seeds 1-3 (here).

| method | success | policy calls (1 per 16-step chunk) |
|---|---|---|
| plain policy, mean of 4 seeds per scene | 65.0 % | ~41 per rollout |
| episode-level best-of-4 (any of the 4 seeds succeeds) | 90.0 % (27/30) | 4948 total, 165 per scene |
| stage-level search | **96.7 % (29/30)** | 8333 total, 278 per scene |
| search, path without backtracking (greedy) | ≥ 90.0 % | |

- The search solved s5 and s10. The plain policy went 0/4 on both, so episode-level best-of-4 cannot solve them.
- Only s3 failed:
  - The handle lies along the closing axis, and 0/8 candidates grasped it in the first 416 steps.
  - S0 was first finished at step 848, which left 152 steps for S1.
- Backtracking changed the outcome in s4 here and in s26 in smoke2.
  - In smoke2, the earliest S0 finisher put the kettle on a burner where none of the 4 S1 candidates turned the knob.
  - s12's `backtracked` flag is not a real backtrack. The search preferred a candidate that finished both stages at once.
- Checked the final frames of the hard scenes (s2, s5, s8, s9, s10, s24): the kettle is upright on its own burner and that
  burner is lit, so these are not criterion artefacts.
  - The official check accepts any lit burner within 0.15 m of the kettle (`kettle_boiling.py:77`).
- This is an upper bound:
  - perfect world model and perfect evaluator;
  - execution equals imagination;
  - imagining costs no episode time.

## Two 4-stage tasks (`multi/`, finished 2026-10-26 12:29)
Same search (N = 8, keep 3, at most 64 candidates per episode; L = SteamInMicrowave [432, 720, 576, 416], PackIdenticalLunches
[416, 576, 608, 496]); python-side fixture state (e.g. microwave._turned_on) is saved and restored with every node.
Baseline = plain policy with policy_seed 1-3 (here) plus seed 0 (runs/20261025_seen480). One baseline is missing:
PackIdenticalLunches_s20 seed 1 (the scene failed to build, `could not initialize task`).
The run was stopped once and resumed (`multi_resume.out`); the last 22 baselines were re-split over 15 workers.

| method | SteamInMicrowave | PackIdenticalLunches |
|---|---|---|
| plain policy, mean of 4 seeds per scene | 30.8 % | 11.7 % |
| episode-level best-of-4 (any seed succeeds) | 76.7 % | 26.7 % |
| stage-level search | **90.0 % (27/30)** | **63.3 % (19/30)** |
| search, path without backtracking (greedy) | 70.0 % | 43.3 % |
| policy calls, search / best-of-4 | 22230 / 10087 | 39789 / 18461 |

- Backtracking adds about 20 points on both 4-stage tasks, vs about 7 on KettleBoiling.
- Why the failed searches failed:
  - SteamInMicrowave (3): 1 hit the 64-candidate cap. In the other 2, every stage check passed, but the official
    success check failed at the end.
  - PackIdenticalLunches (11): 7 hit the cap, 2 ran out of horizon, and 2 are again "all stage checks passed, official
    check failed".
  - So most failures are budget, plus a few disagreements between the stage checks and the official check.
- The search makes about 8.5x the policy calls of one plain rollout.
