# Stage-level "imagine, then choose" (oracle upper bound), KettleBoiling, 2026-10-26

> **Record only (2026-11-05).** This experiment ran on the earlier pipeline, which had known problems (`../20261104_env_audit`). Its code was removed from the working tree; it is in git at commit `84e894a`. Its large outputs (git-ignored) were moved to `~/mclee/robocasa_old_data/runs/<this folder>/`. This folder keeps the README and the results. New work uses `../envkit`.

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
`analyze.py`. Results in `full/` (`results_w*.jsonl`). The candidate videos and the two smoke tests (`smoke/`, `smoke2/`) were
deleted on 2026-11-04, when `full_v2` and `sim_pil` had superseded these runs; only the report's three showcase scenes
keep their candidate videos (`full_v2/videos/KettleBoiling_s10_single`, `sim_pil/videos/SteamInMicrowave_s12_single`,
`sim_pil/videos/PackIdenticalLunches_s26_single`).

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
- Backtracking changed the outcome in s4 here and in s26 in the smoke test.
  - In the smoke test, the earliest S0 finisher put the kettle on a burner where none of the 4 S1 candidates turned the knob.
  - s12's `backtracked` flag is not a real backtrack. The search preferred a candidate that finished both stages at once.
- Checked the final frames of the hard scenes (s2, s5, s8, s9, s10, s24): the kettle is upright on its own burner and that
  burner is lit, so these are not criterion artefacts.
  - The official check accepts any lit burner within 0.15 m of the kettle (`kettle_boiling.py:77`).
- This is an upper bound:
  - perfect world model and perfect evaluator;
  - execution equals imagination;
  - imagining costs no episode time.

## Re-run of the two 4-stage tasks (`sim_pil/`, launched 2026-10-30 00:49)
The `multi/` imagine numbers below (90.0 % / 63.3 %) are superseded. That run predates every restore fix, starting with
the controller-state restore of 2026-10-27. Its plain baselines (condition baseline, seeds 1-3) stay valid: plain
rollouts do not restore.

The re-run uses the same 30 jobs per task (`jobs_sim_pil.json`: N = 8, keep 3, at most 64 candidates, the same L and
horizons), launched by `run_on_servers.sh` on the GPU0 / GPU3 servers. `imagine.py` changes:
- Nodes reached by stepping keep their last observation and python fixture state (`hidden_state.capture(kenv,
  obs=True)`), and `atomic_eval.restore_node` makes the restore exact. This was checked bit for bit in
  `../20261029_closedloop/preflight.py`.
- Success means the official check held at some step, as for the plain baseline.
  - The old rule also required the official check after restoring the end state.
  - That counted 4 searches as failures although a committed candidate had succeeded (SIM s5, s22; PIL s7, s24).
  - `final_check` is still logged, as a diagnostic.
- A finisher is a candidate whose stage index advanced (or success), as in the closed loop and the critic data.
  - The old rule also required every earlier stage check to still hold at the candidate's end.
  - On SIM it rejected about 20 % of the stage-2 finishers, one of them with official success, mostly through the
    stage-1 release condition (gripper 25 cm from the bowl) while the gripper pushes the door.
  - `prev_ok` is still logged.

**Result (finished 2026-10-30 08:10, 60 / 60 jobs, no errors):**

| task | plain policy, 4 seeds per scene | stage-level search | policy calls per scene |
|---|---|---|---|
| SteamInMicrowave | 30.8 % (37 / 120) | **100 % (30 / 30)** | 720 |
| PackIdenticalLunches | 11.7 % (mean over scenes; 14 / 119 rollouts) | **70.0 % (21 / 30)** | 1292 |

- **SteamInMicrowave.** Every success comes from a committed candidate whose official check held during its rollout.
  - The diagnostic `final_check` holds in 29 of 30. The other is the restore artefact of the microwave flag.
  - 26 searches needed only the greedy 4 x 8 candidates, and 4 backtracked.
- **PackIdenticalLunches.** All 9 failures used the full 64-candidate cap, 8 of them in stage 3 (s1, s7, s8, s14, s16,
  s23, s25, s27, s29). So the target is limited by the budget, not by the policy.

## Two 4-stage tasks (`multi/`, finished 2026-10-26 12:29; imagine part superseded, see above)
Same search (N = 8, keep 3, at most 64 candidates per episode; L = SteamInMicrowave [432, 720, 576, 416], PackIdenticalLunches
[416, 576, 608, 496]); python-side fixture state (e.g. microwave._turned_on) is saved and restored with every node.
Baseline = plain policy with policy_seed 1-3 (here) plus seed 0 (runs/20261025_seen480). One baseline is missing:
PackIdenticalLunches_s20 seed 1 (the scene failed to build, `could not initialize task`).
The run was stopped once and resumed; the last 22 baselines were re-split over 15 workers.

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
