# runs/

## Research goal (agreed with the user, 2026-10-29)
**The frozen GR00T N1.5 can already do these tasks: its own samples contain successful trajectories.**
- **Evidence:** stage-level "imagine, then choose" (`20261026_imagine/`). The simulator is the oracle: it imagines 8 of
  the policy's own rollouts per stage and keeps one that finishes.

  | task | plain policy | imagine |
  |---|---|---|
  | KettleBoiling | 65.0 % | 96.7 % |
  | SteamInMicrowave | 30.8 % | 90.0 % |
  | PackIdenticalLunches | 11.7 % | 63.3 % |

  - The last two tasks still need a re-run with the restore fixes.
  - More samples could go higher (most PackIdenticalLunches failures hit the candidate cap).
- **Backtracking counts.** The committed path is still one forward trajectory, so these numbers prove that a solution
  exists among the policy's samples.
- **Goal.** A stage trajectory is a sequence of chunks, so the same solutions can be reached by choosing chunks. We
  want this performance by choosing among the policy's own candidate chunks ("think then leap"). We choose only at the
  chunks that decide the outcome, so we need far fewer evaluations than the stage-level search.

Objectives:
1. **Target:** the imagine success rates above (the existence proof / upper bound).
2. **Find the critical chunks:** the chunk where a stage's outcome gets decided, i.e. where V = P(finish | state, plain
   policy) jumps (`20261028_critical/`).
3. **Choose well there:**
   - first with an oracle (the simulated outcome);
   - then with a VLM critic that judges the observation and the candidates' action trajectories.
4. **Final comparison:** plain policy vs our method (oracle, then VLM) vs the imagine target, on task success and on the
   number of evaluations.

**Current step: which chunks decide success or failure, and can we find them?**
- **Premise.** From the same scenes the stage-level search finds a path that finishes, so the plain run made bad
  choices somewhere.
- **Approach.** Policy steering, as imagine does: at some decision points, pick a better chunk among the policy's own
  candidates to raise the chance of finishing the task.
- **Questions this step answers:**
  1. Which chunks decide success or failure (the critical chunks)?
  2. Can we find them, i.e. tell when to steer?

Question 1, answered (2026-10-29; `20261028_critical/README.md`):
- **Data.** The restore bugs are fixed (settled-v5). Every affected point was re-collected, checked row by row and gated
  (`20261029_review/fix/README.md`). The data is in `20261026_critic_data/data/<task>/<dataset>`.
- **Located critical chunks** (one chunk, |dV| >= 0.4, > 2 SE): KettleBoiling 5, SteamInMicrowave 6,
  PackIdenticalLunches 21 (`20261028_critical/critical_chunks.json`).
- **Choosing among 8 candidates at them raises the stage-completion chance by +25 points** (cross-validated, 95 % CI
  [+15, +36]; per task +34 / +20 / +25). At random chunks the gain is -2 [-8, +2]. The outcome is decided at a few
  chunks, and steering there works.

Next (running since 2026-10-29 14:00):
- **Closed loop with an oracle** (`20261029_closedloop/`): steer chunk by chunk in real KettleBoiling episodes, and
  compare task success with the plain policy and imagine. The 10 scenes are the ones where the plain policy often
  fails: plain 32.5 %, imagine 10 / 10.
- **Question 2** (`20261029_signals/`): which observable signal, without simulation, tells the critical chunks apart?

**Lesson from imagine's backtracking.** A choice must keep the later stages possible. For example, a kettle placed
where its knob cannot be turned finishes S0 but fails S1. So the critic's label has to look at the whole task, not only
the current stage.

---
Current work (from 2026-10-25): stage-level root-cause diagnosis of GR00T N1.5 on the 16 Composite-Seen tasks.
Everything earlier (experiments, results, launchers, the old README) is in `prev/`.

- `scripts/atomic_eval.py`: eval client. Job condition `baseline_seed` + `scene` (frozen scene) runs the official composite
  instruction and logs when every stage passes; a job field `policy_seed` makes the action noise reproducible (needs
  `serve_seeded.py`) and records per-chunk fingerprints (`digest`: observation sent, action received, sim state).
- `scripts/serve_seeded.py`: GR00T policy server = `third_party/Isaac-GR00T/scripts/run_eval.py`'s own policy +
  per-request seeding and deterministic torch settings.
- `scripts/atomic_steps.py`, `scripts/atomic_steps_seen11.py`: the stages of the 16 tasks. Stage sequence = official
  per-frame subtask annotation of the target datasets; stage checks = the official `_check_success()` split per stage.
- `scripts/validate_atomic_steps.py`: replays official demos and checks that every stage check fires.
- `scripts/{oracle,groot_eval,demo_utils,robot_home}.py`: helpers imported by the above (scene reset, client, demos).
- `scripts/make_scenes*.py` + `scenes_seen5_v2/`, `scenes_seen5_v2_s30/`, `scenes_seen11_v2/`: frozen scenes (s0-19 of the
  original 5 tasks in `scenes_seen5_v2`, s20-29 in `_s30`, the other 11 tasks in `scenes_seen11_v2`).
- `horizons_original_plus.json`: original task horizons (the 16 Composite-Seen values are the paper's).
- `20261025_determinism/`: acceptance test of the seeding (`run_test.sh`, `compare.py`, `render_check.py`).
- `_handoff/`: environment setup notes for moving the project to another machine.

Reproducibility (checked 2026-10-25, details in `20261025_determinism/`):
- Initial state: frozen scenes, fingerprint-checked on load. Physics: bitwise identical given the same state and actions.
- Policy: GR00T's action noise is seeded per chunk (`policy_seed` + `serve_seeded.py`); bitwise identical across runs and
  across GPU0 / GPU3.
- Rendering: NOT bitwise reproducible, left as is on purpose. Semi-transparent fixture glass (oven / microwave door
  material alpha 0.8) rendered with MSAA (offsamples 4) on the GPU gives +-1 on a few pixels in ~2-9% of renders;
  that is enough for a rollout to diverge (same scene + seed: one success, one failure). Making it deterministic
  (MSAA off, opaque glass, CPU rendering) would change the images the policy sees, so we keep the official rendering
  and treat outcomes as stochastic per scene (repeat runs, same-state controls).
- Scene restore: `reset_from_xml_string` leaves the python fixture poses of the discarded random scene (robocasa
  issue #209 / PR #225; on our scenes the counter appliances and accessories, 0.2-1.2 m off). `oracle.reset_to` now
  syncs them from the simulator. None of the 16 tasks' success or stage checks read those poses (the coffee machine
  check uses a simulator site, the toaster checks use contacts and joints), so earlier results are unaffected.

Envs: `conda activate $ROBOCASA_ROOT/envs/groot`; export `ROBOCASA_ROOT=/home/michaellee/mclee/robocasa`.
