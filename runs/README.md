# runs/

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
