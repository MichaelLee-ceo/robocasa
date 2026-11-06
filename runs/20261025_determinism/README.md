# Determinism of the seeded evaluation (2026-10-25)

> **Record only (2026-11-05).** This experiment ran on the earlier pipeline, which had known problems (`../20261104_env_audit`). Its code was removed from the working tree; it is in git at commit `84e894a`. Its large outputs (git-ignored) were moved to `~/mclee/robocasa_old_data/runs/<this folder>/`. This folder keeps the README and the results. New work uses `../../envkit`.

Can an episode be reproduced exactly? Checks for `../../envkit/serve_seeded.py` (seeded action noise) and
`../scripts/atomic_eval.py` (`policy_seed`). The conclusions are summarised in `../README.md` (Reproducibility).

| file | question |
|---|---|
| `run_test.sh`, `compare.py`, `R1_gpu0` ... `R5_gpu0_fixsync` (fingerprints in `results_w*.jsonl`) | The acceptance test: the same 2 frozen-scene episodes with a fresh seeded server per run. Do the per-chunk fingerprints (observation, action, sim state) match across runs, GPUs and render settings? |
| `render_check.py`, `run_render_check.sh`, `render_check/` | Is rendering the same sim state twice reproducible, alone and with concurrent runs? |
| `render_probe.py`, `render_probe/` | Which pixels change between renders of the same state, and on which geoms? |
| `transparent_survey.py` | Which visible geoms are semi-transparent in the frozen scenes? |
| `fixture_pose_check.py`, `fixture_pose_check.json` | After loading a frozen scene, do the python fixtures still describe the discarded random scene (robocasa issue #209)? |

PackIdenticalLunches rendering in particular is not reproducible: the same state gives 2-8 distinct images in 20
renders. Re-querying the policy along a saved path therefore diverges
(`../20261029_closedloop/README.md`), and the closed loop executes recorded actions instead.
