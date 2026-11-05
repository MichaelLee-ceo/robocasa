# Our stage checks against the official subtask labeler (2026-10-25)

> **Record only (2026-11-05).** This experiment ran on the earlier pipeline, which had known problems (`../20261104_env_audit`). Its code was removed from the working tree; it is in git at commit `84e894a`. Its large outputs (git-ignored) were moved to `~/mclee/robocasa_old_data/runs/<this folder>/`. This folder keeps the README and the results. New work uses `../envkit`.

| file | step |
|---|---|
| `check_official_labels.py` | 1. Does the official labeler (`robocasa/scripts/dataset_scripts/add_subtask_labels_to_lerobot.py`) reproduce the stored labels in our checkout? |
| `validate_stage_checks.py` | 2. Do our per-stage checks (`../scripts/atomic_steps.py` TASKS) agree with the official stage segmentation on official demos, for all 16 Composite-Seen tasks? |

The stage checks used everywhere since (critic data, imagine, closed loop) are the ones validated here. Final success is
always the official `_check_success()`.
