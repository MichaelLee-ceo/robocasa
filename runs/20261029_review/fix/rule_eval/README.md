# Offline scoring of gripper-ramp rules (how settled-v5's gripper_ramp was chosen, 2026-10-29)
All scripts score a rule against the true ramp at 77.5k chunk boundaries of saved rollouts
(`../../restore/out/grip_table.npz`, with contact counts in `grip_geo.json`). Run with
`/home/michaellee/mclee/robocasa/envs/groot/bin/python -I <script>`.
- `eval_value.py`: g_final vs value-based hybrids with a fixed lag of 1-2 steps.
- `fit_lag.py`: the servo extrapolation `1 - (gap + k * dgap) / 40 mm`. k = 2 is best: mean error 0.01 steps at mid-ramp,
  free fingers.
- `extrap_all.py`: the extrapolation per gap band and contact class (free / one object touched / fixture / held).
- `fixture_var.py`: fixture contact with moving fingers. Extrapolation helps mid-ramp (3.84 -> 1.49 steps) but raises
  the share of >= 2-step errors at saturated boundaries (6 % -> 10 %). This is the branch flagged at
  PackIdenticalLunches s2 t2240.
- `ramp_cov.py`: rule values at the 18 boundaries of `../../restore/ramp_restore.py`, whose in-env 16-step finger and
  object errors are in `../../restore/out/ramp_restore.jsonl`.

Score any change to `hidden_state.gripper_ramp` here, and then in the env (`../validate_v5.py`, `../run_validate.sh`)
before adopting it.
