# Review of the critic data (2026-10-29)

> **Record only (2026-11-05).** This experiment ran on the earlier pipeline, which had known problems (`../20261104_env_audit`). Its code was removed from the working tree; it is in git at commit `84e894a`. Its large outputs (git-ignored) were moved to `~/mclee/robocasa_old_data/runs/<this folder>/`. This folder keeps the README and the results. New work uses `../../envkit`.

The restore review that found and fixed the hidden-state bugs. It ends with the settled-v5 restore and the
re-collection of every affected point. Each subfolder has its own report.
The exploratory reviews (`integrity/`, `labels/`, `anomaly/`, `analysis/`) were deleted on 2026-11-04; their reports
are in git history (commit 3fbfb89), and `fix/README.md` summarises what they found.

| folder | question | report |
|---|---|---|
| `restore/` | Continuous run vs set_state + settle at chunk boundaries: which hidden state does a restore miss? | `restore/report.md` |
| `fix/` | The settled-v5 restore: the fixes, their validation, the re-collection and the selection of the rows in use | `fix/README.md` |
| `gate_v5/` | The anomaly scan repeated on the settled-v5 data, as the gate before using it | `gate_v5/tables.md` |

- The bugs and their fixes are summarised in `../20261026_critic_data/README.md` (History) and in the memory note on
  restoring hidden controller state.
- Many scripts here read the `run_*` dirs from before the 2026-10-29 reorganization. They document what was run; they
  do not re-run on the current layout (`fix/README.md`).
