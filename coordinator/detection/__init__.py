"""Failure detection (proposal §3, role 2): cheap triggers ("something may be wrong") and a confirmer ("is it?").
base.py           Detector: check(memory, obs, status) -> [Trigger] (status may be None: no tracker answer this chunk)
                  Confirmer: confirm(memory, obs, trigger) -> (bool, reason)
timeout.py        StageTimeout: time on the current attempt over the stage's window (phase 0 p90). Works.
undone.py         Undone: an earlier sub-goal reads not done for a few checks in a row. Works.
progress.py       ProgressStall: the VLM tracker's progress dots not advancing. New (2026-11-07).
proprio.py        EmptyGrasp, Stationary: the proposal's proprio triggers. TODO.
oracle_events.py  OracleEvents: envkit's simulator events (missed grasp, drop, ...): a labelled upper bound. Works.
confirm.py        AcceptAll (works); VLMConfirmer (reference v0: the VLM judges the alarm from the memory)."""
