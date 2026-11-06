# Composite-Seen rerun: the plain policy on 16 tasks x 30 frozen scenes (2026-10-25)

> **Record only (2026-11-05).** This experiment ran on the earlier pipeline, which had known problems (`../20261104_env_audit`). Its code was removed from the working tree; it is in git at commit `84e894a`. Its large outputs (git-ignored) were moved to `~/mclee/robocasa_old_data/runs/<this folder>/`. This folder keeps the README and the results. New work uses `../../envkit`.

The plain baseline of the whole project. The frozen GR00T N1.5 (composite-seen checkpoint-60000) runs every
Composite-Seen task on 30 frozen scenes with the official instruction and original horizons. The action noise is
seeded (`policy_seed`), and every control step's sim state is saved, so each episode can be replayed and diagnosed
stage by stage.
- The seed-0 episodes are the "main runs" that `../20261026_critic_data` branches from.
- KettleBoiling, SteamInMicrowave and PackIdenticalLunches have extra plain seeds in `../20261026_imagine/{full,multi}`
  (condition baseline).

| file / folder | what |
|---|---|
| `make_jobs.py`, `jobs.json` | the job list (16 tasks x 30 scenes) |
| `run.sh` | the run: seeded policy servers on GPU0 / GPU3, workers render on their server's GPU |
| `run/` | results (`results_w*.jsonl`), per-step states (`*_steps.npz`) and videos; only the jsonl is in git |
| `replay_check.py` | a saved episode replays to the same stage verdicts |
| `analyze_stages.py`, `after_rerun.sh` | stage-level root cause of every episode (`../scripts/stage_trace.py` + the official labeler) -> `analysis/` |
| `make_packets.py`, `frames.py` | per-episode packets and contact sheets for the video-based diagnosis -> `packets/`, `diagnosis/` |
| `report/` | the published report (`index.html`; `build_*.py` write its data files); `report_pub/` is its build output. Published at https://claude.ai/artifact/8mpnb3XU5wE39eDZfh5RK5 (private). On 2026-11-04 its Data tab and the upper-bound tab's noise-probe section were removed: both showed data from before the restore fixes |

Only code, the result jsonl and the report sources are in git. Videos, states and the diagnosis images stay on the
machine (`../_handoff/README.md` tells how to copy them).
