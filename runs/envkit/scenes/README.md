# envkit/scenes: the frozen scenes (2026-11-05)

One scene per (task, seed): the 16 Composite-Seen tasks x seeds 0-29, made by `../tests/campaign.sh`
(`run_tests.py freeze`, which calls `envkit.freeze`).

- `<task>_s<seed>.pkl.gz` (not in git, 1-2 MB each): the state right after RoboCasa's official reset (v1.0.1,
  `split="target"`, seeded as envkit seeds it): the pre-compile MJCF and episode metadata, the full MjData, and every
  value of the Python object graph by attribute path. `envkit.Episode(task, seed)` builds a fresh env and loads it; a
  model-hash mismatch raises.
- `<task>_s<seed>.ref.json` (in git): the reference trajectory, the state / proprio / image hashes after each of 8
  seeded random chunks from the scene. `run_tests.py check` replays against it.

The reset is reproducible with envkit's two fixes (see `../../20261104_env_audit`), so a lost `.pkl.gz` can be made again
with `run_tests.py freeze <task> <seed>`. That rewrites the `.ref.json`: its proprio hashes must equal the ones in
git (`git diff`). The state hashes also cover a temporary xml path in the metadata, and the image hashes can change by
render noise.
