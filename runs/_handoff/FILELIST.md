# Files to put in the project repo (sizes on the old machine)

| path | size | note |
|---|---|---|
| runs/scripts | 696K | |
| runs/scripts/sched | 192K | |
| runs/scripts/scratch | 200K | |
| runs/scenes_seen5_v2 | 22M | |
| runs/_handoff | 32K | |
| runs/atomic_horizons_original.json | 4.0K | |
| runs/composite_seen_instructions.json | 8.0K | |
| runs/horizons_original_plus.json | 4.0K | |
| runs/subtask_audit.json | 8.0K | |
| runs/llm_decomposition | 20K | |
| runs/20261012_atomic_decomp_seen/{results_w*.jsonl, *.json, */stats.json} | 788K | 11 files |
| runs/20261012_atomic_repeat_test/{results_w*.jsonl, *.json, */stats.json} | 28K | 3 files |
| runs/20261012_atomic_seeded/{results_w*.jsonl, *.json, */stats.json} | 676K | 18 files |
| runs/20261012_groot_oracle/{results_w*.jsonl, *.json, */stats.json} | 4.3M | 18 files |
| runs/20261012_groot_repro_h1x/{results_w*.jsonl, *.json, */stats.json} | 508K | 17 files |
| runs/20261012_groot_repro_seen_h1x/{results_w*.jsonl, *.json, */stats.json} | 396K | 10 files |
| runs/20261013_accept/{results_w*.jsonl, *.json, */stats.json} | 80K | 7 files |
| runs/20261013_atomic_official/{results_w*.jsonl, *.json, */stats.json} | 44K | 11 files |
| runs/20261013_load_test/{results_w*.jsonl, *.json, */stats.json} | 28K | 7 files |
| runs/20261013_pretrain3/{results_w*.jsonl, *.json, */stats.json} | 560K | 12 files |
| runs/20261013_scenes_main/{results_w*.jsonl, *.json, */stats.json} | 308K | 8 files |
| runs/20261013_scenes_v2/{results_w*.jsonl, *.json, */stats.json} | 1.6M | 13 files |

Not included (copy with rsync if wanted): videos (*.mp4, ~538M), end/step states (*.npz under experiment dirs), report/ and report_v2/ page builds, scenes_seen5/ (v1 scene files with the wrong xml — do not use).

Notes:
* Also include the official-eval stats one level deeper: runs/20261012_atomic_seen_sanity/evals/target/*/stats.json and
  runs/20261013_pretrain_sanity/evals/target/*/stats.json (a few KB).
* runs/20261013_scenes_main/ was run on the v1 scene files (MuJoCo-resaved xml, objects ~10x heavier): results are
  invalid, keep only for the record. runs/20261013_load_test/ is the A/B test that found it.
* runs/20261012_atomic_seeded/ fresh-scene pairing is incomplete (reset(seed) is not reproducible); see thread notes.

Committed on `ece780` in addition to the table above:
* frozen scenes for all 8 tasks in runs/scenes_seen5_v2 (the 5 composite_seen tasks and ArrangeVegetables,
  GatherVegetables, CondimentCollection; 160 files, 22 MB);
* the hand-off start states used by the "from the correct start" runs: runs/20261013_scenes_v2/*/*_single_states.npz
  (416 KB) and runs/20261013_pretrain3/*/*_single_states.npz (240 KB, used by the 600-step reruns);
* small notes: runs/README.md, runs/20261012_groot_oracle/{README.md, instructions_demo0.md, mismatch_and_videos.md,
  tracker_validation_a/b.jsonl}, analysis_stdout.txt in three experiment dirs;
* runs/.gitignore (keeps *.mp4, images, logs, other *.npz, report builds and scenes_seen5/ out of git).

Left for rsync: per-episode metadata under runs/2026101*_groot_repro*/<Task>/ (*.json, *.gz, ~33 MB), test dirs
(runs/test*, runs/smoke_20261012), the _superseded*/_failed* attempt dirs, and the flash-attn wheel
(~/robocasa_handoff/wheels/, 256 MB, sha256 ce91e246...b529).
