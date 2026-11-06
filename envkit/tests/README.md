# envkit/tests: what must hold before an experiment (2026-11-04/05)

The tests that gate the environment layer (`../envkit.py`, `../stages.py`, `../policy.py`; usage: `../README.md`).
They were written after the audit in `../../runs/20261104_env_audit`, which found that the earlier pipeline's restores,
replays and stage checks had gaps. Every new experiment builds on envkit only after these pass on its tasks.

## The tests (`run_tests.py`; its docstring has the details)
| command | what it checks |
|---|---|
| `freeze <task> <seed>` | An official reset in a fresh process, frozen to `../scenes/<task>_s<seed>.pkl.gz`. Then 8 chunks of seeded random actions; their state, proprio and image hashes are the scene's reference trajectory (`.ref.json`). |
| `check <task> <seed>` | In another fresh process: the official reset reproduces the frozen state (reported); the loaded scene equals the frozen one (MjData and the whole Python graph); a replay gives the reference trajectory; restore round trip, continuation and continuation after a branch; a restore onto the identical state changes nothing. `gc.collect()` after every chunk. Images: render noise is reported, a 32x32 thumbnail difference above 0.5 grey levels fails. |
| `stages <task> <seed>` | The stage tracker runs and is pure (the full state digest, images included, is unchanged by a call). |
| `sampling <task> <seed>` | Policy sampling: the same observation and seed give the identical chunk (twice, and on the other server); another seed gives another chunk; after a step and a restore the same seed gives the identical chunk again. |
| `policy <task> <seed>` | GR00T (seeded server) to the horizon with the tracker after every chunk. The official success implies every stage; a replay of the recorded actions reproduces success, tracker and proprio observations at every chunk and the final state; restore A/B (round trip, continuation, after a branch) at the first chunk of every new stage and at the middle. |

## Launchers
- `campaign.sh <gpu> <jobs>`: freeze and check for 16 tasks x seeds 0-29 -> `results/freeze.jsonl`, `results/check.jsonl`.
- `recheck.sh <gpu> <jobs>`: check again for the scenes in `results/recheck_list.txt` -> `results/check_rerun.jsonl`.
- The GR00T servers for `policy`: `../servers.sh <gpu> <port0> [n]` (`../serve_seeded.py`, checkpoint-60000).
- `policy_tests.sh <gpu> <jobs> <seed0> <seed1> "<ports>"`: the policy test for 16 tasks x the seeds ->
  `results/policy.jsonl`, records in `results/policy/` (not in git).
- `policy_rerun.sh <gpu> <jobs>`: the policy test again for the lines of `results/policy_rerun_list.txt` ->
  `results/policy_rerun.jsonl`.
- `summary.py`: the summary below, from `results/` (re-run rows supersede).

Start them detached (`setsid nohup ... &`). GPU rendering needs `CUDA_DEVICE_ORDER=PCI_BUS_ID`; the launchers set it.

## Results (2026-11-05, GPU1, the final code)
| test | scenes | result |
|---|---|---|
| `freeze` | 16 tasks x seeds 0-29 | 480 / 480 ok |
| `check` | the same 480 | 480 / 480 ok. The official reset reproduces the frozen state in 480 / 480 (only the two metadata keys differ). Render noise in 109 scenes, largest thumbnail difference 0.25 (fail above 0.5). |
| `stages` | 16 tasks x seeds 0-2 | 48 / 48 pure |
| `sampling` | 16 tasks x seed 0 | 16 / 16 ok (another seed changes the chunk by up to 0.17-1.0) |
| `policy` | 16 tasks x seeds 0-2 | 48 / 48 ok: 6,966 chunks replayed exactly (success, tracker, proprio), equal final states, restore A/B at 133 points (first chunk of every new stage and the middle), 0 stage gaps |

The first 102 `check` rows ran before the reset comparison left out the two metadata keys, and kept only its first 6
differing keys; they were re-run (`recheck.sh`, `check_rerun.jsonl`). Six `policy` runs of the latched-success tasks
(WashLettuce, RinseSinkBasin, StirVegetables, seeds 0-1) ran before the tracker's latch rule and were re-run
(`policy_rerun.sh`, `policy_rerun.jsonl`).

GR00T successes in these runs (official chunk-end success; 3 scenes per task, so only a smoke check of the policy, not
a success rate): 3 / 3 RinseSinkBasin, PrepareCoffee, StackBowlsCabinet, PreSoakPan, SetUpCuttingStation; 2 / 3
ScrubCuttingBoard, KettleBoiling, WashLettuce, LoadDishwasher; 1 / 3 SteamInMicrowave, StoreLeftoversInBowl,
StirVegetables; 0 / 3 DeliverStraw, GetToastedBread, SearingMeat, PackIdenticalLunches. In DeliverStraw, GetToastedBread
and SearingMeat no seed finishes the first stage, as in the earlier baseline's seeds 0-2 (`../../runs/20261025_seen480`:
2 / 30, 0 / 30, 9 / 30 over 30 scenes).

What is exact and what is not:
- Exact (bit for bit): the loaded scene, the physics and every check given the actions, replays of recorded actions in a
  new process, restores (round trip, continuation, after a branch), the official reset.
- Not exact: camera images. The official rendering (MSAA 4) changes a few edge pixels by +-1 at random on any render,
  even in one process (the same state rendered 300 times gives up to 12 distinct images; MSAA 0 gives one). A live
  policy run, also after a restore, can therefore take another path, like extra policy noise: StirVegetables s0
  succeeded at step 1040 in one run and 1104 in another. Success is judged on the physics, so the official rendering is
  kept (user, 2026-11-05); an exact rerun is a replay of the recorded actions.

Problems found and fixed while running these tests (details in `../../runs/20261104_env_audit/README.md`): render contexts freed
by the garbage collector corrupted later images; RoboCasa's reset ordered counter regions by memory address; a recorded
last chunk (cut at the horizon) could not be replayed; a loaded scene carried the freezing episode's horizon; the
tracker credited stages wrongly after a latched success (StirVegetables).
