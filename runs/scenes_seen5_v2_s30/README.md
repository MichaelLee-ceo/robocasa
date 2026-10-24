# scenes_seen5_v2_s30

Frozen scenes for the 5 original composite-seen tasks (KettleBoiling, PreSoakPan, StackBowlsCabinet,
SteamInMicrowave, PrepareCoffee), generated on ASUS-GPU on 2026-10-14 to extend the evaluation from 20 to 30 seeds.

1. **Evaluation uses only s20–s29 from this folder.** s0–s19 come from `../scenes_seen5_v2` (the original files).
2. **s0–s19 here are only a reproduction check and are not used for evaluation.** Their fingerprints were compared
   with `../scenes_seen5_v2`:
   - PreSoakPan: 16/20 identical; s2, s3, s5, s9 differ (layout, style, objects, nq, phys_sha1).
   - Cause: `../scenes_seen5_v2` was generated with all 5 tasks in one process (KettleBoiling, PreSoakPan,
     StackBowlsCabinet, SteamInMicrowave, PrepareCoffee in that order). Scene sampling depends on the process's
     earlier resets, not only on the seed, so the first reset(seed) of a new env in a different process is not
     guaranteed to produce the same scene. Scenes are therefore defined by the saved files, not by the seed.
   - Other tasks: see the check results added below as they are generated.
3. **How they were generated:** `runs/scripts/make_scenes.py --tasks <Task> --seeds 30 --out runs/scenes_seen5_v2_s30`,
   one process per task, seeds 0–29 in order in one env (same script and per-seed procedure as scenes_seen5_v2:
   robosuite pre-compile MJCF, ep_meta, initial state, fingerprint with phys_sha1; every scene reloaded and checked).
   Logs: `runs/_handoff/verify_asus/scenegen/<Task>.log`.

## Reproduction check of s0–s19 against scenes_seen5_v2
| task | identical fingerprints | differing seeds |
|---|---|---|
| KettleBoiling | 20/20 | - |
| PreSoakPan | 16/20 | s2, s3, s5, s9 |
| StackBowlsCabinet | 20/20 | - |
| SteamInMicrowave | 19/20 | s5 |
| PrepareCoffee | 20/20 | - |
