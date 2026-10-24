# scenes_seen11_v2

Frozen scenes s0–s29 for the other 11 composite-seen tasks (DeliverStraw, GetToastedBread, LoadDishwasher,
PackIdenticalLunches, RinseSinkBasin, ScrubCuttingBoard, SearingMeat, SetUpCuttingStation, StirVegetables,
StoreLeftoversInBowl, WashLettuce), generated on ASUS-GPU on 2026-10-14.

- All 30 seeds (s0–s29) are used for evaluation; there are no earlier scenes for these tasks.
- How they were generated: `runs/scripts/make_scenes.py --tasks <Task> --seeds 30 --out runs/scenes_seen11_v2`,
  one process per task, seeds 0–29 in order in one env (same script and procedure as scenes_seen5_v2: robosuite
  pre-compile MJCF, ep_meta, initial state, fingerprint with phys_sha1; every scene reloaded and checked).
  Logs: `runs/_handoff/verify_asus/scenegen/<Task>.log`.
- A scene is defined by its saved file, not by its seed: the first reset(seed) of a new env in another process is not
  guaranteed to give the same scene (see ../scenes_seen5_v2_s30/README.md).
