# scenes_seen5_v2

Frozen scenes (model XML, episode metadata, initial state and a fingerprint) for the 5 original Composite-Seen tasks
(KettleBoiling, PreSoakPan, PrepareCoffee, StackBowlsCabinet, SteamInMicrowave), scenes s0-s19.
- Made with `../scripts/make_scenes.py`. Loaded with `atomic_eval.load_scene`, which refuses a scene that does not
  reload identically.
- s20-s29 of these tasks are in `../scenes_seen5_v2_s30`, and the other 11 tasks in `../scenes_seen11_v2`.
- Scenes of 3 tasks from an earlier study (ArrangeVegetables, CondimentCollection, GatherVegetables) were deleted on
  2026-11-04.
- The XMLs store the asset paths of the machine that made them. `load_scene` resolves them, and `../scripts/paths.py`
  maps the old checkout prefixes to this checkout.
