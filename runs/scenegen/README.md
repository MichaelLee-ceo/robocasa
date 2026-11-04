# How the frozen scenes were made (2026-10-14)

The record behind `../scenes_seen5_v2_s30` (s20-s29 of the 5 original tasks) and `../scenes_seen11_v2` (s0-s29 of
the other 11 Composite-Seen tasks). They were generated on ASUS-GPU with `../scripts/make_scenes.py`, one task per
process.

| file | what |
|---|---|
| `scenegen_rest.sh`, `scenegen_rest2.sh` | the launchers (the second one replaced the first at the task boundary after GetToastedBread, with a new task order); `touch STOP` here stops them between tasks |
| `logs/<Task>.log`, `logs/rest.log` | per scene: seed, ok, reload check (`reload_mad`), layout, style, objects; the launchers' progress |
| `check_frozen_scenes.py` | the check after the machine move: every scene of `../scenes_seen5_v2` reloads identically |

Every load re-checks a scene: `atomic_eval.load_scene` refuses one whose layout, style, objects, nq or physics hash
differ from the fingerprint saved with it.
