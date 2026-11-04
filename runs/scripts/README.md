# runs/scripts: shared code

| file | role |
|---|---|
| `paths.py` | Where things are: the repository root (ROBOCASA_ROOT, or found from this file), `runs/`, the GR00T python; `local()` maps paths stored by earlier checkouts into this one |
| `serve_seeded.py` | GR00T N1.5 policy server with reproducible action sampling (seed per request) |
| `serve_features.py` | The same server, also returning the action head's features of each sampled chunk (`../20261031_diver_proxy`) |
| `groot_eval.py` | Client side of the evaluation (gym env, multi-step wrapper, per-chunk state logging, video) |
| `atomic_eval.py` | Rollouts: frozen scenes (`load_scene`), exact restores (`restore_node`), stage-by-stage execution (`rollout`) |
| `atomic_steps.py`, `atomic_steps_seen11.py` | The stages of the 16 Composite-Seen tasks and their checks (`TASKS`) |
| `hidden_state.py` | State that `set_state()` misses: controller goals and caches, gripper ramp, warm start, observation clock, python fixtures (`capture` / `restore`) |
| `oracle.py` | Reset to a demo state; subtask completion signals |
| `stage_trace.py` | Stage-level trace and root cause of a rollout from its saved per-step states |
| `robot_home.py` | Put only the robot back to a task's start pose |
| `make_scenes.py` | Freeze one scene per (task, seed); `../scenegen` has the launchers and logs |
| `demo_utils.py` | Read the RoboCasa365 datasets |
| `subgoal_defs.py` | Plain-language definitions of every stage, for the diagnosis packets (`../20261025_seen480/make_packets.py`) |

Environment: the GR00T python is `envs/groot/bin/python` under the repository (or `$GROOT_PY`). Scripts run with
`python -I`, so each one puts the folders it needs on `sys.path` itself.
