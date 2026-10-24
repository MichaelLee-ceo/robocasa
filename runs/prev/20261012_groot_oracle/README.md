# GR00T N1.5 oracle diagnosis on RoboCasa365 composite tasks (2026-10-12)

## Model
- Checkpoint: `robocasa/robocasa365_checkpoints` → `gr00t_n1-5/foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000`
  (pretraining on 300 tasks, then post-trained on the 16 Composite-Unseen target tasks, 500 human demos each). Weights only, no optimizer.
- **What this measures:** the model has *seen 500 demos per task* (including the demos whose initial states we reuse below).
  So these results answer "why does a model that has seen many demos of the task still fail?", not zero-shot generalization.
- Why not the multi-task (pretrain-only) checkpoint: it scores ~4.4% on Composite-Unseen (floor effect); oracle deltas would be unmeasurable at 10 episodes/task.
- Language the checkpoint was trained on: `annotation.human.task_description` (task-level instruction; data config `panda_omron`).
  The per-subtask annotations were released 2026-07-07, after the checkpoints were trained, so **subtask instructions are out-of-distribution
  for this model**. That's why we run both `reset_sub` (subtask instruction) and `reset_task` (task instruction).
- Inference: `Isaac-GR00T` (robocasa-benchmark fork, main), `scripts/run_eval.py --server`, data_config `panda_omron`, embodiment `new_embodiment`,
  4 denoising steps, action chunk 16 executed open-loop (n_action_steps=16), single-frame obs. torch 2.7.1+cu128, flash-attn 2.8.3 (sm_120 OK).

## Tasks (all Composite-Unseen, split=target)
judgment: CuttingToolSelection, BreadSelection, RecycleBottlesByType, SeparateFreezerRack
sequence: WaffleReheat, WashFruitColander, WeighIngredients, ArrangeTea

## Conditions (all start from the *same* demo states, so they are paired)
10 demos per task, evenly spaced over the target dataset: demo index = floor(i * N / 10) for i = 0..9, where N = number of demos in that task's
target dataset (500-520), e.g. 0, 50, 100, ..., 450 (`demos_used.json`). These demos are part of the checkpoint's training data.
Reset = `env.set_ep_meta(ep_meta.json)`, `env.reset()`, `env.reset_from_xml_string(edit_model_xml(model.xml.gz))`, `sim.reset()`,
then `sim.set_state_from_flattened(states.npz[frame])`, `sim.forward()`, `update_sites()`, `update_state()`; plus task attributes
that are normally set while sampling objects (only RecycleBottlesByType.choice, restored from ep_meta). The scene (layout, style, objects, textures) is the demo's own.
| condition | start state | instruction | budget |
|---|---|---|---|
| `baseline_demoinit` | demo frame 0 | original task instruction | original (pre-v1.0.1) task horizon |
| `planner` (oracle planner) | demo frame 0 | current oracle subtask instruction, switched when the tracker marks it done | same |
| `reset_sub` (oracle reset) | demo frame at start of subtask k | subtask k instruction | clip(3 × demo segment length, 240, horizon) |
| `reset_task` (oracle reset, task language) | same | original task instruction | same |

The planner never gives the "task complete" instruction: once every subtask is done it keeps giving the last one.

## Subtask completion signal (`scripts/oracle.py`)
Built per demo by replaying the demo states at segment boundaries (annotations: `subtask_idx`, `annotation.human.subtask`, `subtask_name`, `subtask_stage`):
- manipulated object of a pick/place pair = the object held by the gripper at the demo's end-of-pick frame (gripper-finger contact with the
  object's body tree). If nothing is held there, fall back to the object with the largest displacement over the pair. If the instruction says
  "container" and a tupperware/container object moved with it (>= 50% of its displacement), the container is the target (fix applied 2026-10-12 09:2x,
  see below).
- **pick** done: gripper closed on the object (finger contact) and the object moved > 2 cm since the subtask became active.
- **place** done: object released, and either (a) it touches a body it touched at the demo's end-of-subtask frame but not at pick start
  (new support such as cutting board or microwave), or (b) it is within 15 cm of the demo's end position (same-support moves, e.g. counter to counter).
- **navigate** done: robot base within 0.3 m (xy) of the demo end base position.
- **faucet on/off**: robocasa `sink.get_handle_state()['water_on']` (handle joint in (0.4, π)).
- **other fixture / execute** (open/close door, turn knob, spout …): every non-robot hinge/slide joint that moved > 0.05 in the demo segment
  has covered ≥ 50% of its demo motion.
- The last real subtask also counts as done when `env._check_success()` is true.
- Validation (replaying the 80 demos used, final tracker): 79/80 demos complete all subtasks. The one failing demo (`tracker_unverified.json`:
  RecycleBottlesByType d260, from k1 onward: the place predicate never fires on the demo) is excluded from per-subtask reset stats.
  Before the container fix, SeparateFreezerRack d250 and d450 also failed.
- Known limitation: button-style state (e.g. microwave "on") isn't a joint, so a fixture subtask with no joint motion can only be certified by the task success check.
  `tracker_mismatch` marks episodes where every predicate fired but the task success check never did.

## Failure labels (heuristic, `scripts/analyze_oracle.py`), for the first subtask that was not completed
- `decision`: during that subtask the gripper grasped a different object than the target and never grasped the target.
- `progress/memory`: re-grasped an object that was already placed, or a completed placement was undone.
- `motor`: reached (< 10 cm) or touched the target but never grasped/lifted it; or a fixture skill not achieved.
- `motor_or_decision`: released the target somewhere that is not a valid place (automatic labels can't separate an imprecise place from a wrong location).
- `progress/stuck`: never approached the target, holding the object but never placed it, or navigation not completed.
- `timeout`: fewer than 1.5 × demo-segment-length steps were left when the subtask became active.
These labels are heuristics computed from per-chunk logs (grasped objects, end-effector distance to target). Spot-check videos before quoting them.

## Files
- `results.json`: one record per rollout (task, split, task_type, condition, demo, subtask, success / subtask_success, steps, reached, history, failure_label, failure_detail, specs)
- `summary_by_task.csv`, `summary_reset_by_subtask.csv`; `<Task>/<tag>.mp4` videos (left agent view + wrist, 1 frame per 16-step chunk)
- `jobs.json`, `demos_used.json`, `tracker_validation_*.jsonl`, `tracker_unverified.json`; code in `../scripts/`

## Results (final: after the SeparateFreezerRack target fix and the WaffleReheat fixture-state fix; 948 rollouts, 0 errors)
Reset rollouts excluded from per-subtask stats: RecycleBottlesByType d260 k1-k5 (tracker unverified on that demo), 5 instances x 2 conditions.
In reset rollouts, a full task success during the rollout also counts as subtask success. After the fixes this changes exactly one result
(WashFruitColander d456 reset_task k6).

| condition | success | rate [95% Wilson] |
|---|---|---|
| baseline_demoinit | 48/80 | 60.0% [49.0, 70.0] |
| planner | 35/80 | 43.8% [33.4, 54.7] (paired vs baseline: 21 baseline-only vs 8 planner-only successes, McNemar exact p = 0.024) |
| reset_task (per subtask instance) | 346/389 | 88.9% [85.4, 91.7] |
| reset_sub (per subtask instance) | 317/389 | 81.5% [77.3, 85.0] (paired vs reset_task: 51 task-only vs 22 sub-only, p = 0.0009) |

- By stage (reset_task vs reset_sub): pick 132/148 vs 123/148, place 131/147 vs 130/147, execute 55/64 vs 53/64, **navigate 28/30 vs 11/30**
  (navigate exists only in SeparateFreezerRack).
- By task type, baseline vs planner: judgment 22/40 vs 21/40, sequence 26/40 vs 14/40.
- Conditional subtask success inside full baseline rollouts (instances whose start was reached) = 92.5% vs oracle reset_task on the same
  321 instances = 89.4%. 2x2 (baseline_ok, reset_ok): TT 272, FT 15, TF 25, FF 9. So no sign of drift (`conditional_subtask_success.json`).
- Chain test per demo: `error_accumulation.json`. Reset budgets are tighter than the full horizon, so chain < e2e for some tasks.
- `tracker_mismatch`: baseline 8/80, planner 9/80 (every predicate fired, env success never did); mostly WeighIngredients / ArrangeTea,
  where the place (15 cm fallback) or fixture (>= 50% joint motion) predicate is looser than the env check.
- Seeded baseline on the same 8 tasks (repro run, 20 random seeds each) averages 52.5% vs 60.0% from these demo initial states
  (biggest gaps: WeighIngredients 30% vs 70%, WaffleReheat 75% vs 100%), so starting from training-demo states makes the tasks somewhat easier.

## Changes made during the run (in order)
1. Faucet predicate switched to `water_on`. No effect on results: in every WashFruitColander demo the faucet subtask is the last one, which is
   already certified by `_check_success()`.
2. Container-vs-content target fix (SeparateFreezerRack). In 4/10 demos (d0, d250, d300, d450) the tracker targeted the food inside the
   container (vegetable1/vegetable2/meat) instead of veg_tupperware/meat_tupperware. So correct container grasps counted as "wrong object",
   which could also stall the planner on that subtask (after the fix the planner is still 0/10 on this task,
   failing mostly at k1 navigate). All 64 rollouts of those 4 demos (baseline, planner, reset_task ×7, reset_sub ×7) were rerun with the
   fixed tracker. The old records are kept in `_superseded_sfr_targetbug/`. No other task had this problem (checked every pick/place target
   against its instruction text).
3. WaffleReheat fixture-state contamination (found 2026-10-12 ~10:00). Building the tracker replays demo states, and each `set_state()`
   calls `fixture.update_state()`. At demo frames where the gripper presses the microwave start button with the door closed, that sets the
   Python-level `Microwave._turned_on = True`, which isn't part of the MuJoCo state and carried into the rollout. Confirmed in 6/10 demos
   (d0, d100, d150, d250, d300, d350): `_check_success()` was already True at the close-door start state. All 120 WaffleReheat rollouts were rerun
   after rebuilding the scene (`reset_to`) between building the tracker and the rollout. Old records are in `_superseded_waffle_microwave_state/`.
   An earlier note blaming a "door-open success loophole" in the benchmark was WRONG and is retracted. In the seeded repro run all 15
   WaffleReheat successes have the door closed (opening <= 0.06; `../20261012_groot_repro_h1x/wafflereheat_door_at_success.json`). Among the
   8 tasks, only WaffleReheat has a fixture with Python-level state (microwave). Blender, coffee machine and stand mixer have the same pattern but
   aren't used here.
4. The inference server on port 5555 was stopped by a background time limit mid-run and restarted. In-flight jobs were rerun (resume by tag); none
   were written twice.
