# Environment audit: is the 16-task pipeline correct and reproducible? (2026-11-04)

> The audited code (`runs/scripts`, the earlier pipeline) was removed on 2026-11-05 and is in git at commit `84e894a`.
> The probes in `probes/` ran against it, so they no longer run in this checkout; they are kept as the evidence behind
> the findings.

Before the next phase (state understanding), the user asked for a full check of the environment that every experiment
stands on: the 16-task evaluation scripts, the stage and success checks, and snapshot / restore / continue.

**Method.** Seven auditors read the code and ran CPU probes, one per area:
- evaluation vs the official GR00T / RoboCasa eval;
- stage checks of tasks 1-8 and of tasks 9-16;
- restore of fixture and task state;
- restore of robot, controller and simulator state;
- reproducibility;
- a bug hunt in the live code.

Each finding then went to an independent skeptic who tried to refute it. The run hit the account's usage limit
part-way: 15 of 48 findings got a skeptic. The rest were checked by hand where it mattered (noted below), and several
were found independently by 3-5 auditors.

Files:
- `findings.json`: every finding with its verdict.
- `probes/<auditor>/`: the probe scripts and their small outputs. These are one-off evidence. They use absolute paths
  and are not part of the pipeline.

## Findings, grouped (duplicates merged)

| # | problem | severity | tasks | status |
|---|---|---|---|---|
| A | **Restore misses task-level Python state.** The official success of 5 tasks reads counters and latches stored on the task env, not on a fixture: WashLettuce `washed_time`, StirVegetables `success_time` and `prev_obj_pos_*`, ScrubCuttingBoard `board_contact_*`, RinseSinkBasin `washed_loc`, GetToastedBread `toaster_on`. `hidden_state.capture` / `restore` and `restore_node` never save them, so every branch inherits what earlier branches left. `restore_node`'s `set_state` also runs `update_state` once more and adds +1. Measured: StirVegetables nodes with `success_time` 1-3 come back at 6 / 9 / 13, and a zero-action step returns official success. | critical | the 5 above | verified by 3 skeptics; found by 5 auditors |
| B | **Replays do not reproduce live semantics.** (1) `collect.py` `replay()` never calls `_check_success` per step, so latches set there are missing (GetToastedBread `toaster_on` False at 8 / 8 post-toast points). (2) Live checks run on 1-substep-stale kinematics, while replay (`set_state` + `forward`) uses fresh kinematics, so contact-gated counters and latches differ. In 15 / 480 seen480 episodes the stage timing differs, e.g. StirVegetables s7 succeeds live and fails in replay. Replayed demos rarely trigger the GetToastedBread latch: `labeler_check` reports final success 14 / 20 there. | critical / major | GetToastedBread, RinseSinkBasin, StirVegetables, others at margins | measured by 3 auditors; (2) verified |
| C | **`load_scene` is not history-free.** Controller caches and StirVegetables `prev_obj_pos_*` survive from the previous episode in the same env. The 1e-13 difference grows to 0.02-0.18 within one stage under identical actions. This affects imagine roots and imagine's plain seeds (env reused across jobs) and the critic t = 0 rows. seen480 and the closed loop are not affected (fresh env per episode). | major | all, if the env is reused | measured |
| D | **Re-runs are not exact.** (1) Render noise: the same restored state gives a different policy image in 20-65 % of renders, on 15 of 16 tasks (window, fridge glass, cups, stove flames, sink water; not only oven and microwave doors). (2) The GPUs differ (RTX PRO 6000 Blackwell on GPU0 / 3, RTX 6000 Ada on GPU1, RTX 3090 on GPU2). bf16 policy kernels differ by architecture, runs mixed GPUs, and no row logs its device (KettleBoiling imagine ran on GPU1, the rest on GPU0 / 3). (3) Physics is bitwise only on CPUs that select the same OpenBLAS kernel. (4) EGL device = GPU only under `CUDA_DEVICE_ORDER=PCI_BUS_ID`. (5) Executed actions are never saved, although recorded actions replay bitwise in a new process (measured on 6 rollouts). | critical / major | all | measured |
| E | **Protocol vs the official eval.** (1) Horizons are RoboCasa v1.0.0's. This checkout is v1.0.1, and its `get_task_horizon()`, which GR00T's `run_eval.py` uses, is exactly 1.5x for all tasks. The 1x choice once matched the paper's Table 11, but that reason is no longer written down. SteamInMicrowave plain successes cluster at the cap (median step 1200 of 1400). (2) We count success at any step; GR00T's eval counts it at the last step of each chunk. (3) Frozen-scene episodes start with controller goals from a deterministic settle, not the official reset. | major / minor | all | verified |
| F | **Stage checks vs the official success.** Dead-end-prone stages: PackIdenticalLunches 0-2 (item in either tupperware; plain baseline: 6 / 12 stage-2 passes were dead ends) and SteamInMicrowave 1-2 (no re-check that the vegetable is still in the bowl). Order assumption: StackBowlsCabinet (one bowl at a time). Latch mismatch: GetToastedBread stage 0, in both directions. Not a conjunct of the success: DeliverStraw stage 0 (drawer open). Trivial credit: PreSoakPan stage 0 at t = 16 with no action. Looser / stricter: ScrubCuttingBoard stage 0, SteamInMicrowave stage 1. Transient conjuncts make earlier stages look regressed (SearingMeat). The StirVegetables diagnosis cannot see the pot-off-burner cause. | major / minor | see text | partly verified |
| G | **`settle()` / `gripper_ramp()` are validated on 3 tasks only.** They infer the hidden state at saved-trajectory points; sponge, straw, spatula, lettuce, faucet handles, toaster lever and racks were never covered. | major | 13 tasks | code reading |
| H | **Code bugs.** `imagine.progress()` raises KeyError on 'pick' / 'process' stages (4 tasks crash in collect / imagine). `closedloop.py` knows only 3 tasks. `rollout(save=...)` keeps a shallow copy of `S`. `imagine.py` drops the final partial chunk (up to 15 steps fewer than plain). `check.py` silently drops missing plain rows. `atomic_steps.TASKS` still holds 3 tasks outside the 16. | minor | see text | partly verified |

## What this means for the results so far
- **KettleBoiling, SteamInMicrowave, PackIdenticalLunches** have no task-level Python state (checked:
  no `update_state` / `_post_action` in their classes), so A does not touch the critic data, imagine or closed-loop
  results.
- **Single runs are samples, not fixed values** (D1). Imagine 96.7 %, closed loop 10 / 10 and 6 / 10 are one draw each.
  Comparisons between arms are fair: every arm used the same horizons and the same scenes.
- **Part of the SteamInMicrowave imagine gain may be speed** (E1): plain is censored by the 1x horizon.
- **The PackIdenticalLunches oracle failures** (dead ends, F) were already explained in `../20261029_closedloop`.

## Plan
1. **Decisions (user):** horizons (v1.0.0 as in the paper's Table 11, or v1.0.1 as in the current official code);
   success at any step or at chunk ends (the official eval); how to get reproducible runs (D).
2. **One environment layer** used by every experiment:
   - a fresh env per episode, or a reset proven history-free;
   - snapshots of all Python state (fixtures, task env attributes, controllers, observables), with a round-trip assert;
   - one canonical step function shared by live runs, replays and restores, so checks see the same kinematics;
   - every executed action recorded, with observation hashes, host, GPU, CPU kernel and versions;
   - stage checks redefined to be consistent with the official success (no dead-end credit, order-free where the
     official check is), evaluated with live semantics.
3. **A test suite that gates every experiment**, for all 16 tasks and their frozen scenes:
   - scene fingerprints including visuals and geometry;
   - bitwise physics replay of recorded actions;
   - load-order independence;
   - restore A/B at nodes covering every stage, fixture state, held object and base motion;
   - live vs replay equivalence of every check;
   - stage checks vs the official success and labeler, on demos and rollouts;
   - policy determinism on the pinned GPU.
4. Then re-run what the decisions require, and build the state-understanding benchmark on the verified stage tracker.

## Status after the fixes (2026-11-05)

The decisions (user, 2026-11-04): RoboCasa v1.0.1 horizons (`get_task_horizon`, as GR00T's `run_eval.py`); success
read at chunk ends, as the official eval; the paper's baseline numbers are used, not re-run; a fresh env per episode;
the official rendering (MSAA 4). The new layer is `../../envkit/` (`envkit.py`, `stages.py`, `policy.py`;
its usage guide is `../../envkit/README.md`); its tests are in `../../envkit/tests`.

| # | what was done | how it is checked (`../../envkit/tests`) |
|---|---|---|
| A | A snapshot holds the whole MjData and the whole Python object graph reachable from the wrapper stack (task attributes, fixtures, controllers, observables, wrapper buffers, RNG states). A restore writes it back in place, with no `set_state`, `forward` or `update_state`. | `check`: restore round trip, continuation, after a branch, idempotence (random actions, every scene). `policy`: the same at every stage change of a GR00T run, including the official counters. |
| B | One step function for live runs and replays: a replay re-executes the recorded actions through the same `MultiStepWrapper.step`, so every check sees what the live run saw. | `policy`: a replay reproduces success, tracker output and proprio observations at every chunk, and the final state. |
| C | A fresh env per episode. The scene is loaded from a frozen copy of the official reset (MjData dump plus the Python graph, written back by attribute path). | `check`: the loaded state equals the frozen one. |
| D | (1) MSAA 4 kept: the GPU's multisample resolve is random, so any render of the same state can differ by +-1 on a few edge pixels, even in one process (MSAA 0 is exact; GL dithering, shadows, reflections, skybox and fog are not the cause); success is judged on the physics, so the user kept the official rendering (2026-11-05); state comparisons leave images out, and an image change above 0.5 grey levels (32x32 thumbnail) fails. (2)-(4) `envkit.machine()` records the host, render GPU, CPU, BLAS kernel and versions; envkit sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`; tests and servers run on GPU1. (5) Every executed action and an observation hash per chunk are recorded. | `check`: replay across processes, images reported as render noise. |
| E | (1) v1.0.1 horizons. (2) Success = OR over chunk ends of `info["success"][0]`; the episode runs to the horizon. (3) The frozen scene is the official reset state itself (no settle). | by construction (`envkit.Episode`) |
| F | Stage tracker v2 (`stages.py`): the last stage is the official chunk-end success; official counters and latches are read, never re-run; each check is self-contained (no dead-end credit: PackIdenticalLunches counts only items that fit a valid packing, SteamInMicrowave keeps checking the vegetable, StirVegetables needs the pot on its burner); any-order objects are counted; DeliverStraw's drawer stage also counts once the straw is out; GetToastedBread's toasting stage is the official latch; where the official success itself is a latch (WashLettuce, RinseSinkBasin, StirVegetables) every stage counts as done while it holds. | `stages`: pure (state digest unchanged), 48 / 48 scenes. `policy`: official success implies every stage and later stages imply earlier ones at every chunk of 48 GR00T runs (0 gaps). |
| G | Obsolete: a snapshot records the controller and gripper state directly, nothing is inferred. | |
| H | The earlier pipeline was superseded by the new layer and removed from the working tree on 2026-11-05 (in git at commit `84e894a`; the earlier experiment folders keep their READMEs and results). | |

Four more problems were found while building the tests:
- **Freed render contexts corrupt later images.** Every hard reset builds a new MjSim and render context. The old pair
  sits in a reference cycle until Python's cyclic garbage collector frees it. robosuite's `MjRenderContext.__del__`
  then frees MuJoCo's GL objects in whichever GL context is current, usually the live one. From then on every camera
  image is stale (another camera's last image), black or garbage, while the physics is unaffected. One `gc.collect()`
  after a step reproduces it. envkit now frees each context's objects in its own GL context and collects right after
  the reset. A scan of the 18,950 earlier rollout videos (1.87 million frames) found no black, garbage or swapped
  frames. In plain runs the collector did not reach the old contexts within an episode (checked on full episodes), so
  earlier results are not affected.
- **RoboCasa's reset was not reproducible.** `Counter.get_reset_regions` orders the candidate counter tops with
  `list(set(...))` of xml Elements, which hash by memory address. The same generator state then put an object on
  either side of the sink. A failed placement then restarts `_load_model` and shifts every later draw (PreSoakPan s1
  built two different models). envkit replaces that one line with an order-keeping de-duplication. The region is still
  drawn uniformly by the fixture's rng, so the scene distribution is unchanged. This was also the cause of the earlier
  observation "the lettuce lands at x = 3.81 vs 2.57 with equal RNG states". With the fix and envkit's seeding, the
  official reset reproduces the frozen state (`check` reports `reset_reproducible`). The only exceptions are two
  metadata keys: a temporary xml path and the order of a group-name list.
- **A recorded last chunk could not be replayed.** The horizon is not a multiple of 16 (KettleBoiling: 1500 = 93 x 16
  + 12), so the last chunk executes, and records, only the remaining actions. The official `MultiStepWrapper.step`
  indexes action row `step` before it checks `done`, so passing the 12 recorded rows raised an IndexError at row 12.
  `Episode.step` now pads a short chunk with copies of its last row, which never execute, and refuses a short chunk
  that would not end the episode.
- **A loaded scene carried the freezing episode's horizon.** The frozen Python graph includes the wrapper's
  `max_episode_steps`, so `Episode(task, seed, horizon=H)` ran to the official horizon whatever H was. `_load_scene` now
  sets the episode's own horizon after the write-back. All runs so far used the official horizon, so none is affected.
