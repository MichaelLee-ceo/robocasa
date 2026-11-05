# Closed-loop chunk steering with an oracle (2026-10-29)

> **Record only (2026-11-05).** This experiment ran on the earlier pipeline, which had known problems (`../20261104_env_audit`). Its code was removed from the working tree; it is in git at commit `84e894a`. Its large outputs (git-ignored) were moved to `~/mclee/robocasa_old_data/runs/<this folder>/`. This folder keeps the README and the results. New work uses `../envkit`.

**Question.** The critical chunks are where choosing among candidate chunks changes the outcome: +25 points of stage
completion, against about 0 at random chunks (`../20261028_critical/README.md`). If we steer chunk by chunk in a real
episode, how close does task success come to the stage-level imagine target (`../20261026_imagine/`)? And how many
chunks actually needed a choice?

**Method** (`closedloop.py`, one process per episode):
- The episode runs for real from the frozen scene's start, up to the task's original horizon (KettleBoiling 1000,
  SteamInMicrowave 1400, PackIdenticalLunches 2600). The last chunk may be partial (8 steps).
- **At every chunk boundary:**
  - Snapshot the current state. That is the sim state, the python fixture state, the stage memory and the controller
    state. The capture is exact, because every node is reached by stepping.
  - Imagine N = 8 candidates. Each is the policy's chunk with its own noise seed, then the plain policy until the
    current stage's check holds or L[k] steps pass, with one continuation each.
  - **Critical** means some candidates finish the stage and some do not. Then the earliest finisher is executed:
    "steered". Otherwise candidate 0 is executed, which is what the plain policy would draw.
- The chosen chunk's recorded actions are executed for real. A live check confirms that the executed state after 16
  steps equals the imagined one, bit for bit.
- A node keeps its last observation, the observables' sampling clock and the python fixture state
  (`hidden_state.capture(kenv, obs=True)`). `atomic_eval.restore_node` also undoes the extra `update_state()` that
  `set_state` runs, and asserts that the sim state equals the node. So a rollout from a node continues exactly like a
  continuous run. The first KettleBoiling run lacked the observation part; see the history below.
- **Success:** the official check held at some step or at a chunk boundary (both coincide; logged as `reward_success` and
  `all_stages`). The plain baseline and imagine use the same definition.
- **A candidate finishes the stage** when the stage index advances past k (or success), the same definition as the
  critic data and the plain runs' stage history.
- Imagining does not count against the episode's horizon. Its cost is logged as policy calls.
- **Logged per chunk:** stage, how many candidates finished, steered or not, the pick, whether candidate 0 (plain)
  would have failed, and the stage's debug values (KettleBoiling: the burner under the kettle).

**Pilot: KettleBoiling, 10 scenes** where the plain policy often fails: s2, s4, s5, s8, s9, s10, s12, s17, s19, s24.
- Plain policy on them (4 seeds each): 13 / 40 = 32.5 %.
- Stage-level imagine: 10 / 10.
- Launcher `run.sh` (detached, one retry per episode), results in `out/<tag>.json`, log `run.log`.

## SteamInMicrowave and PackIdenticalLunches (launched 2026-10-30)
**Scenes:** the 10 per task where the plain policy fails most (4 seeds; `check.py` PLANNED), chosen by the plain policy
only.
- SteamInMicrowave: s0, s2, s3, s5, s8, s12, s14, s15, s27, s28.
- PackIdenticalLunches: s0, s1, s2, s4, s5, s10, s11, s13, s14, s15.

The imagine target on the same scenes comes from its re-run (`../20261026_imagine/sim_pil`).

**Pre-flight** (`preflight.py`, outputs in `preflight/`). Six full episodes on the plain path, committed node by node like
the closed loop: SteamInMicrowave s24, s6, s5 and PackIdenticalLunches s9, s8, s0.
- **Reset.** load_scene after another rollout leaves controller caches that differ by about 1e-12. The state then
  differs by about 1e-13 (`reset_culprit.py`). The plain baselines, run one after another by workers, are therefore
  fine.
- **Every node (734).** After `restore_node`, the sim state and python fixture state equal the node's. The per-step
  reward never disagreed with the official check at a chunk end.
- **Every 4th node (180): closed-loop A/B.** Candidate 0 continuous (A) vs commit, restore, continue (B), and vs commit,
  restore, replay A's own actions (B2).
  - B2 is bit-identical to A in all 180, in sim state, python fixture state, stage memory and outcome.
  - B is bit-identical until GPU render noise changes an image. The first difference was never in the state keys of
    the observation.
- **Every 8th node (86): imagine A/B** through `imagine.candidate` and its child nodes. Same result.
- **Coverage.**
  - All 4 stages of both tasks.
  - Holding the bowl, the vegetable, meat and vegetables.
  - The base driving up to 59 cm and turned -174 to +178 degrees.
  - The microwave and fridge doors open.
  - Finger contact with fixtures.
- **Rendering.** PackIdenticalLunches kitchens render non-deterministically: the same state gives 2-8 distinct images
  out of 20. So exact replays there hold only until the first render.

**Launched 2026-10-30 00:49:** `./run.sh SteamInMicrowave 0 2 3 5 8 12 14 15 27 28` and
`PORT_OFFSET=10 ./run.sh PackIdenticalLunches 0 1 2 4 5 10 11 13 14 15`. 16 policy servers serve these and the imagine
re-run. A policy call costs about 2.3 s (SIM) or 3 s (PIL) of single-process time.

### SteamInMicrowave result: 10 / 10, the imagine target, against 7.5 % for the plain policy (finished 2026-10-30 10:43)
`check.py` finds every chunk consistent, and the live check held everywhere. On these 10 scenes the plain policy succeeds
3 / 40 (7 scenes 0 / 4), and imagine 10 / 10.

| scene | ours | steps | chunks | steered | steered where candidate 0 would fail | policy calls | steered / chunks in stages 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|
| s0 | yes | 928 | 58 | 15 | 5 | 5688 | 8/22 / 4/24 / 1/9 / 2/3 |
| s2 | yes | 1152 | 72 | 23 | 6 | 8652 | 0/15 / 8/34 / 1/9 / 14/14 |
| s3 | yes | 992 | 62 | 13 | 1 | 6102 | 0/20 / 6/24 / 6/10 / 1/8 |
| s5 | yes | 1072 | 67 | 26 | 5 | 7491 | 2/18 / 12/29 / 5/11 / 7/9 |
| s8 | yes | 1376 | 86 | 24 | 9 | 8051 | 1/18 / 3/27 / 7/18 / 13/23 |
| s12 | yes | 1056 | 66 | 40 | 20 | 8072 | 9/16 / 18/24 / 4/14 / 9/12 |
| s14 | yes | 896 | 56 | 16 | 3 | 5181 | 0/15 / 5/22 / 5/12 / 6/7 |
| s15 | yes | 1264 | 79 | 34 | 13 | 8669 | 3/18 / 11/31 / 1/11 / 19/19 |
| s27 | yes | 864 | 54 | 20 | 2 | 5701 | 5/18 / 13/25 / 0/8 / 2/3 |
| s28 | yes | 1312 | 82 | 44 | 18 | 10756 | 7/18 / 10/23 / 17/28 / 10/13 |
| **all** | **10/10** | | 682 | 255 | 82 | 7,436 per episode | |

- **Choosing chunks reaches the imagine target here too.** On scenes where the plain policy almost never succeeds, all
  10 episodes finish.
- **Where the choices are.** At 255 of 682 chunks (37 %) some candidates finished and some did not. At 82 (12 %) the
  plain draw would have failed the stage.
  - Most choices are in stage 1 (pick the bowl, drive to the microwave, place it) and stage 3 (press the start
    button).
  - In stage 3 some episodes chose at every chunk (s2 14 / 14, s15 19 / 19). A single continuation per candidate is a
    noisy label there: pressing the button succeeds in some continuations and not in others from nearly the same
    state.
- **Cost.** 7,436 policy calls per episode, about 10 x imagine (720 per scene over all 30) and 85 x a plain episode
  (about 87 calls).

### PackIdenticalLunches result: 6 / 10, below the imagine target (8 / 10); 3 failures are dead ends (2026-10-31)
`check.py` finds every chunk consistent. On these scenes the plain policy succeeds 0 / 40. s14 failed too: it spent
most of its horizon in stage 1 (stage 2 of 4 at t 2600; 36,489 policy calls).

| scene | ours | imagine | steps | chunks | steered | steered where candidate 0 would fail | policy calls | stage 3: chunks / with no finisher |
|---|---|---|---|---|---|---|---|---|
| s0 | yes | yes | 1536 | 96 | 38 | 14 | 11259 | 28 / 0 |
| s1 | yes | no | 1680 | 105 | 41 | 13 | 12941 | 27 / 0 |
| s2 | yes | yes | 1248 | 78 | 7 | 0 | 6735 | 25 / 0 |
| s4 | **no** | yes | 2600 | 163 | 26 | 12 | 28377 | 90 / **90** |
| s5 | yes | yes | 1344 | 84 | 40 | 16 | 10067 | 23 / 0 |
| s10 | **no** | yes | 2600 | 163 | 34 | 9 | 27991 | 85 / **85** |
| s11 | **no** | yes | 2600 | 163 | 49 | 12 | 27591 | 73 / **73** |
| s13 | yes | yes | 1440 | 90 | 32 | 16 | 10948 | 24 / 0 |
| s15 | yes | yes | 1712 | 107 | 76 | 43 | 18841 | 28 / 5 |

**Why the 3 failures: the stage-local rule walks into dead ends.**
- Stages 0-2 here only ask for one more item in *either* tupperware. The last stage is the official check: one
  vegetable and one meat in each tupperware, no duplicates. Two items of the same kind in one tupperware make the
  official check unreachable.
- s4, s10 and s11 entered stage 3 with no candidate able to finish it from the very first chunk, for 73-90 chunks.
  Every successful episode had finishers in stage 3.
- The executed path cannot be replayed: re-querying the policy diverges within the first stages, because PIL
  rendering is not reproducible (`replay.py --stop_k 3`: all 5 replays `match: false`). So the placement
  at stage 3 is not observed directly.
- The critic data (every step's state saved) shows the mechanism (`pil_deadend.py`, `pil_deadend.txt`):

  | | n | result |
  |---|---|---|
  | stage-3 decision points with a dead-end placement | 20 | stage-3 success **0.0 %** |
  | stage-3 decision points without | 47 | stage-3 success 27.1 % |
  | stage-1 finishes that are dead ends | 985 | 9.4 % (earliest finisher 7.7 %) |
  | stage-2 finishes that are dead ends | 609 | 17.9 % (earliest finisher 17.1 %) |

  About a quarter of the paths through stages 1-2 end in a dead end, consistent with 3 of 9.
- The placement is a proxy of `OU.check_obj_in_receptacle` from the bodies' positions (no contact check). It misses
  some items at later stages (at the stage-3 points it finds 2 items in 26 of 67), so these shares are lower bounds.
- The same applies to the critic data: a "finished" stage 1 or 2 of PIL is a dead end 9-18 % of the time.

**Fixed before the launch** (found by the pre-flight and two independent review agents):
- **Imagine's success.** It required the official check after restoring the end state. That counted 4 old searches as
  failures although a candidate had succeeded (SIM s5, s22; PIL s7, s24).
- **Imagine's finisher rule** also required every earlier stage check to still hold. That rejected about 20 % of the
  SIM stage-2 finishers, one of them an official success, mostly because of the gripper-25-cm-from-the-bowl release
  condition. Now it uses the same definition as here.
- **Horizon.** The closed loop stopped 8 steps before the horizon.
- **Restore.** `update_state()` ran again at a restore and can change contact-gated fixture flags or write the toaster
  oven's timer knob. `restore_node` now undoes it.
- **Bookkeeping.** `check.py` lists missing episodes, results are written atomically, and the imagine launcher refuses
  a stale jobs file.

## Result: 10 / 10, the imagine target, against 32.5 % for the plain policy (re-run with the observation fix, 2026-10-29 21:00-23:19)
`check.py` finds every chunk of every episode consistent. The live check found execution equal to imagination
everywhere.

| scene | ours | plain (4 seeds) | steps | chunks | steered | steered where candidate 0 would fail | policy calls | stage 1 from | burner under the kettle |
|---|---|---|---|---|---|---|---|---|---|
| s2 | yes | 1/4 | 400 | 25 | 15 | 11 | 2861 | t256 | front_right, 5.6 cm |
| s4 | yes | 2/4 | 576 | 36 | 14 | 7 | 3476 | t416 | front_right, 6.8 cm |
| s5 | yes | 0/4 | 336 | 21 | 11 | 4 | 2061 | t192 | front_left, 4.2 cm |
| s8 | yes | 1/4 | 448 | 28 | 7 | 3 | 2134 | t288 | front_left, 5.0 cm |
| s9 | yes | 2/4 | 400 | 25 | 6 | 3 | 1689 | t256 | front_left, 3.9 cm |
| s10 | yes | 0/4 | 464 | 29 | 7 | 4 | 1993 | t256 | front_left, 3.0 cm |
| s12 | yes | 2/4 | 464 | 29 | 11 | 4 | 3227 | (both stages at once) | |
| s17 | yes | 2/4 | 448 | 28 | 23 | 7 | 2924 | t304 | front_right, 9.4 cm |
| s19 | yes | 2/4 | 480 | 30 | 9 | 3 | 2644 | t304 | front_left, 5.9 cm |
| s24 | yes | 1/4 | 432 | 27 | 11 | 4 | 2224 | t320 | front_left, 1.7 cm |
| **all** | **10/10** | **13/40 = 32.5 %** | | 278 | 114 | 50 | 2523 per episode | | |

- **Choosing chunks reaches the imagine target.** Choosing among the policy's own candidate chunks, chunk by chunk,
  solves every scene. The plain policy solves 32.5 % of them, and two scenes never (0/4).
  - Imagine reaches 10 / 10 too, by searching whole stage trajectories.
- **The first run's failures came from the observation bug, not from the stage-only criterion.**
  - s5, s8 and s12 failed there with stage-1 dead ends. Now all three succeed: s5 enters stage 1 at t192 instead of
    t400, s8 at t288 instead of t720.
  - So a task-aware criterion is not needed on these scenes.
- **The kettle's burner sets how hard stage 1 is, and steering handles both.**
  - front_left (6 scenes): stage 1 needed 0-2 choices and had no chunk where all 8 candidates failed.
  - front_right (s2, s4, s17): a choice at 6-9 of 9-10 stage-1 chunks, and s2 had 2 chunks with no finisher.
  - In s12 the policy turned the burner on before setting the kettle down, so both stages finished together.
- **Where the choices are.** 114 of 278 chunks (41 %) had some candidates finish and some not. At 50 (18 %) the plain
  draw (candidate 0) would have failed the stage.
  - Most are in stage 0 (approach and grasp), plus stage 1 after a front_right placement.
- **Cost: 2,523 policy calls per episode.** That is about 9 x imagine (278 per scene over all 30 scenes) and about
  60 x a plain episode (41). Every chunk imagines 8 candidates to the end of the stage.
  - The goal is to reach this success while choosing only at the few chunks that decide the outcome, judged by a critic
    instead of simulation.

## History: first run (superseded)
**7 / 10 against plain 32.5 %, but not reliable.** The first run (14:00-18:36; outputs deleted after the re-run was
checked) restarted every node with a *forced* observation:
- **What went wrong.** `rollout()` forces a fresh observation at its start. That observation is 1-2 physics substeps
  fresher than what a continuous run sends (restore review F4), and it also advances the sampling clock by a substep.
  - Physically the difference is tiny: < 0.5 mm end-effector, 4-37 % of pixels slightly changed.
  - But at every node the real episode saw a different observation from the one the imagined continuation saw.
- **How it was found.** `lockin_check.py` compares two runs from a node:
  - A: a steered candidate run continuously, as the closed loop imagined it;
  - B: the same candidate with its first chunk committed, then restore and continue with the same seeds.

  | | rows | outcome flips A vs B | first differing observation |
  |---|---|---|---|
  | before the fix (`lockin/`) | 40 | 9 (22 %) | always at step 1, in the state keys and all images |
  | after the fix (`lockin_fix/`) | 8 | 0 | images only (render noise, `../README.md`) in 7, at chunk 0-10; none in 1 |

  - After the fix, the proprioceptive state keys never differ. The only differences are single camera images.
  - When the first chunk's image differs, it is because the committed chunk re-renders it.

  - The clearest case is s8 t144. The picked candidate finished in 64 steps continuously, but after the
    forced-observation restart it did not finish in 416. The first run then had 0 of 8 candidates finishing for 10
    chunks.
- **What the first run showed (the re-run above supersedes it):**
  - steering happened mostly in the approach and grasp of stage 0;
  - in the 3 failures (s5, s8, s12) stage 1 was all-or-nothing: every chunk there had 0 of 8 candidates finishing,
    while no successful episode had a single such chunk;
  - s12 put the kettle on the center burner, 11.5 cm off its center (from `replay.py`; the other replays drifted by
    render noise). No successful plain or closed-loop episode used the center burner;
  - imagine did not backtrack on s5 or s8. It finished stage 0 earlier: t224 vs t400 on s5, t320 vs t720 on s8;
  - the rule takes candidate 0 when all 8 finish, even when candidate 0 is much slower. On s5 at t224 the fastest
    finisher needed 16 steps and candidate 0 needed 240.
- **Rendering noise remains.** Even from an exact node with the same seeds, a candidate's outcome can flip: s5 t0
  candidate 4 finished at 256 in the run and did not finish in the re-check. So one continuation per candidate is a noisy
  label.

Caveat: with one continuation per candidate, "some finish and some do not" also happens by luck at chunks where the
choice does not matter. So the count of steered chunks is an upper bound on the chunks that truly needed a choice.
"steered while candidate 0 would have failed" is the tighter count.
