# Recovery after a detected failure (started 2026-11-05)

**Question.** If a failure is noticed after it happens, how many failures can be recovered without the simulator
(no lookahead, no restore)? Which recovery works best: resampling, a stage-level instruction, or the atomic model?
Once that is known, can a VLM do the noticing?

**Why.**
- In the plain baseline (`../20261025_seen480/diagnosis`), GR00T made no recovery attempt at the fatal error in 57 % of
  failures. It carries on empty-handed after a missed grasp, and it leaves drawers and levers part-way.
- GR00T is memoryless (one frame and state per chunk), so it cannot notice its own errors. A supervisor that notices
  them is the proposal's state understanding.

**Pilot scope.** Two failure types on four tasks (Composite-Seen, envkit):
- missed grasp: KettleBoiling S0, SetUpCuttingStation S0;
- fixture moved part-way or the wrong fixture: DeliverStraw S0 (drawer), GetToastedBread S0 (toaster lever).

## Plan
| phase | what | status |
|---|---|---|
| 0 | Plain GR00T, 30 scenes per task, in envkit (paired baseline and source of failure states); failure events labelled | done for KettleBoiling, SteamInMicrowave, PackIdenticalLunches (below) |
| 1-2 | From known failure points (exact replay), one recovery each: continue, MBR resampling, stage instruction, atomic_seen model, back off; and fresh attempts from the episode start | pilot done (below) |
| 2b | The recovery loop (check, intervene, check, escalate) from the phase 0 failure points | done (below): language and reversed commands do not beat control; retracing to the stage start does (success 4 % -> 22 %, fixed 12 % -> 37 %) |
| 3 | Detection by a VLM (and proprio) instead of the oracle | pilot below |

The old 480-episode baseline is used for planning only. Its scenes, its horizons (2/3 of v1.0.1) and its success
timing differ from envkit, and its states cannot be restored in envkit.

## Recovery from known failure points (pilot, 2026-11-05)
`recover.py` (one trial), `recover.sh` (80 trials), `recover_restart.sh` (fresh attempts); results in
`recover/results.jsonl`; what the robot did: `inspect_trial.py` (replay, per-chunk fixture state; output in
`recover/inspect/`) and `trial_video.py` (camera frames per chunk; mp4 and sheets in `recover/video/`, not in git).
- **Setup.** 8 failure points in the recorded envkit runs (3 scenes per task). A trial replays the run exactly to the
  failure chunk, applies one recovery, and runs up to 40 chunks. It succeeds when the failing stage is done
  (`stages.track`). Detection is perfect by construction: this tests only the recovery. 2 policy seeds per cell.
- **Recoveries.** `continue` (the control); `resample` (8 samples per chunk, keep the medoid); `stage_instr` (the
  stage's sentence on the composite model); `atomic` (the atomic_seen model with the atomic task's sentence; only
  OpenDrawer and PickPlaceCounterToStove are atomic_seen tasks here); `backoff` (the last 3 chunks reversed with the
  gripper open, then continue).

Stages done within 40 chunks (chunk where done, x = not done; "-" = no atomic_seen task for this stage):

| failure point | continue | resample | stage_instr | atomic | backoff |
|---|---|---|---|---|---|
| DS1 DeliverStraw s1, drawer pulled to 0.39 and let go (chunk 23) | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| DS2 DeliverStraw s2, another drawer pulled (chunk 12) | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| DS0 DeliverStraw s0, never reaches the drawer (chunk 15) | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| GTB0 GetToastedBread s0, lever to 0.75 and let go (chunk 15) | 1/2 (3, x) | 1/2 (4, x) | 1/2 (x, 2) | - | 0/2 |
| GTB1 GetToastedBread s1, lever to 0.27 (chunk 63) | 0/2 | 0/2 | 0/2 | - | 0/2 |
| GTB2 GetToastedBread s2, lever to 0.50 and let go (chunk 59) | 0/2 | 0/2 | 0/2 | - | 0/2 |
| KB2 KettleBoiling s2, grasps at air (chunk 8) | 0/2 | 1/2 (x, 25) | 0/2 | 0/2 | 1/2 (13, x) |
| SUCS0 SetUpCuttingStation s0, steak dropped (chunk 48) | 1/2 (x, 15) | 2/2 (27, 26) | 2/2 (18, 15) | - | 0/2 |
| total | 2/16 | 4/16 | 3/16 | 0/8 | 1/16 |

Fresh attempts from the episode start in the same scenes (`restart`, 4 policy seeds, stage 0 within 60 chunks):
DeliverStraw s0 1/4 (chunk 16), s1 0/4, s2 0/4; GetToastedBread s0 0/4, s1 1/4 (chunk 47), s2 0/4; KettleBoiling s2
2/4 (chunks 13, 21).

**What the robot did** (replays of all DeliverStraw and GetToastedBread trials, frames of seven):
- **DeliverStraw: it does not find the target drawer again.**
  - DS2: the wrong drawer stays fully open, and the robot keeps working in it; the hand stays 0.28-0.73 m from the
    target drawer.
  - DS0: the hand never comes within 0.42 m of the target drawer.
  - DS1: 9 of 10 trials end with the half-open drawer pushed closed (0.00-0.34). The one exception (continue, seed 0)
    pushed it to 0.87, still below OpenDrawer's 0.95; successful runs in the old baseline open it to 1.6-1.8.
  - The task sentence says "the drawer in front". After the base has moved, nothing in "Open the drawer." or "Open the
    right drawer." says which drawer.
  - The atomic_seen model was trained with the robot placed beside the drawer (half its width + 0.20 m, and the
    sentence names that side). At DS1 the drawer is straight ahead (lateral offset -0.02 m), outside those placements.
- **GetToastedBread s1, s2: at the right lever, but it does not press it down.** The bread is still in its slot, the
  hand is 0.14-0.5 m from the lever of that slot, and the lever reaches at most 0.28-0.50 (on at 0.90).
- **GetToastedBread s0: one unlucky sample.** At chunk 15 the hand is still on the lever; 3 of 6 continuations press it
  fully within 2-4 chunks. A fresh start does not get there (0/4 in 60 chunks): being at the lever is closer to success
  than the start.
- **Grasps.** SUCS0's own run picked the steak up again 67 chunks later; the recoveries did it in 15-27. KB2: the air
  grasp is followed by a good grasp in 1 of 2 resampled and 1 of 2 backed-off trials.

**Reading.**
- Once a failure is identified, a retry helps where GR00T can do the stage from that state (grasps; the lever when the
  hand is still on it): about one in two per attempt. Five retries would then succeed about 97 % of the time if the
  attempts were independent (to be measured).
- It does not help where GR00T unaided rarely does the stage in that scene: the drawer scenes and toaster s1/s2 are
  about 1 in 12 even from a fresh start. These failure points come from scenes where the plain run failed, so they are
  biased towards scenes that are hard for GR00T.
- **Not concluded: whether guidance can help there.** Every recovery here is a fixed rule with no VLM in it: the same
  sentence whatever went wrong, nothing about where the target is (DS2 was told "Open the drawer." while the wrong
  drawer stood open), one attempt, and no check that GR00T acted on the sentence. Open questions: does GR00T follow
  a changed sentence at all (steerability), and is a good chunk among its samples at these states (selection)?
- MBR resampling picks the typical chunk. Where success is rare, the typical chunk is the failing one; the oracle
  ceiling (choosing chunks with lookahead) finds the rare good ones, which MBR cannot.
- The atomic_seen model did not help (0/8), and its sentences depend on a placement the composite runs do not have.
- 2 seeds per cell: this separates failure types that can be recovered from those that cannot, not one recovery from
  another.

## Does GR00T act on a changed instruction? (steerability probe, 2026-11-05)
`steer_instructions.py` (candidates; `steer/instructions.json`), `steer_probe.py` / `steer_probe.sh` (rollouts;
`steer/probe/`), `steer_score.py` (`steer/scored.jsonl`).
- **Setup.** At each of the 8 failure points: 15 instructions x 4 policy seeds x 8 chunks, composite model, from the
  same restored state. The seed is shared across instructions and the restored observation is identical, so the first
  chunk differs only through the instruction.
- **Instructions.** The task sentence; its clause for the failing stage ("Start the toaster."); the stage's atomic
  template; RoboCasa's 3 official paraphrases; 6 written by Opus 5.5 (given the 16 training sentences, the start and
  current views, the stage and an oracle diagnosis; $0.17 for all 8); 3 controls (empty, another task's sentence,
  nonsense).
- **Action effect** = mean distance of the first chunk from the task sentence's chunk at the same seed, divided by the
  distance between two seeds of the task sentence (0 = ignored, 1 = as much as a new noise sample).

| family | action effect | stage done in 8 chunks | change in closest hand-target distance | change in best drawer / lever / grasp |
|---|---|---|---|---|
| task sentence | 0 | 2 / 32 | | |
| official paraphrases | 0.36 | 4 / 96 | +0.000 m | -0.02 |
| clause | 0.57 | 1 / 32 | -0.010 m | +0.06 |
| stage template | 0.76 | 2 / 32 | -0.015 m | +0.00 |
| Opus-written | 0.77 | 3 / 192 | -0.004 m | -0.02 |
| controls | 1.55 | 2 / 96 | +0.009 m | -0.03 |

(stage done: all at GTB0, where the lever was already half-way.)
- **GR00T reads the instruction, but as a perturbation.** The further the text is from its training sentence, the more
  the chunk changes: paraphrases least, nonsense most. The change is not towards what the sentence asks.
- **No directed effect at the drawer.** "Move to the right and open the drawer in front." and "Open the closed drawer
  on your right ..., not the drawer on your left." leave the hand 0.69-0.70 m from the target drawer at DS2, as the task
  sentence does; no instruction moved a drawer.
- **Small, unconfirmed directed effects (4 seeds):** the stage template brings the hand to the toaster lever at GTB1
  (0.34 m vs 0.50 m) and closer to the kettle at KB2 (0.25 m vs 0.32 m); the clause "Place the meat from the plate to the
  cutting board." grasps the steak in 3 of 4 (task sentence 1 of 4). None of these pressed the lever or lifted the kettle.
- **The instruction keeps GR00T on its task.** With the empty or nonsense instruction at GTB1, the base drives 0.7-1.2 m
  away from the toaster.
- Limits: 8 chunks is short (a stage takes 15+ chunks); 4 seeds; one model (composite).

## Instructions in GR00T's own wording (steer v2, 2026-11-05)
`steer_bank.py` (phrase bank and level rules), `steer_gen.py` (Opus writes from the bank; `steer/v2/instructions.json`),
`steer_rank.py` (ranking from predicted chunks, no simulator; `steer/v2/rank/`), `steer_v2.sh` (ranking, then rollouts
with `steer_probe.py`, `STEER_DIR=steer/v2`), `steer_score.py` (`STEER_DIR=steer/v2`; `steer/v2/scored.jsonl`).
- **Phrase bank.** The 16 composite-seen task sentences split into 29 clauses, plus 12 RoboCasa atomic templates for the
  skills of our failing stages.
- **Levels.** L0: a bank sentence verbatim (slots filled with this scene's words). L1: only bank and scene words,
  recombined. L2: L1 plus at most 4 words from a small list (fully, all the way, again, firmly, left, right, ...).
- **Generation.** Opus 5.5 wrote 3 per level for 6 failure points (the 4 that reacted in the first probe, and DS1, DS2);
  all 54 passed the rule check ($0.17).
- **Ranking without the simulator** (as PALO chooses language): 8 sampled chunks per instruction, commanded path from
  the actions (the OSC model of `../20261030_vlm_critic`, commit 84e894a), scored by how much closer it brings the hand
  to the target's true position (oracle pointing).
- **Rollouts.** Every candidate, 5 policy seeds, 20 chunks, from the restored failure state.

| failure point | task sentence | L0 | L1 | L2 | what changed with bank wording |
|---|---|---|---|---|---|
| SUCS0 steak dropped | 2/5 | 7/15 | 6/15 | 6/15 | nothing beyond noise; GR00T re-grasps on its own here |
| GTB1 lever to 0.27 | 0/5 | 0/15 | 0/15 | 0/15 | the task sentence drives the base 1.0 m away from the toaster; every bank sentence keeps the base within 0.22 m; those naming the lever bring the hand to 0.19-0.27 m of it ("Start the toaster." 0.39 m, task sentence 0.49 m), but the lever reaches at most 0.05 |
| GTB2 lever to 0.50 | 0/5 | 0/15 | 0/15 | 0/15 | none; the lever stays at 0.30-0.37 |
| KB2 air grasp | 0/5 | 0/15 | 0/15 | 0/15 | the hand comes slightly closer (0.18-0.22 m vs 0.24 m); no grasp |
| DS1 drawer at 0.39 | 0/5 | 0/15 | 0/15 | 0/15 | none; the drawer stays at 0.40-0.49 |
| DS2 wrong drawer | 0/5 | 0/15 | 0/15 | 0/15 | "to your right" moves the base 0.6 m, but away from the target drawer |

- **Bank wording steers where GR00T goes, not how it acts.** At GTB1 the stage's wording keeps it at the toaster (the
  task sentence drives it away), but it does not press the lever. No wording opened a drawer or grasped the
  kettle within 20 chunks.
- **The level does not matter much.** L0, L1 and L2 give the same outcomes; the slight expansion neither helps nor
  hurts.
- **The predicted-chunk ranking does not predict the outcome.** Differences between instructions are 0.01-0.03 m over
  one chunk; as in `../20261030_vlm_critic`, one chunk shows too little.

## Phase 0 on the tasks where chunk choice worked (2026-11-06)
`plain.py`, `phase0.sh` (plain GR00T, envkit, 30 scenes per task, the policy test's seeding; records in
`phase0/records/`, not in git), `phase0_summary.py`, `phase0_points.py` (`phase0/points.json`).
KettleBoiling, SteamInMicrowave and PackIdenticalLunches are the tasks where choosing among GR00T's own chunks with the
simulator raised success a lot (65 -> 96.7, 30.8 -> 90.0, 11.7 -> 63.3 % on the earlier pipeline), so good motions are
known to be among the samples there.

| task | success | failed episodes end in stage | stage durations when done (chunks: median / 90 %) |
|---|---|---|---|
| KettleBoiling | 20 / 30 | 0 (kettle to burner): 4, 1 (burner on): 6 | 18 / 39, 10 / 22 |
| SteamInMicrowave | 12 / 30 | 0 (vegetable to bowl): 6, 1 (bowl to microwave): 1, 2 (close door): 3, 3 (press start): 8 | 17 / 20, 28 / 36, 13 / 26, 14 / 24 |
| PackIdenticalLunches | 7 / 30 | 0: 4, 1: 2, 2: 10, 3: 7 (one item per stage) | 18 / 46, 32 / 133, 26 / 75, 34 / 45 |

**Failure points** (`phase0_points.py`): where an oracle supervisor would step in.
- *stalled*: the stage has lasted longer than the 90th percentile of its completions;
- *undone*: a stage that was done is no longer done (e.g. the vegetable fell out of the bowl);
- *late*: the stall point is after the episode's end.
KettleBoiling 10 (stalled 8, undone 2), SteamInMicrowave 18 (stalled 14, undone 4), PackIdenticalLunches 23 (stalled
20, late 3). About 43 have 20 or more chunks left to act.

## The recovery loop from known failure points (2026-11-06)
`loop.py` (one run), `loop.sh` (batch; several runners can share the queue through `.claim` files), `loop_score.py`;
runs in `loop/<arm>/<point>_p<seed>.json` (+ `.pkl.gz` with the chunks, for `trial_video.py`).
Detection is assumed solved: each run starts at a phase 0 failure point (exact replay of the plain run) and an oracle
supervisor (`stages.track`) checks after every chunk. Stuck = no completion within the budget (12 chunks after an
intervention; for later stages the 90th percentile of their phase 0 completions); an earlier stage undone for 3 chunk
ends in a row makes it current again. The run ends at the official success or the horizon.

| arm | ladder (next step each time the stage is stuck) |
|---|---|
| control | fresh samples once, then the policy alone |
| hold | the stage's sentence instead of the task sentence, kept until the stage is done |
| back | reverse the last 4 executed chunks + task sentence; again; reverse 8; again |
| full | hold; back 4 + stage sentence; back 4 + task; back 8 + stage; back 8 + task |

43 failure points (not late, 20 or more chunks left) x 2 policy seeds per arm, 344 runs, 0 errors.
Fixed = the failed stage got done; success = official success at the horizon.

| task | failed stage | points | control fixed / success | hold | back | full |
|---|---|---|---|---|---|---|
| KettleBoiling | 0 kettle to burner | 2 | 0/4 / 0/4 | 0/4 / 0/4 | 0/4 / 0/4 | 1/4 / 0/4 |
| KettleBoiling | 1 burner on | 5 | 2/10 / 2/10 | 1/10 / 1/10 | 1/10 / 1/10 | 2/10 / 2/10 |
| PackIdenticalLunches | 0 | 4 | 3/8 / 0/8 | 2/8 / 0/8 | 1/8 / 0/8 | 1/8 / 0/8 |
| PackIdenticalLunches | 1 | 2 | 0/4 / 0/4 | 2/4 / 0/4 | 0/4 / 0/4 | 0/4 / 0/4 |
| PackIdenticalLunches | 2 | 8 | 3/16 / 0/16 | 2/16 / 0/16 | 0/16 / 0/16 | 0/16 / 0/16 |
| PackIdenticalLunches | 3 | 5 | 0/10 / 0/10 | 0/10 / 0/10 | 0/10 / 0/10 | 0/10 / 0/10 |
| SteamInMicrowave | 0 vegetable to bowl | 6 | 0/12 / 0/12 | 3/12 / 1/12 | 3/12 / 0/12 | 2/12 / 1/12 |
| SteamInMicrowave | 1 bowl to microwave | 1 | 0/2 / 0/2 | 0/2 / 0/2 | 0/2 / 0/2 | 0/2 / 0/2 |
| SteamInMicrowave | 2 close door | 3 | 2/6 / 0/6 | 3/6 / 0/6 | 2/6 / 0/6 | 5/6 / 2/6 |
| SteamInMicrowave | 3 press start | 7 | 0/14 / 0/14 | 1/14 / 1/14 | 1/14 / 1/14 | 2/14 / 2/14 |
| **all** | | 43 | **10/86 / 2/86** | **14/86 / 3/86** | **8/86 / 2/86** | **13/86 / 7/86** |

Paired per failure point (fixed in more / fewer seeds than control; sign test):
- all tasks: hold better at 8, worse at 5; back 5 / 5; full 6 / 5. No arm differs from control.
- SteamInMicrowave: hold 5 / 0 (p 0.06), back 4 / 0, full 4 / 0 (p 0.12): a consistent lean, not significant.
- PackIdenticalLunches: back 0 / 4 and full 0 / 4: reversing hurts there.
- In full, the last intervention before a fix: hold 3, back 4 + stage 4, back 4 + task 4, back 8: 2.

**What the robot does when it stays stuck** (most runs stay in the failed stage to the horizon; `trial_video.py` on
`loop/hold/*`, sheets in `loop/video/`, not in git):
- KettleBoiling s26, burner on: the hand goes to the knobs (the wrist camera shows them close) and never turns one.
- SteamInMicrowave s6, press start: the hand hovers at the panel (the wrist camera shows the Start key) and never
  presses it.
- PackIdenticalLunches s6, last item: all four items are in the tupperwares, but one fish lies on top of a vegetable
  and does not touch the tupperware, so `check_obj_in_receptacle` (contact + centre within 0.7 of the radius) does not
  count it. The robot hovers and later goes back to the fridge. s2 and s18 are dead ends at the failure point (two of
  a kind in one tupperware), s24 becomes one.
The stage sentence and the backing-off move the robot to the right place, as in the steerability probe; the last
motion (turn, press, put the item fully in) is what fails, and a retry from almost the same state fails the same way.
PackIdenticalLunches s11 (stage 2): the arm is stuck inside the fridge in an odd pose (the wrist camera sees only the
arm itself) and stays there for over 100 chunks.

**State or scene?** (`stage_start.py`, `stage_start.sh`; `stage_start/`). The same scene, restored exactly to the
moment the failed stage began (phase 0 replayed up to there), with fresh samples and the task sentence: the stage is
done within its stall window in **64/172 = 37 %** (4 seeds x 43 points), against **5/86 = 6 %** from the failure point
(control, the same window). The failure state, not the scene, is most of the problem.

| failed stage | from the stage start | from the failure point (control) |
|---|---|---|
| KettleBoiling 0 / 1 | 2/8 / 8/20 | 0/4 / 0/10 |
| PackIdenticalLunches 0 / 1 / 2 / 3 | 6/16 / 3/8 / 12/32 / 0/20 | 1/8 / 0/4 / 2/16 / 0/10 |
| SteamInMicrowave 0 / 1 / 2 / 3 | 19/24 / 2/4 / 8/12 / 4/28 | 0/12 / 0/2 / 2/6 / 0/14 |

Caveats: this restores the objects too (a simulator-only restore); for a stage 0 point the stage start is the episode
start (the SteamInMicrowave "undone" points: a full restart with the vegetable back in the sink). The last item of
PackIdenticalLunches and the start button stay hard even from the stage start (0 % and 14 %).

**Reversing chunks is not a return** (`check_reverse.py`, on `loop/back` records). After a "back" the hand is often
2-7 cm from where it was, but the arm joints differ by 0.1-1.4 rad; where the arm was blocked it goes elsewhere:
PackIdenticalLunches s11, hand moved 0.6 cm in the last 4 chunks, 31 cm away after reversing them (the commanded
deltas were large but blocked; negated, they move freely); a back 8 there drove the base 1.2 m away.

## Retrace to the stage start (the stage as the unit, 2026-11-06)
The design (the user's): the stage is the unit; the supervisor remembers where each stage began; when the stage is
stuck the robot goes back to that point the way a person redoes a step, by retracing the path it took, and GR00T
tries the stage again. Nothing in the world is restored: what the robot carried goes back with it and is put down
where it was picked up; what slid or fell stays where it is.

**`retrace.py` (v2, joint space, untimed).** A `Recorder` keeps the robot's measured state after every env step: the
7 arm joints, the base (x, y, yaw), the gripper command. To go back, the entries are played backwards at the
original speed (one entry per step; paused only if the arm lags by more than 0.1 rad): the arm by joint-space
impedance toward the entry's joints (in place of the arm's OSC while retracing; the OSC goal is then set to the
reached pose), the base by velocity commands toward the entry's pose, measured at every step, and the gripper with
the command in effect at that entry, so it opens where it had closed (holding still 8 steps before and after each
change). Only loops back to practically the same state (joints within 0.05 rad, base within 2 cm) and standing still
are skipped; there are no shortcuts. **Its time is not counted** (a restart): the env runs with a free horizon,
GR00T keeps the official horizon on its own time, and success is the official success at a chunk end within it (a
success reached while retracing counts and is flagged).

Why v2 (the user, 2026-11-06: "reverse 回去而已", and the retrace time should not count). v1 tracked the hand pose with
shortcuts and counted its time: of its 325 retraces 121 did not arrive, 40 because the episode ended during the
retrace and 30 at its 30-chunk cap (time), and 51 stopped short with the base in place but the hand 8 cm / 0.44 rad
off: tracking only the hand pose let the redundant arm fold into another configuration and stop at a joint limit
(SteamInMicrowave s4: joint 4 at its limit, joint 2 from -1.05 to +1.52). A 2x playback speed shook a rim-held bowl
loose; one entry per step does not.

Conventions measured first (16-step probes): hand delta 5 cm and 0.5 rad per unit in the base frame; in arm mode the
goal is the achieved pose + delta, in base mode the goal accumulates; base 3.0 / 2.9 cm and 0.066 rad per unit and
step; gripper 1 = close; the torso is never used.

**Checks** (`check_retrace.py`: from the failure point back to the failed stage's start, untimed; live frames in
`loop/video/retrace/<id>_live.mp4`, since a retrace cannot be replayed from the actions):

| point | path since the stage began | retrace (chunks) | end error (joints / base) | objects |
|---|---|---|---|---|
| KettleBoiling s26 | 368 steps | 12 | 0.001 rad / 0.0 cm | kettle 4.1 cm from its start (3.8 at the failure) |
| KettleBoiling s9 | 368 steps, 7 gripper changes | 23 | 0.001 rad / 0.1 cm | none moved |
| KettleBoiling s2 | 640 steps (from the episode start) | 62 | 0.015 rad / 0.0 cm | none moved (v1: stuck at a joint limit) |
| SteamInMicrowave s4 | 480 steps | 28 | 0.002 rad / 0.0 cm | bowl and vegetable had slipped out before the failure; not taken back (v1: stuck at a joint limit) |
| SteamInMicrowave s14 | 672 steps, holding the bowl | 29 | 0.001 rad / 0.0 cm | bowl carried back but released 50 cm from its start (a rim grasp); the vegetable had fallen 2 m away |
| PackIdenticalLunches s27 | 2144 steps, holding meat1, 9 gripper changes | 145 | 0.001 rad / 0.2 cm | meat1 back to 9 cm of its start (was 1.2 m away) |
| PackIdenticalLunches s20 | 752 steps | 115 | 0.001 rad / 0.0 cm | items within 1.1 cm of their starts |

**Undone rule for this arm.** SteamInMicrowave stage 1 ("bowl in the microwave") includes "the gripper is 0.25 m from
the bowl", and KettleBoiling stage 0 "the gripper is 0.25 m from the kettle". Pressing start or turning a knob brings
the hand that close, so the stage reads as undone (s18: a reset back to stage 1, 5 m away, while the bowl stayed in the
microwave). An earlier stage now counts as undone only at chunk ends where the hand is clear of every object.

**A teleport was tried first and dropped.** `loop.restore_robot` puts only the robot (joints, controllers) back to a
snapshot, exactly (`check_restore_robot.py`: a no-op reset and a reset plus the objects of A are bit-identical to the
continuous replay; a `forward()` that overwrote MuJoCo's `qacc_warmstart` was fixed). As a recovery it is not
realistic: a held item stays in the air and falls, and the time to get back is not paid. Its 24 runs were discarded.

**Results, v2** (43 failure points x 4 policy seeds, 0 errors; v1 runs moved to `loop_archive/retrace_v1_handpose_timed/`):

| task | failed stage | points | control fixed / success | retrace v1 | **retrace v2** |
|---|---|---|---|---|---|
| KettleBoiling | 0 kettle to burner | 2 | 0/8 / 0/8 | 1/8 / 1/8 | 3/8 / 3/8 |
| KettleBoiling | 1 burner on | 5 | 4/20 / 4/20 | 13/20 / 13/20 | 9/20 / 9/20 |
| PackIdenticalLunches | 0 | 4 | 3/16 / 0/16 | 6/16 / 0/16 | 12/16 / 1/16 |
| PackIdenticalLunches | 1 | 2 | 0/8 / 0/8 | 0/8 / 0/8 | 2/8 / 0/8 |
| PackIdenticalLunches | 2 | 8 | 5/32 / 0/32 | 1/32 / 1/32 | 6/32 / 0/32 |
| PackIdenticalLunches | 3 | 5 | 1/20 / 1/20 | 2/20 / 2/20 | 2/20 / 2/20 |
| SteamInMicrowave | 0 vegetable to bowl | 6 | 1/24 / 0/24 | 2/24 / 1/24 | 7/24 / 4/24 |
| SteamInMicrowave | 1 bowl to microwave | 1 | 0/4 / 0/4 | 0/4 / 0/4 | 0/4 / 0/4 |
| SteamInMicrowave | 2 close door | 3 | 5/12 / 1/12 | 10/12 / 5/12 | 12/12 / 8/12 |
| SteamInMicrowave | 3 press start | 7 | 1/28 / 1/28 | 10/28 / 10/28 | 11/28 / 11/28 |
| **all** | | 43 | **20 / 7** (12 % / 4 %) | 45 / 33 (26 % / 19 %) | **64 / 38** (37 % / 22 %) |

Paired per failure point against control (sign test): fixed better at 23 points, worse at 1 (p < 0.0001); success
better at 18, worse at 0 (p < 0.0001). By task, fixed / success: KettleBoiling 6 / 0 and 6 / 0, SteamInMicrowave 9 / 0
and 10 / 0, PackIdenticalLunches 8 / 1 and 2 / 0.

- 410 retraces: 402 arrived (joints within 0.03 rad of the stage start), median 23 chunks (p90 63; untimed). No success
  happened while retracing; 6 undone events.
- Why a retrace does not arrive (PackIdenticalLunches s4, stage 1, its first retrace, the same for every seed; checked
  2026-11-06 by a scratch replay with per-step contacts and torques): during the stage GR00T kept the arm inside the
  open fridge and pressed against it (robot-fridge contact at 1533 of 2145 steps, 50-840 N, joint 1 at its 80 N m
  limit). Going back, 21 chunks in (entry 2092 of the path), link 5 is wedged against the fridge's inner panel
  (`fridge_main_group_g9`, 574 N): joints 1, 2, 4 push at -70, -77, +68 N m (limit 80), joint 4 stays 0.14-0.18 rad
  from its target, the base sits 1.3-1.9 cm off, and the retrace waits for the arm until its 400-chunk cap. A path
  that was forced through a contact cannot always be played backwards: contact and friction are not reversible.
  There is no stop for a stall in the middle of the path (only at its last entry).
- 47 of the 64 fixes came after the first retrace (from the exact stage start, objects restored, 37 % in the window;
  one retrace: 47/172 = 27 %).
- **PackIdenticalLunches: the stage gets fixed (22/76) but the task rarely succeeds (3/76).** After the fix the run
  stops in a later stage, often with time left (s14: fixed after 24 policy chunks with 173 left, ends in stage 1 or 2).
  A later stage is called stuck only after its phase 0 p90 duration (stage 1: 133 chunks, stage 2: 75), so it gets one
  more try at most. The stuck detection (a timeout), not the retrace, limits this task.

**Results, v1** (hand-pose retrace with shortcuts, its time counted; 43 failure points x 4 policy seeds; control re-run with seeds 2-3 for the pairing; 0 errors):

| task | failed stage | points | control fixed / success | retrace fixed / success |
|---|---|---|---|---|
| KettleBoiling | 0 kettle to burner | 2 | 0/8 / 0/8 | 1/8 / 1/8 |
| KettleBoiling | 1 burner on | 5 | 4/20 / 4/20 | **13/20 / 13/20** |
| PackIdenticalLunches | 0 | 4 | 3/16 / 0/16 | 6/16 / 0/16 |
| PackIdenticalLunches | 1 | 2 | 0/8 / 0/8 | 0/8 / 0/8 |
| PackIdenticalLunches | 2 | 8 | 5/32 / 0/32 | 1/32 / 1/32 |
| PackIdenticalLunches | 3 | 5 | 1/20 / 1/20 | 2/20 / 2/20 |
| SteamInMicrowave | 0 vegetable to bowl | 6 | 1/24 / 0/24 | 2/24 / 1/24 |
| SteamInMicrowave | 1 bowl to microwave | 1 | 0/4 / 0/4 | 0/4 / 0/4 |
| SteamInMicrowave | 2 close door | 3 | 5/12 / 1/12 | **10/12 / 5/12** |
| SteamInMicrowave | 3 press start | 7 | 1/28 / 1/28 | **10/28 / 10/28** |
| **all** | | 43 | **20/172 / 7/172** (12 % / 4 %) | **45/172 / 33/172** (26 % / 19 %) |

Paired per failure point (sign test): success better at 14 points, worse at 0 (p = 0.0001); fixed better at 15,
worse at 4 (p = 0.02). By task: KettleBoiling 5 / 0 and SteamInMicrowave 7 / 0 on both; PackIdenticalLunches fixed
3 / 4, success 2 / 0. It is the first intervention here that beats control; the earlier arms (hold, back, full) do not
(with the 4-seed control: fixed better / worse 8 / 9, 6 / 8, 7 / 8).

- 36 of the 45 fixes came after the first retrace. 325 retraces: 204 arrived (63 %), median 13 chunks, 30 hit the
  30-chunk cap; 265 included gripper changes. Undone events: 2 (the hand-clear rule).
- From the exact stage start (objects restored, a simulator-only diagnostic) the stage gets done in its window 37 %
  of the time; one retrace gets 36/172 = 21 % without restoring anything.
- PackIdenticalLunches gains nothing on "fixed": only 70 of its 143 retraces arrive (median 19 chunks, 30 at the cap);
  a run spends a median 30.5 of its 109 remaining chunks retracing; items put down elsewhere or dropped stay there.

**Arrival is what matters.** The first retrace from a point is the same for every policy seed (the path is recorded
and the retrace is deterministic), so the points split cleanly:

| first retrace | points | exact stage-start restore (diagnostic) | retrace: failed stage fixed |
|---|---|---|---|
| arrives | 29 | 44/116 | 43/116 |
| does not arrive | 14 | 20/56 | 2/56 |

When the robot gets back, retracing is as good as restoring the whole world. The gap to the 37 % ceiling is entirely the
points where it does not get back. Why it does not (first retraces; per-chunk error, contacts and joints at the end
from a scratch replay of 9 points):
- **30-chunk cap on long walks** (PackIdenticalLunches s2, s4, s12, s25, s27): 1200-2144 recorded steps, the base still
  0.9-1.7 m from the stage start.
- **The arm caught on a fixture**: KettleBoiling s2 (link 5 on the range hood, 17 cm short after 28 chunks, creeping
  0.4 cm per chunk, just over the no-progress limit); SteamInMicrowave s4 (link 4 and the gripper on the sink);
  PackIdenticalLunches s15 (link 5 on the fridge door, which the stage had opened; 3 cm short).
- **Joint limits**: the arm reaches the recorded hand poses in another joint configuration (the arm has 7 joints for a
  6-D pose, and the controller's null space drifts), and a joint ends at its limit: KettleBoiling s2 and
  SteamInMicrowave s4 (joint 4), SteamInMicrowave s17 (joint 6; 14 cm, 0.8 rad short), PackIdenticalLunches s23
  (joints 1, 5, 6, 7; 20 cm, 0.7 rad short).
- **Near misses** (2.5-3 cm short; KettleBoiling s5 with the kettle against arm links 1-2, PackIdenticalLunches s15):
  0 of 12 fixed, so these points look hard for another reason, not because of the last 3 cm.

## Detectors
- **Oracle events** (`../../envkit/events.py`; pure reads, tested): missed_grasp, dropped, fixture_short, wrong_element.
  On 12 recorded runs they fire where expected: lever pressed to 0.27-0.75 and released; drawer left at 0.39; a
  cabinet door or a neighbouring drawer moved.
- **Rule limits.** The rules miss failures they were not written for:
  - KettleBoiling s2 closes the gripper on air 0.24-0.47 m from the kettle's centre, never within the "near" radius;
  - DeliverStraw s0 never touches the drawer and wanders for the whole episode.
- **VLM interface** (`../../coordinator/vlm/pilot.py`): one query = the task instruction, the stage sentences, and a clip (4 moments
  x 3 cameras). There is one fixed JSON answer: stages done, event type, evidence, confidence. `PacketVLM` runs Claude
  (subagents) offline; API backends implement the same `answer()`.

## VLM pilot (Claude as the VLM, blind)
`make_vlm_pilot.py`, `score_vlm_pilot.py`; packets and answers in `pilot_vlm/`.
- **Data.** 6 recorded envkit runs (KettleBoiling s0, s2; SetUpCuttingStation s0; DeliverStraw s0, s1;
  GetToastedBread s0) give 60 queries, one every 8 chunks up to chunk 80. Each query shows the last 4 chunk ends, 2
  chunks apart, from 3 cameras.
- **Who answered.** Subagents that saw only the packets, never the ground truth. Packet ids are hashes, so each query
  is judged on its own.
- **Stage flags vs `stages.track`:** 117 / 120 correct. Every "done" was recognised. The 3 errors are false "done" on
  the toaster lever: it was pressed to 0.75 (threshold 0.90) and sprang back, and "pressed" looked the same as
  "pressed far enough".
- **Failure flags vs outcome.** A window counts as "stuck" when the current stage does not complete in the next 16
  chunks.
  - 35 of 48 stuck windows were flagged (recall 0.73).
  - 2 of 12 progressing windows were flagged, both after the task had already succeeded (the robot pushed a pot into
    the kettle).
  - Most misses are early windows, before anything had visibly gone wrong.
- **Detection time:**
  - Air grasps (KettleBoiling s2): the first window.
  - Wandering, drawer never touched (DeliverStraw s0): 12 s.
  - Drawer left half-open (DeliverStraw s1): the same window as the oracle.
  - Missed grasps of the steak (SetUpCuttingStation s0): before the oracle.
  - Toaster lever short (GetToastedBread s0): about 20 s after the oracle, as "no progress".
- **Event types** are coarse. "no_progress" dominates, with some missed_grasp, fixture_short and wrong_element.
- **A prompt error, fixed.** Stage sentences that name the current scene ("the left burner") were first made at t = 0
  and named the wrong burner. That gave 3 false alarms and 3 missed "done". They are now made at query time; the
  first answers are kept in `pilot_vlm/v1_kettle_s0/`.
- **Cost.** About 10-17 min and 160k tokens per 30 queries with subagents, so this works offline only. A live closed
  loop needs an API backend.

## Which VLM (2026-11-05)
`compare_vlm_models.py` asks each model the same 60 pilot queries: the panel image and the prompt the subagents saw.
Answers are in `pilot_vlm/answers_<model>.jsonl`. "False done" is the dangerous error: the controller would move on
from a stage that is not done.

| model | stage flags correct | false "done" | missed "done" | stuck windows flagged | progressing windows flagged | median latency | cost / query |
|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 (effort medium, the default) | 0.950 | **0** | 6 | 0.79 | **1 / 12** | 4.8 s | $0.0155 |
| Claude Opus 5.5, effort low | 0.933 | 1 | 7 | 0.85 | 3 / 12 | 3.4 s | $0.0124 |
| Claude Sonnet 5.5 | 0.925 | 1 | 8 | 0.83 | 6 / 12 | 3.4 s | $0.0069 |
| Claude Haiku 5.5 | 0.742 | 21 | 10 | 0.48 | 3 / 12 | 3.0 s | $0.0004 |
| Qwen3.5-9B (local, transformers, GPU1, no thinking) | 0.900 | 9 | 3 | 0.90 | 4 / 12 | 9.9 s | 0 |
| Qwen3-VL-8B (local, transformers, shared 3090, no thinking) | 0.825 | 13 | 8 | 0.98 | 9 / 12 | 7.1 s | 0 |
| Qwen3.8-27B-FP8 (local, vLLM, GPU1, no thinking) | 0.700 | 32 | 4 | 0.67 | 5 / 12 | 21.7 s | 0 |

- **Opus 5.5 is the only model with no false "done".** Its 6 missed "done" are all KettleBoiling after success, where a
  pot crowds the kettle on the burner. Effort low saves 20 % and adds errors, so keep medium. Sonnet flags half the
  progressing windows. Haiku calls drawers and levers done when they are not.
- **Why the 27B first looked worse than the 9B (checked 2026-11-05).** Each row below is a separate run of the same 60
  queries, all through the gemma4 env's vLLM, no thinking, temperature 0.

  | run | Qwen3.8-27B-FP8 | Qwen3.5-9B |
  |---|---|---|
  | panel image, JSON schema enforced | 0.700 (false done 32) | 0.829 (14) |
  | panel image, free JSON | 0.667 (36) | 0.883 (11) |
  | panel image, free JSON, Marlin FP8 kernel (weight-only) | 0.683 (36) | |
  | 12 frames interleaved with their labels, free JSON | **0.833 (16)** | 0.792 (20) |

  - **Ruled out:**
    - truncation: 108-179 output tokens, all ended normally;
    - vision weights quantized: vLLM keeps all 111 vision linear layers in bf16;
    - image preprocessing, prompt and sampling: identical to the 9B; no penalties in generation_config;
    - the FP8 matmul kernel: Triton W8A8 and Marlin give the same answers on 53 of 60.
  - **Found:**
    - the 4 x 3 panel confuses the 27B: interleaved frames lift it from 0.68 to 0.83, while the 9B prefers the panel;
    - vLLM's enforced JSON schema costs the 9B 5 points, consistent with vLLM issue #35700.
  - **Not tested:** the bf16 27B, which needs about 60 GB on one GPU, and thinking mode. With each model's better
    format the two are within noise (about +-0.06 on 120 flags), and both are well below Opus 5.5.
  - Earlier in this section, the 0.900 for the 9B is the transformers server (`vlm_server.py`, panel). Its "vLLM" row
    first reused those answers because the answer files had the same name. Runs now have their own files
    (`answers_vllm_*`).
- **The local models answered without thinking; the Claude models think before answering.**
  - With thinking on, Qwen3.8-27B-FP8 ran at about 7 tokens/s on the RTX 6000 Ada (vLLM 0.22 dev in the gemma4 env,
    eager mode). One answer took more than 10 min, so it was not run.
  - The bf16 27B (52 GB) fits only GPU0 / GPU3, which were busy.
- **Serving notes:**
  - `../../coordinator/vlm/vlm_server.py` (transformers, mm_bak env) serves Qwen3.5-9B and Qwen3-VL-8B. The FP8 27B needs the
    `kernels` package there.
  - The gemma4 env's vLLM serves Qwen3.8-27B-FP8 (`vlm.VLLMVLM`). It needs the env's `bin` on PATH (ninja), its
    `lib` on LD_LIBRARY_PATH (libstdc++), and `VLLM_USE_FLASHINFER_SAMPLER=0` (no curand headers for the JIT sampler).

## Files
| file | what |
|---|---|
| `make_vlm_pilot.py` | query packets and ground truth from recorded runs |
| `score_vlm_pilot.py` | scores answers against the ground truth |
| `compare_vlm_models.py` | asks the same queries of API and local models (`vlm.ClaudeVLM`, `LocalVLM`, `VLLMVLM`) and scores them |
| `recover.py`, `recover.sh`, `recover_restart.sh` | recovery trials from known failure points; `recover/results.jsonl` |
| `steer_instructions.py`, `steer_probe.py`, `steer_probe.sh`, `steer_score.py` | steerability probe: candidate instructions, rollouts, scores (`steer/`) |
| `steer_bank.py`, `steer_gen.py`, `steer_rank.py`, `steer_v2.sh` | steer v2: phrase bank, leveled instructions, ranking from predicted chunks (`steer/v2/`) |
| `plain.py`, `phase0.sh`, `phase0_summary.py`, `phase0_points.py` | phase 0: plain runs, summary, failure points (`phase0/`) |
| `loop.py`, `loop.sh`, `loop_score.py` | the recovery loop from known failure points: arms control / hold / back / full (`loop/`) |
| `stage_start.py`, `stage_start.sh` | state or scene: the failed stage again from its start, fresh samples (`stage_start/`) |
| `check_reverse.py` | does reversing chunks bring the robot back (on `loop/back` records) |
| `check_restore_robot.py` | checks `loop.restore_robot`: no-op and completeness, bit-identical replays |
| `retrace.py`, `check_retrace.py` | retrace to a stage start along the recorded path; its check (`loop/retrace_check/`) |
| `inspect_trial.py`, `trial_video.py` | replay a trial record (`recover/` or `loop/`): fixture state per chunk (`recover/inspect/`), camera frames (`recover/video/`, `loop/video/`) |
| `pilot_vlm/` | `packets/<qid>/` (prompt.md, answer.json; panel.png is not in git), `truth.jsonl`, `scored.jsonl`, `v1_kettle_s0/` |
