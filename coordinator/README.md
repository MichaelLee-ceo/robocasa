# coordinator: VLM supervision around a frozen GR00T

This is the method of the ECE 780 proposal *Knowing When It's Stuck: VLM Supervision for Progress Tracking and Failure
Recovery of Frozen VLA Executors on Long-Horizon Tasks* (`../ECE780_Proposal.pdf`, Figure 1). GR00T N1.5 stays frozen
and performs every action. The coordinator keeps a memory and decides when and how to step in.

Status:
- **Works, verified:** every slot has a working oracle implementation, and the mechanisms are done: the loop, the
  memory, the retrace, the VLM backends, and turning the memory into VLM requests. The oracle implementations read the
  simulator; they are the proposal's upper-bound ablation.
- **Runs, not yet evaluated:** the VLM modules (tracker, confirmer, recovery) are reference versions. Their prompts and
  thresholds still need tuning and scoring against the oracle.
- **Not written yet:** the proprio triggers and the back-off motion.

The working configuration `oracle_retrace` is the method behind the current result (`../runs/20261105_recovery`,
"Retrace to the stage start"). It was compared with same-scene reruns over 43 failure points x 4 seeds:

| | failed stage fixed | task success |
|---|---|---|
| same-scene reruns | 12 % | 4 % |
| `oracle_retrace` | 37 % | 22 % |

Both differences are significant in paired tests (p < 0.0001).

```
          ┌──────────────────────── Coordinator (loop.py) ─────────────────────────┐
          │  tracking/            detection/                 recovery/              │
          │  "where am I?"        triggers + confirmer       "what now?"            │
          │  StageStatus          "is it a failure?"         Intervention           │
          │        ▲                    ▲                          ▲                 │
          │        └──── context.py: memory -> VLM request ────────┘                 │
          │                          ▲            │ vlm/: Claude API, OpenAI-compatible │
          │   memory.py (M_t): plan, attempts, keyframes, frames, robot trace, notes  │
          └───────────┬──────────────────────────────────────────▲───────────────────┘
      sentence + seed │ or motion/retrace.py drives               │ Obs: 3 views, proprio,
      (only if needed)│ the robot back                            │ GR00T's own time
          ┌───────────▼──────────────┐   16-step chunk   ┌────────┴──────────────┐
          │ executor.py: frozen GR00T│ ────────────────► │ kitchen scene (envkit) │
          └──────────────────────────┘ ◄──────────────── └───────────────────────┘
```

## Layout
| path | what | status |
|---|---|---|
| `schema.py` | the records passed between modules: `Obs`, `SubGoal`, `StageStatus`, `Trigger`, `Intervention` | done |
| `memory.py` | `Memory`, the proposal's M_t (table below) | done |
| `context.py` | memory -> VLM request, one builder per decision (progress, confirm, recovery, plan); `ContextConfig` | done |
| `prompts.py` | the system prompt, the questions, the answer schemas: the pilot's validated system prompt, progress question, schema and event definitions (`coordinator/vlm/pilot.py`) unchanged; the confirm, recovery and plan questions are new | done |
| `loop.py` | `Coordinator`: the loop, the clock, the success rule, executing interventions | done |
| `configs.py` | named configurations (which implementation fills each slot) and `build()` | done |
| `executor.py` | GR00T through the envkit client; the sentence and noise stream come from the memory | done |
| `paths.py` | repository paths; makes `envkit` (top level) importable | done |
| `points.py` | runs from the phase 0 failure points exactly as loop.py ran them (replay, start trigger, loop.py's rule for "fixed") | done |
| `vlm/` | backends behind one interface (below): subclasses of the pilot's `ClaudeVLM`, `VLLMVLM`, `LocalVLM` (`coordinator/vlm/pilot.py`, loaded by `vlm/pilot.py`) that take any request; `MockVLM`; `make_vlm(spec)` | done, tested |
| `tracking/` | `OracleTracker` (done); `VLMTracker` (reference v0) | |
| `detection/` | `StageTimeout`, `Undone`, `OracleEvents`, `AcceptAll` (done); `VLMConfirmer`, `VLMEvent` (reference v0); `EmptyGrasp`, `Stationary` (**TODO**) | |
| `recovery/` | `RetraceRetry`, `Ladder` (done); `unstick.py`: `VLMUnstick`, `FixedUnstick` (done; a blocked retrace); `VLMRecovery` (reference v0) | |
| `motion/` | `retrace.py`: `RobotTrace`, `retrace()`, final version (done, checked on 47 failure points); `backoff.py` (**TODO**) | |
| `gr00t_wording.py` | the sentences GR00T was trained on, as a phrase bank, and the rules for instructions written from it | done |
| `scripts/` | `run.py` (one run), `batch.sh` (failure points x seeds), `watch_batch.py` (watchdog for a batch: deletes error results and re-runs them, stops runs over a time limit, status file, valid-only results and a summary against loop.py at the end; start it with setsid), `eval_tracker.py` / `score_tracker.py` (the VLM tracker on recorded runs), `vlm_check.py` (test a backend; see what the VLM is sent) | done |
| `tests/` | `test_offline.py` (retrace planning, memory, timeout, undone, policies, robot state), `test_vlm.py` (backends against stub servers, registry, answer checks, request building, VLM modules with a mock), `check_retrace_state.py` (does the retrace follow the path and restore the robot's state), `check_retrace_fallback.py` (fallbacks for a blocked retrace) | pass |

The environment layer is `../envkit`, at the top level too (verified on 480 scenes; usage guide `../envkit/README.md`).
The recorded failure points are in `../runs/20261105_recovery/phase0`. Outputs go to `../runs/20261106_coordinator`.

## Relation to the original code
The coordinator is the code of `runs/20261105_recovery` (`loop.py`, `retrace.py`) and the pilot's VLM module,
reorganized into modules and then extended.

| original | here | how |
|---|---|---|
| `20261105_recovery/retrace.py` (v2) | `motion/retrace.py` | first copied; replaced by the final version on 2026-11-07 (below) |
| `loop.py` `stall_limits`, `hand_clear` | `detection/timeout.py` `phase0_windows`, `tracking/oracle.py` `hand_clear` | copied unchanged |
| `loop.py` `run()`, arm `retrace` | `loop.py`, `memory.py`, `points.py`, `detection/undone.py`, `detection/timeout.py`, `recovery/policies.py` | restructured into modules. Kept from loop.py: `starts` (each stage's entry: replay and moving on, every passed stage) and `back_to` (that entry, or the arrival point of the last retrace); undone handled before stuck; the free retrace time and the success rule; the rule for "fixed" (moved on from the failed stage, else done or success at the end) |
| `loop.py` noise streams (`loop/<point>/p<seed>`, `start`, `s<k>`, `a<k>.<n>`) | `executor.py`, `memory.stream` | the same names, so the same seed gives the same GR00T noise |
| `loop.sh` queue (order, shards, ports) | `scripts/batch.sh` | byte-identical lines, so a run lands on the same GPU and GR00T server as loop.py's |
| `envkit/vlm.py` (the pilot's backends, prompt, `Query`) | `vlm/pilot.py` | moved unchanged; the backends in `vlm/` subclass its classes, and the requests extend its `Query` |
| `envkit/vlm_server.py` | `vlm/vlm_server.py` | moved unchanged |
| `runs/envkit` | `../envkit` (top level) | moved; only the paths that found the repository root changed |

**The restructured loop was proven equal to loop.py first** (2026-11-06; `tests/check_equivalence.py` at commit
97a6269 on branch `ece780-coordinator`; records in `../runs/20261106_coordinator/equivalence*.jsonl`):
- the check fed the coordinator loop.py's recorded GR00T chunks and compared every GR00T request (env chunk, noise
  stream, instruction), every retrace (chunk, length) and the outcome (success and its time, "fixed", "fixed_after");
- two independent reviews found four rare paths that differed (the retrace target after an undone stage; the order of
  undone and stuck; the entries of skipped stages; the end-of-run check for "fixed"); all four were changed back;
  then 18 of 18 runs checked were equal (before the fixes, 94 of 95);
- live, with matched seeds, GPUs and servers (seeds 4-7, 23 pairs): GR00T's inputs and outputs bit-identical until a
  camera image differed by MSAA noise; "fixed" agreed in 23 of 23 pairs, success in 21 of 23.

**Then the retrace was replaced by the final version (2026-11-07, the user's decision; one version kept):** it also
restores the torso, the OSC goal and rest, follows the path more tightly, and stops when the arm is blocked, which
starts the unstick fallback (loop below). From here on the coordinator's runs differ from loop.py's by design; the v2
results stay in `../runs/20261105_recovery`, the comparison of the versions in `../runs/20261106_coordinator/README.md`.

## The loop (loop.py)
At every chunk end:
1. **observe**: `memory.observe(Obs)`: the three views GR00T sees, its proprio, GR00T's own time, the trace entry.
2. **track** (every `track_every` chunks: 1 for the oracle, 4 for a VLM): `tracker.update -> StageStatus`.
   - Sub-goals that became done are closed, and the next sub-goal begins.
   - `done = None` means no answer this time; the coordinator keeps its current view.
3. **detect**: each detector's `check(memory, obs, status) -> [Trigger]`. `status` is None on chunks without
   tracking.
4. **confirm**: `confirm(memory, obs, trigger) -> (ok, reason)`. Every trigger is logged, accepted or not. An accepted
   `undone` makes that sub-goal current again.
5. **recover**: for the first accepted trigger:
   - `recovery.decide(memory, obs, trigger) -> Intervention`;
   - the attempt it ends is closed with how it ended (e.g. `timeout -> retrace`);
   - `execute` carries it out.
6. **act**: GR00T's next chunk.

The run ends in three ways:
- at the official success: at a chunk end, within the official horizon, counted on GR00T's own time;
- when GR00T's time reaches the horizon;
- when the env ends.

Retrace steps are not GR00T's time: a retrace stands for a restart (decided 2026-11-06). The env runs with a free
horizon. A success reached while retracing counts, and is flagged.

**Retrace and unstick** (`execute`, kind `retrace`): `motion/retrace.py` rewinds the robot's trace to where the
sub-goal's attempt began: the trace is a stack of the states of every part GR00T moves (arm joints and speeds, base,
torso, gripper command, OSC goal), popped one entry per step at the forward speed; on arrival the arm is held until
at rest and the OSC goal is restored, so GR00T starts again from the state it had (47 failure points: 45 back, hand
within 0.53 mm median; `../runs/20261106_coordinator/README.md`). If the robot is blocked (it lags its entry far and
the lag does not shrink while the playback waits, e.g. an arm wedged in a fridge GR00T had pushed it into), the unstick
fallback runs (`_unstick`, at most `unstick_tries` = 2 times): `recovery/unstick.py` chooses one sentence in GR00T's
wording (the VLM, when the run has one: it sees the views now and at the sub-goal's start; else the task sentence),
GR00T runs `unstick_chunks` = 8 chunks with it (their time not counted, as part of going back), and the retrace
resumes along the path recorded before it began, from the last entry at which the hand was 0.25 m or more from where
it got stuck. At PackIdenticalLunches s4 the 27B wrote "Close the fridge door.", which freed the arm 4 of 4 times with
the earlier retrace; with the final one the block is nearer the start and resuming runs into the fridge again (open).

## Contracts
| slot | method | returns |
|---|---|---|
| tracker | `make_plan(instruction, obs)`, `update(memory, obs)` | `[SubGoal]`, `StageStatus` (done per sub-goal, current, confidence, evidence, hand_clear, note, error) |
| detector | `check(memory, obs, status)` | `[Trigger(kind, stage, chunk, detail)]` |
| confirmer | `confirm(memory, obs, trigger)` | `(bool, reason)` |
| recovery | `decide(memory, obs, trigger)` | `Intervention(kind, stage, instruction, reason)`; executable kinds: continue, resample, retrace, reinstruct, stop (+ backoff once written) |
| VLM backend | `ask(VLMRequest) -> VLMResponse`, `ask_many([...])` | `data` (parsed JSON), `error`, `seconds`, tokens, `cost_usd` |

Rules:
- **No simulator in non-oracle modules.** A module may read the simulator only if it says `uses_sim = True`. Every
  result lists the modules that did (`uses_sim`), so a "VLM" result that leaned on the simulator shows up.
- **GR00T never sees the memory.** It gets its own observation and a sentence, as in training.
- **The VLM sees what a real robot has:** the views, the proprio, the memory. Every VLM call is logged in
  `memory.vlm_calls`: module, latency, tokens, cost, answer.
- **A VLM call never crashes a run.** A failed call returns a `VLMResponse` with `error` set, and each module has a
  fallback: the tracker gives no answer, the confirmer acts, recovery falls back to retrace.
- **A false "done" is the dangerous error.** The coordinator moves on and never retries that sub-goal. Score a VLM
  tracker against `OracleTracker` on recorded runs before it runs in the loop at scale.

## Memory (memory.py): what is kept for the decisions
T = tracking "where am I?", C = confirming "is it a failure?", R = recovery "what now?", X = the retrace mechanism.

| field | content | for |
|---|---|---|
| `plan` | per sub-goal a sentence and a completion condition | T C R |
| `stages[k].attempts` | every attempt at sub-goal k: when it began, why (first time, after a retrace, ...), the trace entry to go back to, how it ended (done, `timeout -> retrace`, undone, episode end) | C R X |
| `stages[k].done_times`, `undone_times` | when it was done, and undone | T |
| `stages[k].last_status` | the tracker's latest verdict and evidence on it | T |
| `expected[k]` | how long it usually takes (the timeout window) | C R |
| `keyframes` | the scene when each sub-goal first began, when its latest attempt began, when it was done, at each trigger | T C |
| `stages[k].history` | up to 8 moments spread over the current sub-goal's attempts | C R |
| `frames` | the last 24 chunk ends (3 views + proprio) | T C R |
| `trace` | the robot's state after every env step: 7 arm joints, base pose, hand pose, gripper command | X; gripper events and motion for C R |
| `notes` | short facts the VLM asked to keep ("the lemon fell out of the tilted bowl") | T C R |
| `triggers`, `interventions` | what fired, the confirmer's verdict and reason, what was done and its result | C R |
| `status_log`, `vlm_calls`, `free_steps` | every tracker answer, every VLM call, the retrace time | analysis, clock |

Derived views for the VLM:
- `robot_state(obs)`: the gripper, the finger gap, whether it is probably holding something, motion recently and since
  the attempt began, gripper changes in the attempt;
- `attempt_summary(k)`;
- `time_on_stage(obs)`.

## What the VLM is sent (context.py)
Every request is built on the pilot's `Query` (`coordinator/vlm/pilot.py`). It keeps the pilot's intro (the task sentence,
the sub-goals in order, how many moments and cameras) and its frame lines ("t = 32.8 s (chunk 41), wrist camera:"
before each image). Progress requests also keep the pilot's question and answer format, which were validated in the
pilot. The memory blocks go between the intro and the frames.

| block | progress | confirm | recovery |
|---|---|---|---|
| the pilot's intro: task sentence, sub-goals in order | yes | yes | yes |
| each sub-goal's state from the memory (done at, CURRENT: attempt n, time in it vs. usual, not started) | yes | yes | yes |
| the tracker's previous verdict on the current sub-goal | yes | | |
| the alarm in words | | yes | yes |
| every attempt at the sub-goal and how it ended; interventions and results | | yes | yes |
| robot sensors (`robot_state`) | yes | yes | yes |
| notes | yes | yes | yes |
| moments (each: label, time, 3 views) | first start of the sub-goal, previous completion, 3 recent, now | attempt start, 3 spread over the attempt, recent, now | first start, attempt start, 2 over the attempt, recent, now |
| question and JSON format (prompts.py) | the pilot's: sub-goals done, event, event detail, evidence, confidence (an event other than none becomes a note) | failure?, kind, reason, confidence, note | action (from the options given), instruction, reason |

`ContextConfig` sets the cameras, the number of moments and which blocks go in. To see exactly what is sent, run
`scripts/vlm_check.py <spec> --point <id> --dry`. It writes each request's text and its images in order, under
`../runs/20261106_coordinator/vlm_check/<id>/`.

An example, live, with Claude Opus 5.5 at SteamInMicrowave s14. The ground truth there: the lemon had been placed
in the bowl, then fell out while the bowl was carried, so sub-goal 1 is undone.
- The tracker, using the pilot's question, called sub-goal 1 not done and the event `dropped` ("while the robot
  carried the tilted bowl by its rim, the lemon fell out").
- The confirmer called it `dropped`, with confidence 0.8.
- Recovery chose `retrace`.
- The three calls took about 9 s each and cost $0.075 in total.

A local Qwen3.5-9B (`hf:` backend) on the same requests, with the earlier wording, contradicted itself: it said the
lemon was in the bowl and then that it was on the counter. That matches the model comparison below.

## VLM backends (vlm/)
```python
from coordinator.vlm import make_vlm, VLMRequest
vlm = make_vlm("claude:claude-opus-5-5")            # or "vllm:Qwen/Qwen3.8-27B-FP8@8102", "hf:qwen3.5-9b@8101", "mock"
res = vlm.ask(VLMRequest(parts=[("text", "..."), ("image", rgb)], system="...", schema={...}))
res.data, res.error, res.seconds, res.input_tokens, res.cost_usd
```
| spec | backend | notes |
|---|---|---|
| `claude:<model>` | `ClaudeVLM`, the official `anthropic` SDK | structured outputs enforce the schema; `effort=` (Opus 5.5 default medium); server-side fallback on; key from `ANTHROPIC_API_KEY` / `CLAUDE_API_KEY` or the repository's `.env` (never printed). $/MTok in / out: Opus 5.5 4 / 20, Sonnet 5.5 2 / 10, Haiku 5.5 0.10 / 0.50 |
| `openai:<model>`, `gemini:<model>` | `OpenAICompatVLM`, cloud | key from `OPENAI_API_KEY` / `GEMINI_API_KEY`; pass `price=(in, out)` to get costs |
| `vllm:`, `sglang:`, `ollama:`, `lmstudio:` `<model>@<port>` | `OpenAICompatVLM`, local server | `structured=False` turns off schema enforcement (it cost the 9B Qwen 5 points in the pilot); `thinking=` sets the Qwen chat-template switch |
| `hf:<name>@<port>` | `HFServerVLM` with the pilot's server `coordinator/vlm/vlm_server.py` | a transformers server (mm_bak env) for models vLLM does not serve |
| `mock` | `MockVLM` | tests and dry runs |

A new backend subclasses `vlm.base.VLM` and implements `_call(req) -> dict(text, model, input_tokens, output_tokens,
stop_reason)`. The base class handles the rest: timing, JSON parsing, schema checks and cost.

Which model: the 2026-11-05 comparison (`../runs/20261105_recovery/README.md`, "Which VLM") judged stage flags on
60 pilot queries.

| model | stage flags correct | false "done" |
|---|---|---|
| Claude Opus 5.5 (effort medium) | 0.95 | 0 |
| Qwen3.5-9B (local) | 0.90 | 9 |
| Qwen3.8-27B-FP8 (local, vLLM) | 0.70-0.83 | (see that README) |

Local servers on this machine:
- **transformers:** `~/miniconda3/envs/mm_bak/bin/python coordinator/vlm/vlm_server.py --model Qwen/Qwen3.5-9B --port 8101`.
- **vLLM:** the gemma4 env, `vllm serve ... --port 8102`. It needs that env's `bin` on PATH and its `lib` on
  LD_LIBRARY_PATH, plus `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **GPUs:** use `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and pick the GPU with `CUDA_VISIBLE_DEVICES`.

## Configurations (configs.py)
| config | tracker | detectors | confirmer | recovery | runs now? |
|---|---|---|---|---|---|
| `plain` | | | | | yes: GR00T alone |
| `oracle_retrace` | oracle | timeout, undone | accept | retrace | yes: the current method |
| `oracle_retrace_ev` | oracle | + oracle events | accept | retrace | yes: an earlier-detection bound |
| `oracle_vlm_recov` | oracle | timeout, undone | accept | VLM | yes: the recovery decision alone (VLM calls only at triggers) |
| `vlm_track` | VLM | timeout, undone | accept | retrace | yes: the tracker alone |
| `vlm_confirm` | VLM | timeout, undone | VLM | retrace | yes |
| `vlm_full` | VLM | timeout, undone | VLM | VLM | yes: the proposal's coordinator |
| `proprio_triggers` | oracle | timeout, undone, empty grasp, stationary | accept | retrace | after `detection/proprio.py` |
| `oracle_ladder` | oracle | timeout, undone | accept | ladder | after `motion/backoff.py` |

The VLM configs use the task's stage sentences as the checklist (`--plan given`, the default). With `--plan vlm`, the
VLM writes the checklist itself.

Cost, at about $0.02 per Opus call:
- the tracker checks every 4 chunks, roughly 25 calls in a 100-chunk episode, about $0.50;
- a run of `vlm_full` over the 86 failure-point runs costs roughly $40-80.

Ask before large API spends.

## Running
```bash
# GR00T servers, two per GPU (envkit/README.md)
setsid nohup envkit/servers.sh 1 5581 2 > /dev/null 2>&1 < /dev/null &

PY="envs/groot/bin/python -I"; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1
$PY coordinator/tests/test_offline.py && $PY coordinator/tests/test_vlm.py           # no server, no API
$PY coordinator/scripts/vlm_check.py claude:claude-opus-5-5                          # one cheap live call
$PY coordinator/scripts/vlm_check.py mock --point SteamInMicrowave_s14 --dry        # what the VLM would be sent
POLICY_PORT=5581 $PY coordinator/scripts/run.py point SteamInMicrowave_s6 oracle_retrace 0
POLICY_PORT=5581 $PY coordinator/scripts/run.py point SteamInMicrowave_s6 vlm_full 0 --vlm claude:claude-opus-5-5
POLICY_PORT=5581 $PY coordinator/scripts/run.py episode KettleBoiling 3 oracle_retrace 0
SHARD=0/1 setsid nohup coordinator/scripts/batch.sh 1 14 oracle_retrace "0 1" \
    > runs/20261106_coordinator/oracle_retrace_1.log 2>&1 < /dev/null &
```
Each run writes `runs/20261106_coordinator/<config>/<run>_p<seed>.json`. It holds the result, the whole memory summary
(attempts, triggers, interventions, notes, VLM calls with answers and costs) and `uses_sim`. A point run also writes
`<run>_p<seed>.pkl.gz` with every env chunk, as loop.py's records do.

## To do (owners to be assigned)
| task | what | owner |
|---|---|---|
| VLM tracker | tune `prompts.PROGRESS_Q` and `ContextConfig`; score against the oracle on recorded runs (precision / recall of "done" per sub-goal, false "done" rate); choose `done_streak`, `track_every`, the model | ___ |
| proprio triggers | `detection/proprio.py`: `EmptyGrasp` (closed, finger gap under 4 mm), `Stationary` (no motion for H chunks); score against `OracleEvents`; then `proprio_triggers` | ___ |
| VLM confirmer | tune `CONFIRM_Q`; score on labelled triggers (timeouts that were real failures vs. slow progress) | ___ |
| back-off and VLM recovery | `motion/backoff.py`; `VLMRecovery` against `RetraceRetry` (`oracle_vlm_recov`) | ___ |
| evaluation | full episodes (`run.py episode`) on both composite splits; Wilson intervals, McNemar, sub-goals completed, detection precision / recall, VLM calls and cost; the proposal's baselines | ___ |

Why detection matters next: in PackIdenticalLunches, retrace fixes the failed stage 22 / 76 times, but the task
succeeds only 3 / 76. A later stage is called stuck only after its phase 0 p90 duration, which is up to 133 chunks, so
it rarely gets a retry.

## Idealisations to state in any report
- The oracle modules read the simulator: detection is assumed solved there.
- From failure points, the replay records where each sub-goal began from the recorded oracle tracker.
- The retrace reads the base pose from the simulator; a real robot would use odometry. The arm uses joint impedance
  while retracing, and the retrace's time is not counted.
