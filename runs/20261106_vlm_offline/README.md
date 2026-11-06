# VLM supervisor, offline on recorded GR00T runs (started 2026-11-06, overnight)

Questions (the user's, 2026-11-06): what can the VLM do here, for progress tracking, failure detection and recovery
control, and what does the memory add? Local models only (no API spend): Qwen3.5-9B, and the bf16 Qwen3.8-27B on GPU0
(allowed by the user for this).

## Setup
- **Episodes.** The 90 phase 0 plain runs (`../20261105_recovery/phase0/records`: KettleBoiling, PackIdenticalLunches,
  SteamInMicrowave x 30 seeds; GR00T alone; 39 successes, 51 failures). They are replayed exactly from the recorded
  chunks (`replay_matches_record` checks the length and the outcome), so the VLM sees what GR00T saw.
- **The coordinator watches.** `coordinator/scripts/eval_tracker.py` runs the coordinator with the VLM tracker and no
  detector or recovery. The VLM's memory is built from its own verdicts, as in a live run: a sub-goal it calls done is
  closed and the next one begins. It is asked every 4 chunks (the coordinator's default).
- **The oracle** (`stages.track`, from the record) is logged next to every answer and never shown to the VLM.
- **The question** is the pilot's (`coordinator/vlm/pilot.py`): one done flag per sub-goal, an event (none,
  missed_grasp, dropped, fixture_short, wrong_element, no_progress, other), evidence, confidence. Free JSON (vLLM's
  enforced schema cost the 9B 5 points in the pilot), no thinking, temperature 0.
- **Context arms** (what the memory adds; the recent frames are the same in every arm: 4 chunk ends, 2 apart, 3
  cameras, as in the pilot):

  | arm | memory in the request |
  |---|---|
  | frames | none: the pilot's query |
  | state | + the memory's progress state: each sub-goal done at / CURRENT with its time vs. its usual time |
  | text | + the tracker's previous verdict and evidence, the robot's own sensors (gripper, finger gap, motion), the notes kept so far |
  | full | + keyframe images: when the current sub-goal began, when the previous one was done |

## Scores (`coordinator/scripts/score_tracker.py`)
- **Progress tracking:** flag accuracy; false "done" (a check calling a sub-goal done that the oracle does not); the
  memory ahead of the oracle (moved on too early: the dangerous error, that sub-goal is never retried); lag from the
  oracle's completion to the memory moving on.
- **Failure detection:** at the 48 phase 0 failure points (not "late"), whether the VLM reported an event on the failed
  sub-goal before the oracle supervisor's point (its stall window), and how many chunks earlier; false alarms while the
  sub-goal is in fact progressing (done within 16 chunks) and in successful runs.

## Log
- 27B first try failed: on the Blackwell GPU0, vLLM's ViT flash-attn kernel is "PTX compiled with an unsupported
  toolchain" (driver 570) -> `--mm-encoder-attn-backend TORCH_SDPA --attention-backend TRITON_ATTN`.
- 27B second try: 24 render contexts (about 0.55 GB each) on GPU0 next to the server ran the GPU out of memory and
  killed the engine; those answers were discarded (`qwen27b/_failed_first_try/`). Now 8 render contexts on GPU0 at
  most, the server at util 0.66; the 9B's workers render on GPU2.

## Results (frames vs full memory; 90 runs, 2843 checks per arm; `score_tracker.py`, and a rescoring against the
oracle's first completion of each sub-goal, so that a sub-goal reading undone only because the hand came near does not
count against the VLM)

| | 9B frames | 9B full | 27B frames | 27B full |
|---|---|---|---|---|
| flag accuracy | 0.66 | 0.75 | 0.80 | **0.90** |
| checks with a false "done" | 0.52 | 0.59 | 0.42 | 0.26 |
| premature move-ons (runs with one) | 234 (85) | 162 (74) | 172 (81) | **58 (41)** |
| completions never registered (of 202) | 17 | 39 | 15 | 57 |
| failures caught, 3 events in a row (of 48) | 31 | 39 | 33 | 40 |
| lead over the p90 timeout, median chunks | 28 | 17 | 24 | 22.5 |
| false triggers per run (3 in a row) | 0.64 | 1.21 | 0.46 | 0.70 |
| checks with an event while progressing | 0.51 | 0.76 | 0.45 | 0.55 |
| median latency | 4.3 s | 4.3 s | 8.6 s | 8.5 s |

- **Progress tracking: not reliable with these local models.** Even the 27B with memory moves on before the oracle in
  41 of 90 runs (18 times on a sub-goal the oracle never completed); in a closed loop those sub-goals would never be
  retried. False "done" concentrates on PackIdenticalLunches sub-goals 1-2 (identical items).
- **Memory helps the 27B mostly by making it cautious:** premature move-ons 172 -> 58 and flag accuracy 0.80 -> 0.90,
  but completions never registered 15 -> 57. For the 9B it helps less and raises the false alarms.
- **Failure detection: early but noisy.** Both models report "no_progress" in about half of the checks where the
  sub-goal is in fact progressing. Three reports in a row: the 27B catches 40 of 48 failures a median 22.5 chunks
  (18 s) before the oracle supervisor's timeout, with 0.7 false triggers per run.
- Not run: the state / text arms; the closed loop (VLM early detection + retrace).
