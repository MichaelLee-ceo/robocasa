# The to-do list tracker, offline on recorded GR00T runs (started 2026-11-07)

The user's progress-tracking design (2026-11-07): the coordinator keeps a to-do list the way a person keeps one.
- **Plan.** The VLM splits the composite task instruction into items in its own words, one skill per item (a
  pick-and-place is one item; placements without a fixed order by count), each with what "done" looks like and what
  must hold before the next item can start (the hand-off).
- **Tracking.** At every chunk end, only the current item: done yes / no, its progress level (not started, reaching,
  handling, finishing, done: the progress dots), advancing or not, and whether the earlier items still hold.
- **Ticking.** Done at 2 checks in a row, then a separate hand-off check; a failed check is recorded in the to-do
  list and repeated 4 chunks later at the earliest.
- **The status map.** The to-do list ([x] / [>] / [ ]) with the current item's conditions, its earlier attempts and
  how they ended, failed hand-off checks and its recent progress dots; shown to the VLM at every check and saved with
  each run.
Code: `coordinator/tracking/vlm.py`, `coordinator/context.py` (todo_block, item_moments, item_request,
handoff_request, plan_request), `coordinator/prompts.py` (PLAN_Q, ITEM_Q, handoff_question),
`coordinator/detection/progress.py` (stall and hand-off triggers, scored here from the logged answers).

## Setup
- The 90 phase 0 plain runs (`../20261105_recovery/phase0/records`), replayed exactly; the coordinator only watches
  (`coordinator/scripts/eval_tracker.py`), its to-do list built from the VLM's own answers. The oracle
  (`stages.track`) is logged next to every answer and never shown to the VLM.
- Local Qwen3.8-27B (vLLM, GPU0), no thinking, temperature 0, free-form JSON; every chunk.
- Arms (only the images differ; two cameras, left shoulder and wrist, so that 9 moments fit the server's 18-image
  limit):
  - `attempt/`: anchored at the item: the moment the previous item was done (the hand-off), the attempt's start,
    4 moments spread over the attempt, the last 2 chunk ends, now;
  - `recent/`: a fixed window: the last 9 chunk ends, 2 chunks apart.
- Watchdog: `coordinator/scripts/watch_eval.py` (setsid; results with failed VLM calls or a replay mismatch are
  deleted and re-run, at most 3 tries; `STATUS.txt`; `SUMMARY.md` from `coordinator/scripts/score_tracker.py` at the
  end).

## Smoke tests before the batch
- Plan prompt: the 27B first split "pick ... and place ..." into two items (3 for KettleBoiling, the oracle has 2)
  and tied PackIdenticalLunches items to particular containers; with the rules and an example (not one of these tasks)
  all 7 checked plans had the oracle's number of items. PackIdenticalLunches items are still ordered by kind
  (vegetable, meat, vegetable, meat) with order-free conditions ("one tupperware holds a steak"); scoring compares
  counts.
- Item question: with "done" only as a progress level, the 27B never said "done" (KettleBoiling s0: "kettle resting
  on the burner, arm retracted", answered "finishing" for 10 chunks); "done" is now an explicit yes / no.

## During the batch
- **PackIdenticalLunches: the plan orders the items by kind.** The 27B writes "first lime, second lime, first fish,
  second fish" (or vegetable, meat, ...) despite the rule; GR00T often places another kind first (s6: a fish at chunk
  16). The tracker then rightly says the current item (a lime) is not done while the oracle, which counts placements,
  has the first stage done: scored as "missed" and a long lag, and the dots read "not advancing" for 13 checks while
  the robot places the fish (in a closed loop a stall trigger would rewind correct work). A stronger rule with an
  example of the same structure (apples and pears, by count) did not change the 27B's plans (3 runs), so it was not
  used. The batch continues with the prompt it started with; the PackIdenticalLunches scores are to be read with this.
  Possible fixes (for the user to decide): let the to-do list tick a later item when it is seen done (out of order),
  or let the VLM say which item the robot is working on.
- 18:05-18:09: that prompt test was made in the repository while the batch ran; three runs launched in that window
  (PackIdenticalLunches s9 recent, s10 both) were stopped before writing anything and re-run with the batch's prompt.

## Results (finished 2026-11-07 23:35; 180 runs, 22,608 checks, no run re-run; `SUMMARY.md`, `*/score.json`)
Every chunk, local 27B under 16 parallel runs: median 13.5 s per call; 6 hours in all.

| 90 runs | v0 (all sub-goals at once, every 4 chunks; `../20261106_vlm_offline`, 27B full) | **to-do, attempt window** | to-do, recent window |
|---|---|---|---|
| plans with the oracle's number of items | (given) | 90/90 | 90/90 (one run's answers unusable) |
| premature ticks (runs with one) | 58 (41) | **21 (18)** | 40 (29) |
| completions never registered | 57/202 (28 %) | 28/162 (17 %) | **17/160 (11 %)** |
| tick lag after the oracle, median / p90 (chunks) | | 2 / 26 | 1 / 22 |
| item "done" accuracy (false done / missed) | (flag accuracy 0.90) | 0.83 (0.009 / 0.16) | 0.87 (0.022 / 0.11) |
| failure points with a stall while the oracle is on the failed stage | 40/48 (VLM events, 3 in a row) | 34/48 | 36/47 |
| ... with the tracker also on that item (what a retry needs) | | 23/48, lead 31 chunks | 25/47, lead 21 |
| false stalls per run (n = 6) | 0.70 | 0.68 | 0.40 |

By task (attempt / recent): premature ticks KettleBoiling 9 / 8, SteamInMicrowave 2 / 10, PackIdenticalLunches 10 / 22;
never registered 2 / 1, 18 / 9, 8 / 7.

- **The to-do list cuts the dangerous error.** Premature ticks fall from 58 (v0) to 21 with the attempt window; the
  recent window has twice as many (40), most on PackIdenticalLunches (22) and SteamInMicrowave (10). Seeing where
  the item began keeps the VLM from calling it done too early.
- **The hand-off condition is too strict, and it is the main cause of missed completions.** The attempt window's
  hand-off check blocked 69 ticks of items the oracle had done (and prevented 13 premature ones). 40 of the 69 are
  SteamInMicrowave "close the door": "the gripper is still in contact with the door" while GR00T goes straight on to
  the start button next to it; 8 are KettleBoiling "the gripper is still over the kettle". The plan prompt's example
  ("let go of the object and moved away from it") made the VLM require moving away; the oracle's stage checks ask
  only that the gripper no longer touches the object, for the same reason (the next stage often works right next to
  it). Missed completions put the tracker behind the robot (9 of the 48 failure points).
- **Stall detection works when the tracker is on the right item:** of the 28 failure points where the tracker was on
  the failed item at the point (attempt), 21 were caught by the progress dots (n = 6), a median 31 chunks before the
  p90 timeout. The rest are lost to tracking errors (tracker behind at 9 points, ahead at 11).
- **PackIdenticalLunches:** the kind-ordered plan (above) costs missed completions and premature ticks on items 3-4.

## Next (for the user to decide)
1. Hand-off = "let go of it and no longer touching it" (as the oracle's stage checks), not "moved away".
2. PackIdenticalLunches order: tick a later item when it is seen done, or let the VLM say which item the robot is
   working on.
3. Then the closed loop (`vlm_todo`: VLM tracking + stall / hand-off / undone triggers + retrace) against the oracle
   tracker's 28 %.
