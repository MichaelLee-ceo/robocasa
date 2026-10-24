# Atomic steps for the 11 remaining composite-seen tasks

Files: `atomic_steps_seen11.py` (TASKS dict, same entry format as `runs/scripts/atomic_steps.py`, plus `step_segments(demo)`),
`audit11.py` -> `audit11.json` (annotation audit of all demos), `validate_seen11.py` -> `val_<Task>.json`, `make_report.py` -> `report.json`,
`diag.py` / `probe_toast.py` (diagnostics for the failures), `logs/`.

Method:
- **Annotation sequences.** Taken from `annotation.human.subtask_name` over all ~500 demos per task. NavigateKitchen is dropped and adjacent
  same-name segments are merged. Merging by name alone collapses two consecutive pick/place steps of the same skill into one (e.g. LoadDishwasher
  cup + bowl). So a new step also starts when a `pick` stage follows a `place`/`execute` stage of the same name. Both views are in `report.json`
  (`name_merged_sequences`, `step_sequences`). The majority step sequence gives the steps.
- **Validation.** 10 evenly spaced demos per task (`floor(i*N/10)`), plus all/some minority-sequence demos as extras. For each demo: `reset_to` once
  (fresh fixture Python state), then every frame runs `set_state` (which runs `update_state`), one `env._check_success()` (what `reward()` does in
  `env.step`; needed for the success terms that latch inside `_check_success`), and the sequential tracker. Separately, after its own `reset_to`, each
  step predicate is checked to be False at the start frame of its annotated segment. It held for every step in every demo (with `S={}`).
- **Timing.** "d" below = first-pass frame minus the annotated end of the step (end of its last non-navigate segment).
- **Python-level state.** All per-episode bookkeeping (completion order, toaster latch) happens online in `S`. `prepare()` returns `{}`, so the checks
  also work for seeded episodes.
- **`seg` field.** It indexes the majority stage sequence only. NavigateKitchen insertions vary between demos in several tasks (SetUpCuttingStation
  has 11 stage sequences, StirVegetables 20, PackIdenticalLunches 23). For a given demo use `step_segments(demo)`, not `segs[seg]`.

| task | steps | annotation | validation (10 demos) | d range per step (frames) |
|---|---|---|---|---|
| ScrubCuttingBoard | PickUpSponge > ScrubCuttingBoard | 1 seq (504/504) | **8/10** | -4..9 / 6..16 |
| WashLettuce | TurnOnSinkFaucet > PickPlaceCounterToSink | 2 seqs, majority 494/501 | 10/10 (+2/2 minority) | 0..0 / 23..88 |
| RinseSinkBasin | TurnOnSinkFaucet > TurnSinkSpout | 1 seq (509) | 10/10 | 0..0 / -15..76 |
| LoadDishwasher | PickPlaceCounterToDishwasher x2 > CloseDishwasher | 1 seq (501) | 10/10 | -5..13 / -12..3 / 0..35 |
| StoreLeftoversInBowl | PickPlaceCounterToCounter x2 > PickPlaceCounterToFridge | 1 seq (503) | 10/10 | -7..-3 / -5..0 / 9..20 |
| SetUpCuttingStation | PickPlaceDrawerToCounter > PickPlaceCounterToCounter | 2 seqs, majority 503/504 | 10/10 (+1/1 minority) | -15..-5 / 7..26 |
| StirVegetables | PickPlaceCounterToStove x2 > PickUpSpatula > StirVegetables | 1 seq (501) | 10/10 | -7..9 / -6..3 / -16..10 / -39..-10 |
| SearingMeat | PickPlaceCabinetToStove > PickPlaceCounterToStove > TurnOnStove | 1 seq (501) | 10/10 | 5..72 / 0..3 / -11..53 |
| PackIdenticalLunches | PickPlaceFridgeToCounter x4 | 2 seqs, majority 499/501 | 10/10 (+2/2 minority) | 10..20 / 12..22 / 11..17 / 10..30 |
| DeliverStraw | OpenDrawer > PickPlaceDrawerToCounter | 1 seq (504) | 10/10 | -93..-6 / 15..129 |
| GetToastedBread | TurnOnToaster > PickPlaceToasterToCounter | 1 seq (506) | **7/10** | -6..0 / 10..23 |

## Step-check notes
- **Steps on several objects in any order** (LoadDishwasher dishes, StoreLeftovers drumstick/vegetable, StirVegetables veg1/veg2, PackIdenticalLunches
  4 foods). Step k passes when at least k+1 of the objects are done (in rack / in bowl / in pot / in a tupperware) and released. The order differs
  between demos, so the checks don't fix it.
- **PackIdenticalLunches.** Steps 1-3 only count foods in some tupperware. They don't check the one-veg-plus-one-meat-per-tupperware rule; the last
  step (`_check_success`) does.
- **SearingMeat.** Step 1 = pan touching the stove within 0.15 m of the task knob's burner, and gripper far. The official
  `check_obj_location_on_stove` also needs the burner on, so that part is left to step 3 (= `_check_success`). The +64/+72 outliers on step 1 come
  from waiting for `gripper_obj_far` (0.25 m).
- **ScrubCuttingBoard PickUpSponge.** Passes when the gripper touches the sponge and the sponge is off the counter. `OU.check_obj_grasped` was tried
  first. It misses many real sponge holds because the fingers stay partly open on the soft sponge (ep 50: gripper touching the sponge from frame 80,
  "grasped" only from about frame 300).
- **GetToastedBread TurnOnToaster.** The annotated segment is press + wait (the wait lasts until the lever is back up). The check passes when a slot
  that held the bread was switched on and has since finished: `turned_on` is False again and the lever is <= 0.1. A first version passed at
  "finished" alone, about 140-175 frames before the annotated end, while the lever was still rising. The bread-in-slot condition is latched from any
  earlier frame because of the contact problem described below.
- **Minority annotation orders still pass** (WashLettuce faucet-last demos 51 and 135; SetUpCuttingStation meat-first demo 413). The later step only
  passes once the earlier one is done.
- **PackIdenticalLunches demos 426 and 444.** They have only 3 annotated pick/place units (one place directly followed by a pick without a new pick
  label), but all 4 steps pass on them.

## Problems / not reliably checkable
- **ScrubCuttingBoard ep 50 and 151 fail.** The official `_check_success` is never True when these demos are replayed. The env's own contact timer
  only reaches 2 and 4 (5 needed). It counts contacts only while `OU.check_obj_grasped(sponge)` is True, and that is False during most of the
  scrubbing (same grasp-detector problem as above). Our step 1 passes on both demos; only the official final check fails. Unless replay differs
  from the live run, the official metric seems to undercount real scrubbing.
- **GetToastedBread ep 50, 354 and 455 fail.** The official `_check_success` is never True when these demos are replayed. It latches
  `toaster_on` only if the bread touches a slot floor at a frame where the toaster is on. In the replayed states the bread touches the slot floor
  before the lever goes down but has no contacts at all while toasting (`probe_toast.py`; it sits about 6 cm lower with the lever). So in these
  demos the official success can't be reproduced from states, and our step 2 (= `_check_success`) can't pass. Step 1 passes on all 10.
- **DeliverStraw OpenDrawer** passes up to 93 frames (about 1/3 of the segment) before the annotated end. The drawer reaches the atomic task's >= 0.95
  open threshold while the annotated segment continues.
- **Final steps with accumulated success.** WashLettuce (25 frames of washing) and SearingMeat/DeliverStraw (`gripper_obj_far`) pass 10-129 frames
  after the annotated end. That is a property of the official success, not a predicate bug.
- **Coverage.** Only 10 demos per task (+ extras) were replayed. The ScrubCuttingBoard and GetToastedBread rates (2/10 and 3/10 unreproducible
  official success) are small-sample estimates.
