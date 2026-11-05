"""Stage tracker v2 (2026-11-05): the stages of the 16 Composite-Seen tasks, checked consistently with the official
success.

Stages follow the official RoboCasa365 subtask annotation (as in the earlier atomic_steps.py, NavigateKitchen dropped). A stage's
check is built from the conjuncts of the task's official _check_success() and the success of the RoboCasa atomic task
of the same name. It is written so that:
  * the last stage is the official success itself: envkit's chunk-end success (info["success"] after the last executed
    step of the chunk), never a new call of _check_success();
  * checks only read the env: no _check_success(), update_state(), forward() or set_state(). The official counters and
    latches are read where they live: WashLettuce washed_time, StirVegetables success_time, ScrubCuttingBoard
    board_contact_*, RinseSinkBasin washed_loc, GetToastedBread toaster_on;
  * where the official success is a latch (WashLettuce, RinseSinkBasin, StirVegetables: LATCHED), every stage counts as
    done while it holds: the task is complete even if later moves undo an earlier stage;
  * every check is self-contained: stage k holds now only if what stages < k left behind still holds, so a broken
    earlier result takes the credit away (no dead-end credit). PackIdenticalLunches counts only items that can stay in
    a valid final packing (one vegetable and one meat per tupperware). SteamInMicrowave keeps checking that the
    vegetable is in the bowl. StirVegetables needs the pot on the task's burner (the official stir counter does);
  * objects handled in any order are counted, not named (PreSoakPan, StackBowlsCabinet, LoadDishwasher,
    StoreLeftoversInBowl, StirVegetables, PackIdenticalLunches);
  * a stage that is a means, not a conjunct of the success, also counts as done once a later stage shows it was done
    (DeliverStraw: drawer open, or the straw already out of it). GetToastedBread's toasting stage is the official
    toaster_on latch;
  * intermediate placements need the gripper off the object (no contact) rather than 0.25 m away (OU.gripper_obj_far):
    the next stage often works right next to it. The official success keeps its own distance tests.

track(ep) -> dict(t, stage = number of leading stages done now, done = [bool per stage], official, ever, info)
  info carries the official counters and per-task details (e.g. washed_time, dead_end).
instructions(ep) -> the sentence of every stage (the atomic task's template with this scene's words).
usage: import stages; stages.track(ep) after every ep.step (envkit.Episode)."""
import numpy as np
import robocasa.utils.object_utils as OU


# ---------------------------------------------------------------- helpers (all pure reads)
def _lang(e, o):
    return OU.get_obj_lang(e, obj_name=o)


def _pos(e, o):
    return np.array(e.sim.data.body_xpos[e.obj_body_id[e.objects[o].name]])


def _released(e, o):
    """the gripper is not touching the object"""
    return not e.check_contact(e.robots[0].gripper["right"], e.objects[o])


def _in(e, o, r, **k):
    return bool(OU.check_obj_in_receptacle(e, o, r, **k))


def _count(e, objs, pred):
    return sum(1 for o in objs if pred(e, o))


def _water_on(e):
    return bool(e.sink.get_handle_state(env=e)["water_on"])


def _nearest_burner(e, o):
    p = _pos(e, o)[:2]
    best = (None, 1e9)
    for loc, site in e.stove.burner_sites.items():
        if site is not None:
            d = float(np.linalg.norm(np.array(e.sim.data.get_site_xpos(site.get("name")))[:2] - p))
            best = min(best, (loc, d), key=lambda x: x[1])
    return best


def _on_burner(e, o, loc=None, th=0.15):
    """o touches the stove within th (xy) of a burner (of burner loc, if given); the burner need not be on"""
    if not OU.check_obj_fixture_contact(e, o, e.stove):
        return False
    if loc is None:
        return _nearest_burner(e, o)[1] < th
    site = e.stove.burner_sites.get(loc)
    if site is None:
        return False
    return float(np.linalg.norm(np.array(e.sim.data.get_site_xpos(site.get("name")))[:2] - _pos(e, o)[:2])) < th


# ---------------------------------------------------------------- per-task checks
def _kettle_placed(e):
    return _on_burner(e, "obj") and OU.gripper_obj_far(e, "obj")


def _sink_placed(e, o):
    return OU.obj_inside_of(e, o, e.sink, partial_check=False) and OU.gripper_obj_far(e, o)


def _bowl_in_cabinet(e, b):
    return OU.obj_inside_of(e, b, e.cabinet) and OU.gripper_obj_far(e, b)


def _veg_in_bowl(e):
    return _in(e, "vegetable", "bowl")


def _bowl_in_microwave(e):
    return _veg_in_bowl(e) and OU.obj_inside_of(e, "bowl", e.microwave)


def _sponge_held(e):
    """official grasp test (the one the scrub counter uses) with the sponge off the counter; or scrubbing has begun"""
    return (OU.check_obj_grasped(e, "sponge") and not OU.check_obj_fixture_contact(e, "sponge", e.counter)) \
        or e.board_contact_timer > 0


def _scrub_info(e):
    sweep = 0.0
    if e.board_contact_positions:
        p = np.array(e.board_contact_positions)
        sweep = float(np.linalg.norm(p.max(0) - p.min(0)))
    return dict(board_contact_timer=int(e.board_contact_timer), sweep=round(sweep, 3))


def _pot_ok(e):
    """the official stir counter's condition: the pot on the task's burner, that burner on"""
    return e.stove.check_obj_location_on_stove(env=e, obj_name="pot", threshold=0.15) == e.knob


def _veg_in_pot(e, o):
    return _in(e, o, "pot") and _released(e, o)


_VEG, _MEAT, _TUPS = ["vegetable0", "vegetable1"], ["meat0", "meat1"], ["tupperware0", "tupperware1"]


def _lunch(e):
    """items released in a tupperware that can stay in a valid final packing (per tupperware at most one vegetable and
    one meat count), and whether a tupperware already holds two of a kind (dead end unless an item is taken out)"""
    n, dead = 0, False
    for t in _TUPS:
        v = [o for o in _VEG if _in(e, o, t) and _released(e, o)]
        m = [o for o in _MEAT if _in(e, o, t) and _released(e, o)]
        n += min(len(v), 1) + min(len(m), 1)
        dead = dead or len(v) > 1 or len(m) > 1
    return n, dead


def _drawer_done(e):
    """OpenDrawer's atomic success (every drawer joint >= 0.95), or the straw is already out of the drawer"""
    return all(v >= 0.95 for v in e.drawer.get_door_state(env=e).values()) or not OU.obj_inside_of(e, "straw", e.drawer)


def _pan_placed(e):
    return _on_burner(e, "pan", e.knob) and _released(e, "pan")


def _meat_in_pan(e):
    return _pan_placed(e) and _in(e, "meat", "pan", th=0.07) and _released(e, "meat")


def S(skill, instr, done=None):
    """a stage: done(e) -> bool now; done=None marks the last stage (the official chunk-end success)"""
    return dict(skill=skill, instr=instr, done=done)


STAGES = {
    "KettleBoiling": [
        S("PickPlaceCounterToStove", lambda e: f"Pick the {_lang(e, 'obj')} from the counter and place it on the stove.",
          _kettle_placed),
        S("TurnOnStove", lambda e: f"Turn on the {(_nearest_burner(e, 'obj')[0] or 'front').replace('_', ' ')} burner of the stove."),
    ],
    "PreSoakPan": [
        S("PickPlaceCounterToSink", lambda e: "Pick the pan or the sponge from the counter and place it in the sink.",
          lambda e: _count(e, ["obj1", "obj2"], _sink_placed) >= 1),
        S("PickPlaceCounterToSink", lambda e: "Pick the other item from the counter and place it in the sink.",
          lambda e: _count(e, ["obj1", "obj2"], _sink_placed) >= 2),
        S("TurnOnSinkFaucet", lambda e: "Turn on the sink faucet."),
    ],
    "StackBowlsCabinet": [
        S("PickPlaceCounterToCabinet", lambda e: f"Pick a {_lang(e, 'bowl1')} from the counter and place it in the cabinet.",
          lambda e: _count(e, ["bowl1", "bowl2"], _bowl_in_cabinet) >= 1),
        S("PickPlaceCounterToCabinet", lambda e: "Pick the other bowl and stack it on top of the bowl in the cabinet."),
    ],
    "SteamInMicrowave": [
        S("PickPlaceSinkToCounter",
          lambda e: f"Pick the {_lang(e, 'vegetable')} from the sink and place it on the {_lang(e, 'bowl')} located on the counter.",
          lambda e: _veg_in_bowl(e) and _released(e, "vegetable")),
        S("PickPlaceCounterToMicrowave", lambda e: f"Pick the {_lang(e, 'bowl')} from the counter and place it in the microwave.",
          lambda e: _bowl_in_microwave(e) and OU.gripper_obj_far(e, "bowl")),
        S("CloseMicrowave", lambda e: "Close the microwave door.",
          lambda e: _bowl_in_microwave(e) and e.microwave.is_closed(env=e)),
        S("TurnOnMicrowave", lambda e: "Press the start button on the microwave."),
    ],
    "PrepareCoffee": [
        S("CoffeeSetupMug", lambda e: f"Pick the {_lang(e, 'obj')} from the cabinet and place it under the coffee machine dispenser.",
          lambda e: e.coffee_machine.check_receptacle_placement_for_pouring(e, "obj") and OU.gripper_obj_far(e, "obj")),
        S("StartCoffeeMachine", lambda e: "Press the button on the coffee machine to serve coffee."),
    ],
    "ScrubCuttingBoard": [
        S("PickUpSponge", lambda e: f"Pick up the {_lang(e, 'sponge')} from the counter.", _sponge_held),
        S("ScrubCuttingBoard", lambda e: f"Scrub the {_lang(e, 'cutting_board')} with the sponge, then release the sponge."),
    ],
    "WashLettuce": [
        S("TurnOnSinkFaucet", lambda e: "Turn on the sink faucet.", _water_on),
        S("PickPlaceCounterToSink", lambda e: f"Pick the {_lang(e, 'lettuce')} from the counter and place it in the sink."),
    ],
    "RinseSinkBasin": [
        S("TurnOnSinkFaucet", lambda e: "Turn on the sink faucet.", _water_on),
        S("TurnSinkSpout", lambda e: "Turn the sink spout to the left, center and right to wash all locations of the sink basin."),
    ],
    "LoadDishwasher": [
        S("PickPlaceCounterToDishwasher", lambda e: "Pick a dish from the counter and place it in the dishwasher.",
          lambda e: _count(e, ["dish0", "dish1"], lambda e, o: e.dishwasher.check_rack_contact(e, o) and _released(e, o)) >= 1),
        S("PickPlaceCounterToDishwasher", lambda e: "Pick the other dish from the counter and place it in the dishwasher.",
          lambda e: _count(e, ["dish0", "dish1"], lambda e, o: e.dishwasher.check_rack_contact(e, o) and _released(e, o)) >= 2),
        S("CloseDishwasher", lambda e: "Close the dishwasher door."),
    ],
    "StoreLeftoversInBowl": [
        S("PickPlaceCounterToCounter", lambda e: "Pick a leftover from the plate and place it in the bowl.",
          lambda e: _count(e, ["chicken_drumstick", "vegetable"], lambda e, o: _in(e, o, "bowl") and _released(e, o)) >= 1),
        S("PickPlaceCounterToCounter", lambda e: "Pick the other leftover from the plate and place it in the bowl.",
          lambda e: _count(e, ["chicken_drumstick", "vegetable"], lambda e, o: _in(e, o, "bowl") and _released(e, o)) >= 2),
        S("PickPlaceCounterToFridge", lambda e: f"Pick the {_lang(e, 'bowl')} from the counter and place it in the fridge."),
    ],
    "SetUpCuttingStation": [
        S("PickPlaceDrawerToCounter", lambda e: f"Pick the {_lang(e, 'knife')} from the drawer and place it on the {_lang(e, 'receptacle')}.",
          lambda e: _in(e, "knife", "receptacle") and _released(e, "knife")),
        S("PickPlaceCounterToCounter", lambda e: f"Pick the {_lang(e, 'meat')} from the plate and place it on the {_lang(e, 'receptacle')}."),
    ],
    "StirVegetables": [
        S("PickPlaceCounterToStove", lambda e: f"Pick a vegetable from the counter and place it in the {_lang(e, 'pot')}.",
          lambda e: _pot_ok(e) and _count(e, ["veg1", "veg2"], _veg_in_pot) >= 1),
        S("PickPlaceCounterToStove", lambda e: f"Pick the other vegetable from the counter and place it in the {_lang(e, 'pot')}.",
          lambda e: _pot_ok(e) and _count(e, ["veg1", "veg2"], _veg_in_pot) >= 2),
        S("PickUpSpatula", lambda e: f"Pick up the {_lang(e, 'spatula')} from the counter.",
          lambda e: _pot_ok(e) and _count(e, ["veg1", "veg2"], _veg_in_pot) >= 2 and OU.check_obj_grasped(e, "spatula")),
        S("StirVegetables", lambda e: "Stir the vegetables in the pot with the spatula."),
    ],
    "SearingMeat": [
        S("PickPlaceCabinetToStove", lambda e: f"Pick the {_lang(e, 'pan')} from the cabinet and place it on the {e.knob.replace('_', ' ')} burner of the stove.",
          _pan_placed),
        S("PickPlaceCounterToStove", lambda e: f"Pick the {_lang(e, 'meat')} from the container and place it in the {_lang(e, 'pan')}.",
          _meat_in_pan),
        S("TurnOnStove", lambda e: f"Turn on the {e.knob.replace('_', ' ')} burner of the stove."),
    ],
    "PackIdenticalLunches": [
        S("PickPlaceFridgeToCounter", (lambda kk: lambda e: f"Pick an item from the fridge and place it in a tupperware ({kk + 1} of 4).")(k),
          (lambda kk: lambda e: _lunch(e)[0] >= kk + 1)(k))
        for k in range(3)
    ] + [S("PickPlaceFridgeToCounter", lambda e: "Pick the last item from the fridge and place it in the tupperware that is still missing it.")],
    "DeliverStraw": [
        S("OpenDrawer", lambda e: "Open the drawer.", _drawer_done),
        S("PickPlaceDrawerToCounter", lambda e: f"Pick the {_lang(e, 'straw')} from the drawer and place it in the {_lang(e, 'glass_cup')} on the dining counter."),
    ],
    "GetToastedBread": [
        S("TurnOnToaster", lambda e: "Push down the lever of the toaster to turn it on.", lambda e: bool(e.toaster_on)),
        S("PickPlaceToasterToCounter", lambda e: "Place the toasted item on a plate."),
    ],
}

INFO = {  # official counters and per-task details, for analysis and the state-understanding ground truth
    "ScrubCuttingBoard": _scrub_info,
    "WashLettuce": lambda e: dict(washed_time=int(e.washed_time), water_on=_water_on(e)),
    "RinseSinkBasin": lambda e: dict(washed_loc=[bool(x) for x in e.washed_loc], water_on=_water_on(e)),
    "StirVegetables": lambda e: dict(success_time=int(e.success_time), pot_ok=bool(_pot_ok(e))),
    "PackIdenticalLunches": lambda e: dict(zip(("packed", "dead_end"), _lunch(e))),
    "GetToastedBread": lambda e: dict(toaster_on=bool(e.toaster_on)),
    "KettleBoiling": lambda e: dict(zip(("burner", "dist"), _nearest_burner(e, "obj"))),
}


# tasks whose official success is a latch (a counter or flags that never reset: washed_time, washed_loc, success_time):
# once it holds the task is complete, whatever later moves undo (stirring pushes vegetables out, the water is turned
# off), so every stage counts as done. Every other official success holds only while its state does, so a stage that
# fails while it holds is a wrong stage definition (tests/run_tests.py checks this).
LATCHED = ("WashLettuce", "RinseSinkBasin", "StirVegetables")


def track(ep):
    """the stage state of an envkit.Episode now (call after ep.step; pure: reads the env only)"""
    e, st = ep.kenv, STAGES[ep.task]
    official = bool(ep.chunks[-1]["success"]) if ep.chunks else False
    done = [official if s["done"] is None else bool(s["done"](e)) for s in st]
    if official and ep.task in LATCHED:
        done = [True] * len(done)
    stage = next((i for i, d in enumerate(done) if not d), len(done))
    info = INFO[ep.task](e) if ep.task in INFO else {}
    return dict(t=ep.t, stage=stage, done=done, official=official, ever=bool(ep.success), info=info)


def instructions(ep):
    return [s["instr"](ep.kenv) for s in STAGES[ep.task]]
