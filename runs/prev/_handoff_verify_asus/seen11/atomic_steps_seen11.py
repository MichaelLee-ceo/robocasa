"""Atomic-step decomposition of the remaining 11 composite_seen target tasks (same entry format as runs/scripts/atomic_steps.py).

For every step:
  * skill        : the subtask annotation name (annotation.human.subtask_name) of that step in the majority annotation sequence
  * instr(env,S) : atomic-template instruction where a RoboCasa atomic task with that name exists, else the annotation sentence
  * check(env,S) : the conjuncts of the composite task's official _check_success() that this step is responsible for
                   (same OU.* / fixture predicates); the last step uses env._check_success()
  * seg          : index of the step's first annotated segment in the MAJORITY stage sequence of the dataset
                   (NavigateKitchen segments included). The number/position of NavigateKitchen segments varies between
                   demos in several tasks, so for a given demo use step_segments(demo) below instead of segs[seg].
S is a per-episode dict; all bookkeeping (which object went first, latches) is filled online by the checks themselves,
so the checks also work for seeded (non-demo) episodes. Start every episode with S = {}.

Some composite success conditions accumulate Python-level state in env.update_state() / env._check_success()
(ScrubCuttingBoard contact timer, WashLettuce washed_time, RinseSinkBasin washed_loc, StirVegetables success_time,
GetToastedBread toaster_on + the Toaster fixture's own timers). In a live rollout env.step() runs update_state() and
reward() -> _check_success() every control step, so these are maintained automatically. When replaying demo states,
call set_state() (which runs update_state) AND env._check_success() once per frame, and rebuild the scene with reset_to()
before replaying (Python state is not part of the MuJoCo state).
"""
import numpy as np
import robocasa.utils.object_utils as OU


def _lang(env, name):
    return OU.get_obj_lang(env, obj_name=name)


def _released(env, obj):
    """gripper not touching the object."""
    return not env.check_contact(env.robots[0].gripper["right"], env.objects[obj])


def _picked(env, obj, fixture):
    """object grasped and lifted off its support fixture."""
    return OU.check_obj_grasped(env, obj) and not OU.check_obj_fixture_contact(env, obj, fixture)


def _held_lifted(env, obj, fixture):
    """gripper touching the object and the object off its support fixture. Used for the sponge: OU.check_obj_grasped
    (both finger joints < 0.035) misses many real sponge holds because the fingers stay partly open on the soft sponge."""
    return (env.check_contact(env.robots[0].gripper["right"], env.objects[obj])
            and not OU.check_obj_fixture_contact(env, obj, fixture))


def _water_on(env):
    return bool(env.sink.get_handle_state(env=env)["water_on"])


def _order_step(objs, done, released, k):
    """step k (0-based) of a set of 'same skill on several objects, any order' steps: passes when >= k+1 of `objs`
    are done and released. Records the completion order in S['order'] (used for the instructions)."""
    def check(e, S):
        order = S.setdefault("order", [])
        for o in objs:
            if o not in order and done(e, o) and released(e, o):
                order.append(o)
        return len([o for o in order if done(e, o)]) >= k + 1
    return check


def _next_obj(objs, k):
    """object the k-th order step is (most likely) about: completed one if known, else the first not yet done."""
    def f(e, S):
        order = S.get("order", [])
        if len(order) > k:
            return order[k]
        rest = [o for o in objs if o not in order]
        return rest[0] if rest else objs[-1]
    return f


# ---------------------------------------------------------------- per-task helpers
def _pan_on_knob_burner(e):
    """SearingMeat: pan on the stove, within 0.15 m (xy) of the burner of the task's knob (the official check uses
    stove.check_obj_location_on_stove(pan, 0.15) == knob, which additionally needs that burner on -> step 3)."""
    if not OU.check_obj_fixture_contact(e, "pan", e.stove):
        return False
    site = e.stove.burner_sites.get(e.knob)
    if site is None:
        return False
    pos = e.sim.data.body_xpos[e.obj_body_id["pan"]][:2]
    return np.linalg.norm(np.array(e.sim.data.get_site_xpos(site.get("name")))[:2] - pos) < 0.15


def _toast_done(e, S):
    """GetToastedBread step 1 (TurnOnToaster press + wait, merged): a toaster slot pair that held the bread was switched
    on (latched) and has since finished and popped up (turned_on False again and its lever back at <= 0.1, which is
    where the annotated 'wait' stage ends).
    The bread-in-slot condition is latched from check_slot_contact at any earlier frame (and cleared if the bread is
    grasped): in replayed demo states the bread often loses contact with the slot floor while the lever is down, so
    requiring contact at the moment of switching on (as the composite _check_success does) misses real turn-ons."""
    st = e.toaster.get_state(e)
    for i, sp in enumerate(st.keys()):
        if e.toaster.check_slot_contact(e, "obj", slot_pair=i):
            S["bread_slot"] = i
    if OU.check_obj_grasped(e, "obj") and "toast_slot" not in S:
        S.pop("bread_slot", None)
    b = S.get("bread_slot")
    if b is not None and e.toaster.get_state(e, slot_pair=b)["turned_on"]:
        S["toast_slot"] = b
    if "toast_slot" not in S:
        return False
    s2 = e.toaster.get_state(e, slot_pair=S["toast_slot"])
    return (not s2["turned_on"]) and float(s2["lever"]) <= 0.1


def _drawer_open(e):
    """OpenDrawer atomic success: every joint of the drawer >= 0.95 open (kitchen_drawer.ManipulateDrawer)."""
    return all(v >= 0.95 for v in e.drawer.get_door_state(env=e).values())


_LUNCH = ["vegetable0", "vegetable1", "meat0", "meat1"]


def _in_tupperware(e, o):
    return OU.check_obj_in_receptacle(e, o, "tupperware0") or OU.check_obj_in_receptacle(e, o, "tupperware1")


TASKS = {
    "ScrubCuttingBoard": [
        dict(skill="PickUpSponge", seg=0,
             instr=lambda e, S: f"Pick up the {_lang(e, 'sponge')} from the counter.",
             check=lambda e, S: _held_lifted(e, "sponge", e.counter)),
        dict(skill="ScrubCuttingBoard", seg=1,
             instr=lambda e, S: f"Scrub the {_lang(e, 'cutting_board')} with the sponge, then release the sponge.",
             check=lambda e, S: e._check_success()),
    ],
    "WashLettuce": [
        dict(skill="TurnOnSinkFaucet", seg=0,
             instr=lambda e, S: "Turn on the sink faucet.",
             check=lambda e, S: _water_on(e)),
        dict(skill="PickPlaceCounterToSink", seg=1,
             instr=lambda e, S: f"Pick the {_lang(e, 'lettuce')} from the counter and place it in the sink.",
             check=lambda e, S: e._check_success()),
    ],
    "RinseSinkBasin": [
        dict(skill="TurnOnSinkFaucet", seg=0,
             instr=lambda e, S: "Turn on the sink faucet.",
             check=lambda e, S: _water_on(e)),
        dict(skill="TurnSinkSpout", seg=1,
             instr=lambda e, S: "Turn the sink spout to the left, center and right to wash all locations of the sink basin.",
             check=lambda e, S: e._check_success()),
    ],
    "LoadDishwasher": [
        dict(skill="PickPlaceCounterToDishwasher", seg=0,
             instr=lambda e, S: f"Pick the {_lang(e, _next_obj(['dish0', 'dish1'], 0)(e, S))} from the counter and place it in the dishwasher.",
             check=_order_step(["dish0", "dish1"], lambda e, o: e.dishwasher.check_rack_contact(e, o), _released, 0)),
        dict(skill="PickPlaceCounterToDishwasher", seg=2,
             instr=lambda e, S: f"Pick the {_lang(e, _next_obj(['dish0', 'dish1'], 1)(e, S))} from the counter and place it in the dishwasher.",
             check=_order_step(["dish0", "dish1"], lambda e, o: e.dishwasher.check_rack_contact(e, o), _released, 1)),
        dict(skill="CloseDishwasher", seg=4,
             instr=lambda e, S: "Close the dishwasher door.",
             check=lambda e, S: e._check_success()),
    ],
    "StoreLeftoversInBowl": [
        dict(skill="PickPlaceCounterToCounter", seg=0,
             instr=lambda e, S: f"Pick the {_lang(e, _next_obj(['chicken_drumstick', 'vegetable'], 0)(e, S))} from the plate and place it in the bowl.",
             check=_order_step(["chicken_drumstick", "vegetable"], lambda e, o: OU.check_obj_in_receptacle(e, o, "bowl"), _released, 0)),
        dict(skill="PickPlaceCounterToCounter", seg=2,
             instr=lambda e, S: f"Pick the {_lang(e, _next_obj(['chicken_drumstick', 'vegetable'], 1)(e, S))} from the plate and place it in the bowl.",
             check=_order_step(["chicken_drumstick", "vegetable"], lambda e, o: OU.check_obj_in_receptacle(e, o, "bowl"), _released, 1)),
        dict(skill="PickPlaceCounterToFridge", seg=4,
             instr=lambda e, S: f"Pick the {_lang(e, 'bowl')} from the counter and place it in the fridge.",
             check=lambda e, S: e._check_success()),
    ],
    "SetUpCuttingStation": [
        dict(skill="PickPlaceDrawerToCounter", seg=0,
             instr=lambda e, S: f"Pick the {_lang(e, 'knife')} from the drawer and place it on the {_lang(e, 'receptacle')}.",
             check=lambda e, S: OU.check_obj_in_receptacle(e, "knife", "receptacle") and _released(e, "knife")),
        dict(skill="PickPlaceCounterToCounter", seg=2,
             instr=lambda e, S: f"Pick the {_lang(e, 'meat')} from the plate and place it on the {_lang(e, 'receptacle')}.",
             check=lambda e, S: e._check_success()),
    ],
    "StirVegetables": [
        dict(skill="PickPlaceCounterToStove", seg=0,
             instr=lambda e, S: f"Pick the {_lang(e, _next_obj(['veg1', 'veg2'], 0)(e, S))} from the counter and place it in the {_lang(e, 'pot')}.",
             check=_order_step(["veg1", "veg2"], lambda e, o: OU.check_obj_in_receptacle(e, o, "pot"), _released, 0)),
        dict(skill="PickPlaceCounterToStove", seg=2,
             instr=lambda e, S: f"Pick the {_lang(e, _next_obj(['veg1', 'veg2'], 1)(e, S))} from the counter and place it in the {_lang(e, 'pot')}.",
             check=_order_step(["veg1", "veg2"], lambda e, o: OU.check_obj_in_receptacle(e, o, "pot"), _released, 1)),
        dict(skill="PickUpSpatula", seg=4,
             instr=lambda e, S: f"Pick up the {_lang(e, 'spatula')} from the counter.",
             check=lambda e, S: _picked(e, "spatula", e.counter)),
        dict(skill="StirVegetables", seg=5,
             instr=lambda e, S: "Stir the vegetables in the pot with the spatula.",
             check=lambda e, S: e._check_success()),
    ],
    "SearingMeat": [
        dict(skill="PickPlaceCabinetToStove", seg=0,
             instr=lambda e, S: f"Pick the {_lang(e, 'pan')} from the cabinet and place it on the {e.knob.replace('_', ' ')} burner of the stove.",
             check=lambda e, S: _pan_on_knob_burner(e) and OU.gripper_obj_far(e, "pan")),
        dict(skill="PickPlaceCounterToStove", seg=3,
             instr=lambda e, S: f"Pick the {_lang(e, 'meat')} from the container and place it in the {_lang(e, 'pan')}.",
             check=lambda e, S: OU.check_obj_in_receptacle(e, "meat", "pan", th=0.07) and _released(e, "meat")),
        dict(skill="TurnOnStove", seg=5,
             instr=lambda e, S: f"Turn on the {e.knob.replace('_', ' ')} burner of the stove.",
             check=lambda e, S: e._check_success()),
    ],
    "PackIdenticalLunches": [
        dict(skill="PickPlaceFridgeToCounter", seg=4 * k,  # majority stage seq: [pick,nav,place,nav] x 3 + [pick,nav,place]
             instr=(lambda kk: lambda e, S: f"Pick the {_lang(e, _next_obj(_LUNCH, kk)(e, S))} from the fridge and place it in an empty spot of a tupperware on the counter.")(k),
             check=_order_step(_LUNCH, _in_tupperware, OU.gripper_obj_far, k))
        for k in range(3)
    ] + [
        dict(skill="PickPlaceFridgeToCounter", seg=12,
             instr=lambda e, S: f"Pick the {_lang(e, _next_obj(_LUNCH, 3)(e, S))} from the fridge and place it in the tupperware that is still missing it.",
             check=lambda e, S: e._check_success()),
    ],
    "DeliverStraw": [
        dict(skill="OpenDrawer", seg=0,
             instr=lambda e, S: "Open the drawer.",
             check=lambda e, S: _drawer_open(e)),
        dict(skill="PickPlaceDrawerToCounter", seg=1,
             instr=lambda e, S: f"Pick the {_lang(e, 'straw')} from the drawer and place it in the {_lang(e, 'glass_cup')} on the dining counter.",
             check=lambda e, S: e._check_success()),
    ],
    "GetToastedBread": [
        dict(skill="TurnOnToaster", seg=0,
             instr=lambda e, S: "Push down the lever of the toaster to turn it on.",
             check=_toast_done),
        dict(skill="PickPlaceToasterToCounter", seg=2,
             instr=lambda e, S: "Place the toasted item on a plate.",
             check=lambda e, S: e._check_success()),
    ],
}

def step_segments(demo):
    """(start, end) frame of every step in THIS demo: NavigateKitchen dropped, adjacent same-name segments merged, a new
    step starts when a 'pick' stage follows a 'place'/'execute' stage of the same name. Returns a list of dicts
    (skill, start, end, seg_indices) without the final 'done' segment."""
    units, last_stage = [], None
    for i, s in enumerate(demo["segs"]):
        if s["skill"] == "NavigateKitchen":  # not part of any step (a unit's end = end of its last non-navigate segment)
            continue
        if s["stage"] == "done":
            break
        if not units or units[-1]["skill"] != s["skill"] or (s["stage"] == "pick" and last_stage in ("place", "execute")):
            units.append(dict(skill=s["skill"], start=s["start"], end=s["end"], seg_indices=[i]))
        else:
            units[-1]["end"] = s["end"]; units[-1]["seg_indices"].append(i)
        last_stage = s["stage"]
    return units


def prepare(env, demo, set_state):
    """kept for interface compatibility with atomic_eval.py; all per-episode info is filled online."""
    return {}
