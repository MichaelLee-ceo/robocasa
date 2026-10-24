"""Atomic steps of 5 composite_seen tasks (the other 11 are in atomic_steps_seen11.py, merged into TASKS at the end).

The step sequence is the official RoboCasa365 per-frame subtask annotation (annotation.human.subtask_name of the
composite target datasets, NavigateKitchen dropped); checked on 2026-10-25 to equal the annotated sequence of every
demo (>= 99% of demos in all 16 tasks). The per-step checks are ours: the official composite _check_success() split
into the conjuncts each step is responsible for, using the same official OU.* / fixture predicates.

For every step:
  * skill        : the RoboCasa atomic task it corresponds to (same name as the subtask annotation)
  * instr(env,S) : instruction in the atomic task's own template (robocasa/environments/kitchen/atomic/*.py get_ep_meta),
                   with object / location words filled in for this scene
  * check(env,S) : success = the conjuncts of the composite task's official _check_success() that this step is
                   responsible for (the atomic tasks use the same OU.* predicates); the last step uses env._check_success()
  * seg          : index of the step's first annotated segment (pick / execute) in the demo -> reset start frame
S is a per-episode dict filled by `prepare(env, demo)` (e.g. which bowl is picked first).
"""
import numpy as np
import robocasa.utils.object_utils as OU


def _lang(env, name):
    return OU.get_obj_lang(env, obj_name=name)


def _burner_under(env, obj="obj"):
    pos = env.sim.data.body_xpos[env.obj_body_id[env.objects[obj].name]][:2]
    best, bd = None, 1e9
    for loc, site in env.stove.burner_sites.items():
        if site is None:
            continue
        d = np.linalg.norm(np.array(env.sim.data.get_site_xpos(site.get("name")))[:2] - pos)
        if d < bd:
            best, bd = loc, d
    return best, bd


def _released(env, obj):
    """gripper not touching the object (used instead of OU.gripper_obj_far when the next step manipulates the same
    object cluster, e.g. vegetable inside the bowl that is picked next)."""
    return not env.check_contact(env.robots[0].gripper["right"], env.objects[obj])


def _first_bowl_in(env, S):
    """demo-start runs: S['first'] comes from the demo. Seeded runs (no demo): whichever bowl is placed first."""
    cands = [S["first"]] if S.get("first") else ["bowl1", "bowl2"]
    for b in cands:
        if OU.obj_inside_of(env, b, env.cabinet) and _released(env, b):
            S["first"], S["second"] = b, ("bowl2" if b == "bowl1" else "bowl1")
            return True
    return False


def _kettle_on_burner(env):
    if not OU.check_obj_fixture_contact(env, "obj", env.stove):
        return False
    _, d = _burner_under(env)
    return d < 0.15 and OU.gripper_obj_far(env, "obj")


TASKS = {
    "KettleBoiling": [
        dict(skill="PickPlaceCounterToStove", seg=0,
             instr=lambda e, S: f"Pick the {_lang(e, 'obj')} from the counter and place it on the stove.",
             check=lambda e, S: _kettle_on_burner(e)),
        dict(skill="TurnOnStove", seg=2,
             debug=lambda e, S: dict(burner_under_kettle=_burner_under(e)[0], dist=round(float(_burner_under(e)[1]), 3),
                                     on_stove=bool(OU.check_obj_fixture_contact(e, "obj", e.stove)),
                                     on_pan=bool(e.check_contact(e.objects["obj"], e.objects["stove_distr"]))),
             instr=lambda e, S: f"Turn on the {(_burner_under(e)[0] or 'front').replace('_', ' ')} burner of the stove.",
             check=lambda e, S: e._check_success()),
    ],
    "PreSoakPan": [
        dict(skill="PickPlaceCounterToSink", seg=0,
             instr=lambda e, S: f"Pick the {_lang(e, 'obj1')} from the counter and place it in the sink.",
             check=lambda e, S: OU.obj_inside_of(e, "obj1", e.sink, partial_check=False) and OU.gripper_obj_far(e, "obj1")),
        dict(skill="PickPlaceCounterToSink", seg=2,
             instr=lambda e, S: f"Pick the {_lang(e, 'obj2')} from the counter and place it in the sink.",
             check=lambda e, S: OU.obj_inside_of(e, "obj2", e.sink, partial_check=False) and OU.gripper_obj_far(e, "obj2")),
        dict(skill="TurnOnSinkFaucet", seg=4,
             instr=lambda e, S: "Turn on the sink faucet.",
             check=lambda e, S: e._check_success()),
    ],
    "StackBowlsCabinet": [
        dict(skill="PickPlaceCounterToCabinet", seg=0,
             instr=lambda e, S: f"Pick the {_lang(e, S.get('first') or 'bowl1')} from the counter and place it in the cabinet.",
             check=lambda e, S: _first_bowl_in(e, S)),
        dict(skill="PickPlaceCounterToCabinet", seg=2,
             instr=lambda e, S: f"Pick the {_lang(e, S.get('second') or 'bowl2')} from the counter and place it in the cabinet.",
             instr_aug=lambda e, S: "Pick the other bowl and stack it on top of the bowl in the cabinet.",
             check=lambda e, S: e._check_success()),
    ],
    "SteamInMicrowave": [
        dict(skill="PickPlaceSinkToCounter", seg=0,
             instr=lambda e, S: f"Pick the {_lang(e, 'vegetable')} from the sink and place it on the {_lang(e, 'bowl')} located on the counter.",
             check=lambda e, S: OU.check_obj_in_receptacle(e, "vegetable", "bowl") and _released(e, "vegetable")),
        dict(skill="PickPlaceCounterToMicrowave", seg=2,  # pick, navigate, place
             instr=lambda e, S: f"Pick the {_lang(e, 'bowl')} from the counter and place it in the microwave.",
             instr_aug=lambda e, S: "Pick up the bowl, carry it to the microwave and place it inside.",
             check=lambda e, S: OU.obj_inside_of(e, "bowl", e.microwave) and OU.gripper_obj_far(e, "bowl")),
        dict(skill="CloseMicrowave", seg=5,
             instr=lambda e, S: "Close the microwave door.",
             check=lambda e, S: e.microwave.is_closed(env=e)),
        dict(skill="TurnOnMicrowave", seg=6,
             instr=lambda e, S: "Press the start button on the microwave.",
             check=lambda e, S: e._check_success()),
    ],
    "PrepareCoffee": [
        dict(skill="CoffeeSetupMug", seg=0,
             instr=lambda e, S: f"Pick the {_lang(e, 'obj')} from the cabinet and place it under the coffee machine dispenser.",
             check=lambda e, S: e.coffee_machine.check_receptacle_placement_for_pouring(e, "obj") and OU.gripper_obj_far(e, "obj")),
        dict(skill="StartCoffeeMachine", seg=2,
             instr=lambda e, S: "Press the button on the coffee machine to serve coffee.",
             check=lambda e, S: e._check_success()),
    ],
}

# what the atomic template literally says (for the report), when we had to change a location word
TEMPLATE_NOTES = {
    "PickPlaceCounterToStove": 'atomic template: "Pick the {obj} from the plate and place it in the pan." -> source/target words adapted',
    "CoffeeSetupMug": 'atomic template: "Pick the {obj} from the counter and place it under the coffee machine dispenser." -> "counter" changed to "cabinet"',
    "PickPlaceCounterToCabinet": "atomic template has no notion of stacking; the second step relies on the policy to stack",
}


def prepare(env, demo, set_state):
    """Per-episode info derived from the demo (call before rebuilding the scene)."""
    S = {}
    if demo["task"] == "StackBowlsCabinet":
        segs = demo["segs"]
        pe = segs[0]["end"]
        set_state(env, demo["states"][min(pe, len(demo["states"]) - 1)])
        p1 = {n: env.sim.data.body_xpos[env.obj_body_id[env.objects[n].name]].copy() for n in ("bowl1", "bowl2")}
        set_state(env, demo["states"][0])
        p0 = {n: env.sim.data.body_xpos[env.obj_body_id[env.objects[n].name]].copy() for n in ("bowl1", "bowl2")}
        moved = {n: float(np.linalg.norm(p1[n] - p0[n])) for n in p0}
        S["first"] = max(moved, key=moved.get)
        S["second"] = "bowl2" if S["first"] == "bowl1" else "bowl1"
    return S

# --- pretrain-only composite tasks whose two steps are both Atomic-Seen skills -------------------------------------
def _pair_steps(skill, objs, done, instr):
    """two steps on objects A, B in either order: step 1 passes when either object is done (and released); the
    step-2 sentence then names the other one."""
    def first_done(e, S):
        for o in (S["first"],) if S.get("first") else objs:
            if done(e, o) and OU.gripper_obj_far(e, obj_name=o):
                S["first"], S["second"] = o, [x for x in objs if x != o][0]
                return True
        return False
    return [dict(skill=skill, seg=0, instr=lambda e, S: instr(e, S.get("first") or objs[0]), check=first_done),
            dict(skill=skill, seg=2, instr=lambda e, S: instr(e, S.get("second") or objs[1]), check=lambda e, S: e._check_success())]


_sink_to_board = lambda e, o: f"Pick the {_lang(e, o)} from the sink and place it on the {_lang(e, 'cutting_board')} located on the counter."
_counter_to_cab = lambda e, o: f"Pick the {_lang(e, o)} from the counter and place it in the cabinet."
TASKS["ArrangeVegetables"] = _pair_steps("PickPlaceSinkToCounter", ["vegetable1", "vegetable2"],
                                         lambda e, o: OU.check_obj_in_receptacle(e, o, "cutting_board"), _sink_to_board)
TASKS["GatherVegetables"] = _pair_steps("PickPlaceSinkToCounter", ["veg1", "veg2"],
                                        lambda e, o: OU.check_obj_in_receptacle(e, o, "cutting_board"), _sink_to_board)
TASKS["CondimentCollection"] = _pair_steps("PickPlaceCounterToCabinet", ["condiment1", "condiment2"],
                                           lambda e, o: OU.obj_inside_of(e, o, e.cab), _counter_to_cab)



# The other 11 composite-seen tasks (ASUS-GPU, 2026-10-14): steps from the dataset subtask annotations, checks validated
# by demo replay (runs/prev/_handoff_verify_asus/seen11/README.md). Every task's last step is env._check_success().
from atomic_steps_seen11 import TASKS as _TASKS_SEEN11  # noqa: E402
TASKS.update(_TASKS_SEEN11)
