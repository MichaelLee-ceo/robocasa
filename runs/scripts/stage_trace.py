"""Stage-level trace and root cause of a rollout, from its saved per-step sim states (atomic_eval.py save_steps).

For every control step the episode is replayed (oracle.set_state + env._check_success(), so Python-level latches and
timers advance as in the rollout) and recorded:
  * every stage check of atomic_steps.TASKS, evaluated independently (order-free);
  * for the objects a stage moves: position, gripper contact, OU.check_obj_grasped, whether it is at the stage's
    target (the official predicate used by the stage check, without the release part), the distance to that target,
    the eef-object distance and OU.gripper_obj_far;
  * for fixture stages: gripper-fixture contact and the official fixture state (knob angles, water on, door closed,
    drawer joints, toaster / microwave / coffee machine state);
  * the env's own progress counters for process stages (scrub contacts, stir / wash time, rinsed spout directions).
Each stage that is not complete at the end gets a verdict on where the manipulation broke (see verdict_* below).
STAGE_SPECS only says which objects / fixture a stage is about and which official predicate means "at target"; the
stage checks themselves stay in atomic_steps.py."""
import numpy as np
import robocasa.utils.object_utils as OU
from atomic_steps import TASKS

FLOOR_Z = 0.35  # object centre below this = on the floor (counters and fixtures are ~0.8-1.0 m high)


def _in_rec(o, rec, th=None):
    return lambda e, oo=o: OU.check_obj_in_receptacle(e, oo, rec, th=th) if th is not None else OU.check_obj_in_receptacle(e, oo, rec)


def _dist_to_body(e, o, other):
    """xy distance between two object centres (the quantity check_obj_in_receptacle thresholds)."""
    a = e.sim.data.body_xpos[e.obj_body_id[o]][:2]
    b = e.sim.data.body_xpos[e.obj_body_id[other]][:2]
    return float(np.linalg.norm(a - b))


def _dist_to_region(e, o, fixture):
    """distance from the object centre to the fixture's official interior region(s) (get_int_sites), 0 if inside."""
    p = np.array(e.sim.data.body_xpos[e.obj_body_id[o]])
    best = 1e9
    for p0, px, py, pz in fixture.get_int_sites(relative=False).values():
        d2 = 0.0
        for q in (px, py, pz):
            u = q - p0
            L = np.linalg.norm(u)
            t = np.dot(p - p0, u / L)
            d2 += max(0.0, -t, t - L) ** 2
        best = min(best, np.sqrt(d2))
    return float(best)


def _burner_dist(e, o, loc=None):
    """xy distance from the object to a burner site (the given one, else the nearest)."""
    pos = e.sim.data.body_xpos[e.obj_body_id[o]][:2]
    ds = {l: float(np.linalg.norm(np.array(e.sim.data.get_site_xpos(s.get("name")))[:2] - pos))
          for l, s in e.stove.burner_sites.items() if s is not None}
    return ds[loc] if loc is not None else min(ds.values())


def _nearest_burner(e, o):
    pos = e.sim.data.body_xpos[e.obj_body_id[o]][:2]
    ds = {l: float(np.linalg.norm(np.array(e.sim.data.get_site_xpos(s.get("name")))[:2] - pos))
          for l, s in e.stove.burner_sites.items() if s is not None}
    loc = min(ds, key=ds.get)
    return loc, round(ds[loc], 3)


def _knob_onness(e, loc):
    a = abs(e.stove.get_knobs_state(env=e)[loc])
    return float(min(a, 2 * np.pi - a))


def pnp(objs, at_target, dist, release="far"):
    return dict(kind="pnp", objs=objs, at_target=at_target, dist=dist, release=release)


def fixture(name, state, done, value):
    """value(e): scalar official state the stage moves (knob angle, normalized door / drawer joint, lever, on flag)."""
    return dict(kind="fixture", fixture=name, state=state, done=done, value=value)


def _door(e, fx, joint):
    return float(fx.get_joint_state(e, [joint])[joint])


STAGE_SPECS = {
    "KettleBoiling": [
        pnp(["obj"], lambda e, o: OU.check_obj_fixture_contact(e, o, e.stove) and _burner_dist(e, o) < 0.15,
            lambda e, o: _burner_dist(e, o)),
        fixture("stove", lambda e: dict(knob_angles={l: round(_knob_onness(e, l), 3) for l in e.stove.get_knobs_state(env=e)},
                                        kettle_nearest_burner=_nearest_burner(e, "obj")),
                lambda e: any(e.stove.is_burner_on(env=e, burner_loc=l) for l in e.stove.get_knobs_state(env=e)),
                lambda e: max(_knob_onness(e, l) for l in e.stove.get_knobs_state(env=e))),
    ],
    "PreSoakPan": [
        pnp(["obj1", "obj2"], lambda e, o: OU.obj_inside_of(e, o, e.sink, partial_check=False), lambda e, o: _dist_to_region(e, o, e.sink)),
        pnp(["obj1", "obj2"], lambda e, o: OU.obj_inside_of(e, o, e.sink, partial_check=False), lambda e, o: _dist_to_region(e, o, e.sink)),
        fixture("sink", lambda e: {k: v for k, v in e.sink.get_handle_state(env=e).items() if k in ("handle_joint", "water_on")},
                lambda e: bool(e.sink.get_handle_state(env=e)["water_on"]), lambda e: abs(float(e.sink.get_handle_state(env=e)["handle_joint"]))),
    ],
    "StackBowlsCabinet": [
        pnp(["bowl1", "bowl2"], lambda e, o: OU.obj_inside_of(e, o, e.cabinet), lambda e, o: _dist_to_region(e, o, e.cabinet), release="touch"),
        pnp(["bowl1", "bowl2"], lambda e, o: OU.obj_inside_of(e, o, e.cabinet) and (OU.check_obj_in_receptacle(e, "bowl1", "bowl2") or OU.check_obj_in_receptacle(e, "bowl2", "bowl1")),
            lambda e, o: _dist_to_region(e, o, e.cabinet)),
    ],
    "SteamInMicrowave": [
        pnp(["vegetable"], lambda e, o: OU.check_obj_in_receptacle(e, o, "bowl"), lambda e, o: _dist_to_body(e, o, "bowl"), release="touch"),
        pnp(["bowl"], lambda e, o: OU.obj_inside_of(e, o, e.microwave), lambda e, o: _dist_to_region(e, o, e.microwave)),
        fixture("microwave", lambda e: dict(closed=bool(e.microwave.is_closed(env=e))), lambda e: bool(e.microwave.is_closed(env=e)),
                lambda e: 1.0 - _door(e, e.microwave, e.microwave.door_joint_names[0])),
        fixture("microwave", lambda e: dict(turned_on=bool(e.microwave.get_state()["turned_on"]), closed=bool(e.microwave.is_closed(env=e))),
                lambda e: bool(e.microwave.get_state()["turned_on"]), lambda e: float(e.microwave.get_state()["turned_on"])),
    ],
    "PrepareCoffee": [
        pnp(["obj"], lambda e, o: e.coffee_machine.check_receptacle_placement_for_pouring(e, o),
            lambda e, o: float(np.linalg.norm(e.sim.data.body_xpos[e.obj_body_id[o]][:2]
                                              - e.sim.data.site_xpos[e.sim.model.site_name2id(f"{e.coffee_machine.naming_prefix}receptacle_place_site")][:2]))),
        fixture("coffee_machine", lambda e: dict(turned_on=bool(e.coffee_machine.get_state()["turned_on"])),
                lambda e: bool(e.coffee_machine.get_state()["turned_on"]), lambda e: float(e.coffee_machine.get_state()["turned_on"])),
    ],
    "ScrubCuttingBoard": [
        dict(kind="pick", objs=["sponge"]),
        dict(kind="process", objs=["sponge"], progress=lambda e: dict(
            contacts=len(e.board_contact_positions),
            sweep=round(float(np.linalg.norm(np.ptp(np.array(e.board_contact_positions), axis=0))) if e.board_contact_positions else 0.0, 3))),
    ],
    "WashLettuce": [
        fixture("sink", lambda e: {k: v for k, v in e.sink.get_handle_state(env=e).items() if k in ("handle_joint", "water_on")},
                lambda e: bool(e.sink.get_handle_state(env=e)["water_on"]), lambda e: abs(float(e.sink.get_handle_state(env=e)["handle_joint"]))),
        dict(kind="process", objs=["lettuce"], progress=lambda e: dict(washed_time=int(e.washed_time),
                                                                    water_on=bool(e.sink.get_handle_state(env=e)["water_on"]))),
    ],
    "RinseSinkBasin": [
        fixture("sink", lambda e: {k: v for k, v in e.sink.get_handle_state(env=e).items() if k in ("handle_joint", "water_on")},
                lambda e: bool(e.sink.get_handle_state(env=e)["water_on"]), lambda e: abs(float(e.sink.get_handle_state(env=e)["handle_joint"]))),
        dict(kind="process", objs=[], fixture="sink", progress=lambda e: dict(washed_loc=[bool(x) for x in e.washed_loc],
                                                                              spout_ori=e.sink.get_handle_state(env=e)["spout_ori"])),
    ],
    "LoadDishwasher": [
        pnp(["dish0", "dish1"], lambda e, o: e.dishwasher.check_rack_contact(e, o), lambda e, o: _dist_to_region(e, o, e.dishwasher), release="touch"),
        pnp(["dish0", "dish1"], lambda e, o: e.dishwasher.check_rack_contact(e, o), lambda e, o: _dist_to_region(e, o, e.dishwasher), release="touch"),
        fixture("dishwasher", lambda e: dict(closed=bool(e.dishwasher.is_closed(e, th=0.05))), lambda e: bool(e.dishwasher.is_closed(e, th=0.05)),
                lambda e: 1.0 - _door(e, e.dishwasher, e.dishwasher._joint_names["door"])),
    ],
    "StoreLeftoversInBowl": [
        pnp(["chicken_drumstick", "vegetable"], lambda e, o: OU.check_obj_in_receptacle(e, o, "bowl"), lambda e, o: _dist_to_body(e, o, "bowl"), release="touch"),
        pnp(["chicken_drumstick", "vegetable"], lambda e, o: OU.check_obj_in_receptacle(e, o, "bowl"), lambda e, o: _dist_to_body(e, o, "bowl"), release="touch"),
        pnp(["bowl"], lambda e, o: e.fridge.check_rack_contact(e, o), lambda e, o: _dist_to_region(e, o, e.fridge)),
    ],
    "SetUpCuttingStation": [
        pnp(["knife"], lambda e, o: OU.check_obj_in_receptacle(e, o, "receptacle"), lambda e, o: _dist_to_body(e, o, "receptacle"), release="touch"),
        pnp(["meat"], lambda e, o: OU.check_obj_in_receptacle(e, o, "receptacle"), lambda e, o: _dist_to_body(e, o, "receptacle")),
    ],
    "StirVegetables": [
        pnp(["veg1", "veg2"], lambda e, o: OU.check_obj_in_receptacle(e, o, "pot"), lambda e, o: _dist_to_body(e, o, "pot"), release="touch"),
        pnp(["veg1", "veg2"], lambda e, o: OU.check_obj_in_receptacle(e, o, "pot"), lambda e, o: _dist_to_body(e, o, "pot"), release="touch"),
        dict(kind="pick", objs=["spatula"]),
        dict(kind="process", objs=["spatula"], progress=lambda e: dict(success_time=int(e.success_time))),
    ],
    "SearingMeat": [
        pnp(["pan"], lambda e, o: OU.check_obj_fixture_contact(e, o, e.stove) and _burner_dist(e, o, e.knob) < 0.15,
            lambda e, o: _burner_dist(e, o, e.knob)),
        pnp(["meat"], lambda e, o: OU.check_obj_in_receptacle(e, o, "pan", th=0.07), lambda e, o: _dist_to_body(e, o, "pan"), release="touch"),
        fixture("stove", lambda e: dict(knob_angles={l: round(_knob_onness(e, l), 3) for l in e.stove.get_knobs_state(env=e)},
                                        task_knob=e.knob, pan_nearest_burner=_nearest_burner(e, "pan")),
                lambda e: bool(e.stove.is_burner_on(env=e, burner_loc=e.knob)), lambda e: _knob_onness(e, e.knob)),
    ],
    "PackIdenticalLunches": [
        pnp(["vegetable0", "vegetable1", "meat0", "meat1"],
            lambda e, o: OU.check_obj_in_receptacle(e, o, "tupperware0") or OU.check_obj_in_receptacle(e, o, "tupperware1"),
            lambda e, o: min(_dist_to_body(e, o, "tupperware0"), _dist_to_body(e, o, "tupperware1")))
        for _ in range(4)
    ],
    "DeliverStraw": [
        fixture("drawer", lambda e: {k: round(float(v), 3) for k, v in e.drawer.get_door_state(env=e).items()},
                lambda e: all(v >= 0.95 for v in e.drawer.get_door_state(env=e).values()),
                lambda e: float(min(e.drawer.get_door_state(env=e).values()))),
        pnp(["straw"], lambda e, o: OU.check_obj_in_receptacle(e, o, "glass_cup", th=0.5), lambda e, o: _dist_to_body(e, o, "glass_cup")),
    ],
    "GetToastedBread": [
        fixture("toaster", lambda e: {str(k): dict(turned_on=bool(v["turned_on"]), lever=round(float(v.get("lever", 0.0)), 3))
                                      for k, v in e.toaster.get_state(e).items()},
                lambda e: any(v["turned_on"] for v in e.toaster.get_state(e).values()),
                lambda e: max(float(v.get("lever", 0.0)) for v in e.toaster.get_state(e).values())),
        pnp(["obj"], lambda e, o: OU.check_obj_in_receptacle(e, o, "plate"), lambda e, o: _dist_to_body(e, o, "plate")),
    ],
}


def _geom_ids(env, prefix):
    m = env.sim.model
    return {i for i in range(m.ngeom) if (m.geom_id2name(i) or "").startswith(prefix)}


def _touching(env, a, b):
    """any active contact between a geom in id-set a and one in id-set b."""
    d = env.sim.data
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in a and c.geom2 in b) or (c.geom1 in b and c.geom2 in a):
            return True
    return False


def trace_episode(env, task, states, set_state):
    """Replay `states` (env already reset to the episode's scene) and return per-frame records."""
    steps, specs = TASKS[task], STAGE_SPECS[task]
    objs = sorted({o for sp in specs for o in sp.get("objs", [])})
    fixtures = sorted({sp["fixture"] for sp in specs if sp.get("fixture")})
    grip = env.robots[0].gripper["right"]
    eef_site = env.robots[0].eef_site_id["right"]
    grip_geoms = _geom_ids(env, grip.naming_prefix)
    finger_qadr = [env.sim.model.get_joint_qpos_addr(j) for j in grip.joints]  # as the official KettleBoiling labeler
    fx_geoms = {fx: _geom_ids(env, getattr(env, fx).naming_prefix) for fx in fixtures}  # every geom of the fixture

    S, rec = {}, dict(stage=[], grip_width=[], obj={o: dict(pos=[], touch=[], grasp=[], eef_dist=[], far=[], tilt=[]) for o in objs},
                      at=[[{} for _ in range(len(states))] for _ in specs], dist=[[{} for _ in range(len(states))] for _ in specs],
                      fx_touch={f: [] for f in fixtures}, fx_state=[[] for _ in specs], fx_done=[[] for _ in specs],
                      fx_value=[[] for _ in specs],
                      progress=[[] for _ in specs])
    for f in range(len(states)):
        set_state(env, states[f])
        env._check_success()
        rec["stage"].append([bool(s["check"](env, S)) for s in steps])
        eef = np.array(env.sim.data.site_xpos[eef_site])
        rec["grip_width"].append(round(float(sum(abs(env.sim.data.qpos[q]) for q in finger_qadr)), 4))
        for o in objs:
            p = np.array(env.sim.data.body_xpos[env.obj_body_id[o]])
            r = rec["obj"][o]
            r["pos"].append(p.round(4).tolist()); r["touch"].append(bool(env.check_contact(grip, env.objects[o])))
            r["grasp"].append(bool(OU.check_obj_grasped(env, o))); r["eef_dist"].append(round(float(np.linalg.norm(eef - p)), 4))
            r["far"].append(bool(OU.gripper_obj_far(env, o)))
            zz = float(np.array(env.sim.data.body_xmat[env.obj_body_id[o]]).reshape(3, 3)[2, 2])  # object z axis . world z
            r["tilt"].append(round(float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0)))), 1))
        for fx in fixtures:
            rec["fx_touch"][fx].append(_touching(env, grip_geoms, fx_geoms[fx]))
        for k, sp in enumerate(specs):
            if sp["kind"] == "pnp":
                for o in sp["objs"]:
                    rec["at"][k][f][o] = bool(sp["at_target"](env, o))
                    rec["dist"][k][f][o] = round(float(sp["dist"](env, o)), 4)
            elif sp["kind"] == "fixture":
                rec["fx_state"][k].append(sp["state"](env)); rec["fx_done"][k].append(bool(sp["done"](env)))
                rec["fx_value"][k].append(round(float(sp["value"](env)), 4))
            elif sp["kind"] == "process":
                rec["progress"][k].append(sp["progress"](env))
    return rec


def _runs(mask):
    """[start, end) intervals where mask is True."""
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append([s, i]); s = None
    return out


def _holds(r, a, b):
    """grasp intervals of an object within [a, b): gripper touching it while it is grasped or lifted >= 2 cm above its
    height at a; gaps of <= 3 frames are merged."""
    z0 = r["pos"][a][2]
    held = [r["touch"][f] and (r["grasp"][f] or r["pos"][f][2] > z0 + 0.02) for f in range(a, b)]
    iv = [[s + a, e + a] for s, e in _runs(held)]
    merged = []
    for s, e in iv:
        if merged and s - merged[-1][1] <= 3:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def verdict_pnp(rec, k, sp, a, T):
    """Root cause of a pick-and-place stage that is not complete at the end, for each of its objects that is not at
    target at the end: never_reached / touched_not_grasped / still_holding[_at_target] / dropped (landed on floor or
    elsewhere) / placed_off_target (with distance to target) / at_target_release_not_met."""
    out = []
    for o in sp["objs"]:
        r = rec["obj"][o]
        at_end = rec["at"][k][T - 1][o]
        if at_end and sp["objs"] != [o] and len(sp["objs"]) > 1:
            continue  # done member of an any-order set
        holds = _holds(r, a, T)
        d = dict(obj=o, attempts=len(holds), dist_to_target_end=rec["dist"][k][T - 1][o], at_target_end=at_end,
                 min_eef_dist=round(min(r["eef_dist"][a:T]), 3), touched=any(r["touch"][a:T]))
        events = []
        for s, e in holds:
            rel = min(e, T - 1)
            settle = min(rel + 40, T - 1)
            z_rel, z_set = r["pos"][rel][2], r["pos"][settle][2]
            ev = dict(grasp=s, release=e if e < T else None, lift=round(max(p[2] for p in r["pos"][s:e]) - r["pos"][s][2], 3),
                      at_target_at_release=rec["at"][k][rel][o], dist_at_release=rec["dist"][k][rel][o],
                      fell=bool(z_rel - z_set > 0.08), on_floor_after=bool(r["pos"][settle][2] < FLOOR_Z),
                      at_target_after=rec["at"][k][settle][o], dist_after=rec["dist"][k][settle][o])
            events.append(ev)
        d["events"] = events
        if not holds:
            d["verdict"] = "never_reached" if d["min_eef_dist"] > 0.10 and not d["touched"] else "touched_not_grasped"
        elif holds[-1][1] >= T:
            d["verdict"] = "still_holding_at_target" if at_end else "still_holding"
        else:
            last = events[-1]
            if at_end:
                d["verdict"] = "at_target_release_not_met"
            elif last["fell"] or last["on_floor_after"]:
                d["verdict"] = "dropped"
                d["landed"] = "floor" if last["on_floor_after"] else "elsewhere"
            else:
                d["verdict"] = "placed_off_target"
        first_bad = next((ev for ev in events if not ev["at_target_after"]), None)
        if first_bad is not None:
            d["first_failed_attempt"] = ("dropped" if first_bad["fell"] or first_bad["on_floor_after"] else
                                         "still_holding" if first_bad["release"] is None else "placed_off_target")
        out.append(d)
    return out


def verdict_fixture(rec, k, sp, a, T):
    """fixture stage not complete at the end: fixture_done_other_condition_failed (the fixture state is reached but the
    stage check needs more, e.g. the burner that is on is not the one under the kettle) / fixture_done_then_undone /
    actuated_insufficient (the official state moved but never reached done; max value given) / touched_no_effect /
    fixture_not_touched."""
    touch = rec["fx_touch"][sp["fixture"]][a:T]
    done = rec["fx_done"][k][a:T]
    val = rec["fx_value"][k][a:T]
    d = dict(fixture=sp["fixture"], touched=any(touch), touch_frames=sum(touch), state_end=rec["fx_state"][k][T - 1],
             value_start=val[0], value_max=max(val), value_min=min(val), value_end=val[-1],
             done_frames=sum(done), done_at_end=bool(rec["fx_done"][k][T - 1]))
    moved = max(abs(v - val[0]) for v in val) > 0.02
    if d["done_at_end"]:
        d["verdict"] = "fixture_done_other_condition_failed"
    elif any(done):
        d["verdict"] = "fixture_done_then_undone"
    elif moved:
        d["verdict"] = "actuated_insufficient"
    elif any(touch):
        d["verdict"] = "touched_no_effect"
    else:
        d["verdict"] = "fixture_not_touched"
    return [d]


def verdict_other(rec, k, sp, a, T):
    d = dict(kind=sp["kind"])
    for o in sp.get("objs", []):
        r = rec["obj"][o]
        d[o] = dict(touched=any(r["touch"][a:T]), held_frames=sum(r["touch"][f] and r["grasp"][f] for f in range(a, T)),
                    min_eef_dist=round(min(r["eef_dist"][a:T]), 3))
    if sp["kind"] == "process":
        d["progress_end"] = rec["progress"][k][T - 1]
    return [d]


def analyze(rec, task):
    """Per-stage completion (order-free, first/last frame) and, for stages not complete at the end, the verdict.
    A stage's window starts when the previous stage (task order) first completed (0 if never)."""
    specs, T = STAGE_SPECS[task], len(rec["stage"])
    st = np.array(rec["stage"])
    stages, prev_first = [], 0
    for k, sp in enumerate(specs):
        col = st[:, k]
        first = int(np.argmax(col)) if col.any() else None
        info = dict(stage=k, skill=TASKS[task][k]["skill"], kind=sp["kind"], first_true=first,
                    true_at_end=bool(col[-1]), undone=bool(col.any() and not col[-1]), window_start=prev_first)
        if not col[-1]:
            fn = {"pnp": verdict_pnp, "fixture": verdict_fixture}.get(sp["kind"], verdict_other)
            info["details"] = fn(rec, k, sp, prev_first, T)
        stages.append(info)
        if first is not None:
            prev_first = max(prev_first, first)
    failed = next((s for s in stages if not s["true_at_end"]), None)
    return dict(stages=stages, first_incomplete_stage=None if failed is None else failed["stage"])
