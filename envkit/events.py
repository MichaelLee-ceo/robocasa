"""Oracle failure events (2026-11-05): after-the-fact detectors that read the simulator (privileged information), for
the recovery experiment (../20261105_recovery). Like stages.py they only read the env (tests/run_tests.py checks it).

Events(ep) keeps per-episode memory across chunks; call ev.update(ep) after every ep.step. It returns the events that
fired at that chunk end; ev.events holds them all. Its memory is plain data (names and numbers), so copy.deepcopy it
together with an envkit snapshot to branch.

Events, in the current stage (stages.track: the first stage not done), for the stages listed in TARGETS:
  missed_grasp   pick stage: the gripper was commanded closed near the target (an attempt); now the target is not
                 grasped, the fingers are closed on nothing and the hand is away from the target, at 2 chunk ends in a
                 row (the policy carries on empty-handed). A new attempt re-arms it.
  dropped        pick stage: the target was grasped earlier in the stage; now it is not grasped and the hand is away,
                 at 2 chunk ends in a row
  fixture_short  fixture stage: the target moved (peak above START) but is below the official threshold, and has fallen
                 back by DROP from its peak or stopped changing without gripper contact, at 2 chunk ends in a row
  wrong_element  fixture stage: a fixture joint other than the target moved by MOVED (normalized) since the stage began
Values: the drawer's opening is OpenDrawer's own measure (Drawer.get_door_state, official threshold 0.95); the toaster
lever is the toaster's own state (get_state, turns on at 0.90); other joints use Fixture.get_joint_state."""
import numpy as np
import robocasa.utils.object_utils as OU
import stages

NEAR, AWAY = 0.15, 0.20      # m, hand to target body: an attempt is a close within NEAR; away is beyond AWAY
EMPTY = 0.005                # |finger joint| below this on both fingers: closed on nothing (open is about 0.04)
START, DROP, STILL = 0.10, 0.10, 0.01
MOVED = 0.10


def _drawer_target(e):
    return [dict(attr="drawer", fixture=e.drawer.name, joint=f"{e.drawer.name}_slidejoint")]


def _toaster_target(e):
    t = e.toaster
    pairs = [sp for sp in t._slot_pairs if t.check_slot_contact(e, "obj", slot_pair=sp)] or list(t._slot_pairs)[:1]
    return [dict(attr="toaster", fixture=t.name, joint=t._joint_names[f"lever_{pairs[0]}"], pair=pairs[0])]


# task -> {stage index: ("pick", object name) | ("fixture", targets(e), official threshold)}
TARGETS = {
    "KettleBoiling": {0: ("pick", "obj")},
    "SetUpCuttingStation": {0: ("pick", "knife"), 1: ("pick", "meat")},
    "DeliverStraw": {0: ("fixture", _drawer_target, 0.95), 1: ("pick", "straw")},
    "GetToastedBread": {0: ("fixture", _toaster_target, 0.90), 1: ("pick", "obj")},
}


def _target_value(e, tg):
    fx = getattr(e, tg["attr"])
    if tg["attr"] == "drawer":
        return float(fx.get_door_state(env=e)["door"])
    return float(fx.get_state(e)[tg["pair"]]["lever"])


def _eef(e):
    return np.array(e.sim.data.site_xpos[e.robots[0].eef_site_id["right"]])


def _fingers(e):
    return [float(e.sim.data.qpos[e.sim.model.get_joint_qpos_addr(j)])
            for j in ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2")]


def _fixture_joints(e):
    """{(fixture name, joint name): normalized value} of every articulated fixture joint"""
    out = {}
    for fx in e.fixtures.values():
        for j in getattr(fx, "_joint_infos", None) or {}:
            out[(fx.name, j)] = float(fx.get_joint_state(e, [j])[j])
    return out


def _touching(e, attr):
    try:
        return bool(e.check_contact(e.robots[0].gripper["right"], getattr(e, attr)))
    except Exception:
        return False


class Events:
    def __init__(self, ep):
        self.task, self.events, self.stage, self.mem = ep.task, [], None, {}
        self._enter(ep, stages.track(ep)["stage"])

    def _cfg(self, stage):
        return TARGETS.get(self.task, {}).get(stage)

    def _enter(self, ep, stage):
        e, c = ep.kenv, self._cfg(stage)
        self.stage, self.mem = stage, {}
        if c and c[0] == "fixture":
            self.mem = dict(target=c[1](e), peak=0.0, prev=None, still=0, streak=0, base=_fixture_joints(e), flagged=[])
        elif c and c[0] == "pick":
            self.mem = dict(attempt=False, grasped=False, streak=0, dstreak=0)

    def _fire(self, ep, kind, **info):
        ev = dict(type=kind, stage=self.stage, t=ep.t, chunk=len(ep.chunks) - 1, **info)
        self.events.append(ev)
        return ev

    def update(self, ep):
        e, out = ep.kenv, []
        stage = stages.track(ep)["stage"]
        if stage != self.stage:
            self._enter(ep, stage)
        c = self._cfg(stage)
        if not c:
            return out
        m = self.mem
        if c[0] == "pick":
            o = c[1]
            cmd_close = bool(ep.chunks[-1]["actions"]["action.gripper_close"][-1, 0] > 0.5) if ep.chunks else False
            dist = float(np.linalg.norm(_eef(e) - np.array(e.sim.data.body_xpos[e.obj_body_id[o]])))
            grasped = bool(OU.check_obj_grasped(e, o))
            empty = all(abs(q) < EMPTY for q in _fingers(e))
            if grasped:
                m.update(grasped=True, attempt=False, streak=0)
            elif cmd_close and dist < NEAR:
                m["attempt"] = True
            m["streak"] = m["streak"] + 1 if (m["attempt"] and not grasped and empty and dist > AWAY) else 0
            if m["streak"] == 2:
                out.append(self._fire(ep, "missed_grasp", target=o, dist=round(dist, 3)))
                m.update(attempt=False, streak=0)
            m["dstreak"] = m["dstreak"] + 1 if (m["grasped"] and not grasped and dist > AWAY) else 0
            if m["dstreak"] == 2:
                out.append(self._fire(ep, "dropped", target=o, dist=round(dist, 3)))
                m.update(grasped=False, dstreak=0)
        else:
            th = c[2]
            v = min(_target_value(e, tg) for tg in m["target"])
            m["peak"] = max(m["peak"], v)
            touching = any(_touching(e, tg["attr"]) for tg in m["target"])
            m["still"] = m["still"] + 1 if (m["prev"] is not None and abs(v - m["prev"]) < STILL and not touching) else 0
            m["prev"] = v
            short = m["peak"] > START and v < th and (v <= m["peak"] - DROP or m["still"] >= 2)
            m["streak"] = m["streak"] + 1 if short else 0
            if m["streak"] == 2:
                out.append(self._fire(ep, "fixture_short", value=round(v, 3), peak=round(m["peak"], 3), threshold=th))
            targets = {(tg["fixture"], tg["joint"]) for tg in m["target"]}
            for key, val in _fixture_joints(e).items():
                if key in targets or list(key) in m["flagged"] or key not in m["base"]:
                    continue
                if abs(val - m["base"][key]) > MOVED:
                    m["flagged"].append(list(key))
                    out.append(self._fire(ep, "wrong_element", fixture=key[0], joint=key[1],
                                          change=round(val - m["base"][key], 3)))
        return out
