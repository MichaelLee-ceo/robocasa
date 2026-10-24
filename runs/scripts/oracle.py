"""Oracle machinery: reset the env to a demo state, and decide when a demo subtask is "done".

Completion signal (documented in README):
  * the manipulated object of a pick/place pair = the non-robot object with the largest displacement
    over the demo's pick+place segments;
  * pick  : object is grasped (robocasa OU.check_obj_grasped) and moved > 2 cm from where it was when the
            subtask became active;
  * place : object not grasped, and EITHER it touches a body it touched at the demo's end-of-subtask frame
            that it did not touch at subtask start (new support, e.g. cutting board / cabinet shelf),
            OR (same support, e.g. counter->counter) it is within PLACE_TOL of the demo end position;
  * fixture / other skills: every non-robot hinge/slide joint that moved > 0.05 in the demo segment has
            covered >= 50% of its demo motion;
  * navigate: robot base within 0.3 m (xy) of the demo end base position;
  * the last real subtask (and "done") additionally / alternatively counts as done when env._check_success().
"""
import json
import numpy as np
import robosuite
import robocasa.utils.object_utils as OU

ROBOT_PREFIX = ("robot0_", "mobilebase0_", "gripper0_")
PLACE_TOL = 0.15


def reset_to(kenv, xml, ep_meta, state=None):
    kenv.set_ep_meta(ep_meta)
    kenv.reset()
    kenv.reset_from_xml_string(kenv.edit_model_xml(xml))
    kenv.sim.reset()
    restore_task_attrs(kenv, ep_meta)
    if state is not None:
        set_state(kenv, state)
    else:
        kenv.sim.forward()
    sync_fixture_poses(kenv)


def sync_fixture_poses(kenv, tol_pos=1e-3, tol_yaw=1e-2):
    """reset_from_xml_string is a soft reset: the python fixtures keep the placements the preceding env.reset() sampled
    (robocasa issue #209 / PR #225); on our frozen scenes this hits the counter appliances and accessories (coffee
    machine, toaster, toaster oven, knife block, ...), 0.2-1.2 m off. Anything computed from fixture.pos / fixture.rot
    (e.g. OU.obj_inside_of via get_int_sites) would then use the wrong place. Write the simulator's pose back into every
    fixture that disagrees, directly on its MJCF element as PR #225 does (some set_pos overrides re-place interior
    objects). The simulator itself is not touched. Returns the names of the synced fixtures."""
    from robocasa.models.fixtures import Fixture
    from robosuite.utils.mjcf_utils import array_to_string
    synced = []
    for name, fx in kenv.fixtures.items():
        if not isinstance(fx, Fixture):
            continue
        try:
            bid = kenv.sim.model.body_name2id(fx.root_body)
        except Exception:
            continue
        pos = np.array(kenv.sim.data.body_xpos[bid])
        mat = np.array(kenv.sim.data.body_xmat[bid]).reshape(3, 3)
        yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
        dyaw = abs((yaw - float(fx.rot) + np.pi) % (2 * np.pi) - np.pi)
        if np.linalg.norm(pos - np.asarray(fx.pos, dtype=float)) > tol_pos or dyaw > tol_yaw:
            fx._obj.set("pos", array_to_string(pos))
            fx._obj.set("euler", array_to_string(np.array([0.0, 0.0, yaw])))
            synced.append(name)
    return synced


def restore_task_attrs(kenv, ep_meta):
    """Task attributes that are normally set while sampling object configs (skipped when the scene is
    rebuilt from ep_meta). Only RecycleBottlesByType among our 8 tasks has one (self.choice)."""
    if type(kenv).__name__ == "RecycleBottlesByType":
        for c in ep_meta["object_cfgs"]:
            if c["name"] == "mystery_middle":
                g = c.get("obj_groups")
                kenv.choice = g if isinstance(g, str) else c["info"]["cat"]


def set_state(kenv, state):
    kenv.sim.set_state_from_flattened(state)
    kenv.sim.forward()
    if hasattr(kenv, "update_sites"):
        kenv.update_sites()
    if hasattr(kenv, "update_state"):
        kenv.update_state()


def _is_robot(name):
    return name is not None and name.startswith(ROBOT_PREFIX)


class SceneProbe:
    def __init__(self, kenv):
        self.e = kenv
        m = kenv.sim.model
        self.obj_names = list(kenv.objects.keys())
        self.obj_bid = {n: m.body_name2id(kenv.objects[n].root_body) for n in self.obj_names}
        self.root = m.body_rootid
        self.fx_joints = []
        for j in range(m.njnt):
            name = m.joint_id2name(j)
            if m.jnt_type[j] in (2, 3) and not _is_robot(name):  # slide / hinge
                self.fx_joints.append((name, m.jnt_qposadr[j]))
        self.base_bid = m.body_name2id("mobilebase0_base") if "mobilebase0_base" in m.body_names else None

    def obj_pos(self):
        d = self.e.sim.data
        return {n: d.body_xpos[b].copy() for n, b in self.obj_bid.items()}

    def joints(self):
        q = self.e.sim.data.qpos
        return np.array([q[a] for _, a in self.fx_joints])

    def base_xy(self):
        return None if self.base_bid is None else self.e.sim.data.body_xpos[self.base_bid][:2].copy()

    def contacts(self, obj):
        m, d = self.e.sim.model, self.e.sim.data
        b = self.obj_bid[obj]
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for x, y in ((b1, b2), (b2, b1)):
                if self.root[x] == b or x == b:
                    n = m.body_id2name(y)
                    if self.root[y] != b and not _is_robot(n) and not _is_robot(m.body_id2name(self.root[y])):
                        out.add(int(y))
        return out

    def grasped(self, obj):
        """gripper fingers in contact with the object's body tree and gripper not fully open.
        (robocasa's OU.check_obj_grasped misses some objects whose contact geoms are not registered)."""
        m, d = self.e.sim.model, self.e.sim.data
        if not hasattr(self, "_grip_bodies"):
            self._grip_bodies = {i for i in range(m.nbody) if (m.body_id2name(i) or "").startswith("gripper0_")}
            q = [m.get_joint_qpos_addr(j) for j in ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2")]
            self._gq = q
        b = self.obj_bid[obj]
        if abs(d.qpos[self._gq[0]]) > 0.039 and abs(d.qpos[self._gq[1]]) > 0.039:
            return False  # fully open
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in self._grip_bodies and self.root[b2] == b) or (b2 in self._grip_bodies and self.root[b1] == b):
                return True
        return False

    def snapshot(self):
        return dict(obj=self.obj_pos(), jq=self.joints(), base=self.base_xy())


def build_subtask_specs(kenv, demo):
    """Replay demo states at segment boundaries to derive a completion predicate per segment.
    Leaves the sim at an arbitrary demo state; caller must set the desired start state afterwards."""
    P = SceneProbe(kenv)
    segs = [s for s in demo["segs"] if s["stage"] != "done"]
    st = demo["states"]
    snaps, cons = {}, {}
    for s in segs:
        for f in (s["start"], min(s["end"], len(st) - 1)):
            if f not in snaps:
                set_state(kenv, st[f]); snaps[f] = P.snapshot()
                cons[f] = {o: P.contacts(o) for o in P.obj_names}
    specs = []
    for i, s in enumerate(segs):
        a, b = s["start"], min(s["end"], len(st) - 1)
        spec = dict(i=i, instr=s["instr"], skill=s["skill"], stage=s["stage"], demo_start=a, demo_end=b,
                    demo_len=b - a, last=(i == len(segs) - 1))
        if s["stage"] in ("pick", "place"):
            # object identity from the pick+place pair this segment belongs to
            j0 = i if s["stage"] == "pick" else max(i - 1, 0)
            while j0 > 0 and segs[j0]["stage"] != "pick":
                j0 -= 1
            j1 = j0 + 1
            while j1 < len(segs) and segs[j1]["stage"] not in ("place",):
                j1 += 1
            j1 = min(j1, len(segs) - 1)
            pa, pb = segs[j0]["start"], min(segs[j1]["end"], len(st) - 1)
            if pa not in snaps:
                set_state(kenv, st[pa]); snaps[pa] = P.snapshot(); cons[pa] = {o: P.contacts(o) for o in P.obj_names}
            if pb not in snaps:
                set_state(kenv, st[pb]); snaps[pb] = P.snapshot(); cons[pb] = {o: P.contacts(o) for o in P.obj_names}
            disp = {o: np.linalg.norm(snaps[pb]["obj"][o] - snaps[pa]["obj"][o]) for o in P.obj_names}
            # prefer the object actually held at the end of the demo's pick segment (e.g. a container,
            # not the food riding inside it); fall back to the largest displacement
            pe = min(segs[j0]["end"], len(st) - 1)
            held = []
            for f in (pe, max(pe - 5, 0), min(pe + 5, len(st) - 1)):
                set_state(kenv, st[f])
                held += [o for o in P.obj_names if P.grasped(o)]
            held = [o for o in set(held) if disp[o] > 0.02]
            obj = max(held, key=disp.get) if held else max(disp, key=disp.get)
            # container vs content: if the instruction names a container and a container-like object moved together with
            # the chosen one (>= 50% of its displacement), the target is the container (e.g. veg_tupperware, not vegetable1)
            instr = (segs[j0]["instr"] + " " + s["instr"]).lower()
            if "container" in instr and not any(x in obj for x in ("tupperware", "container")):
                cands = [o for o in P.obj_names if any(x in o for x in ("tupperware", "container"))
                         and disp[o] >= 0.5 * disp[obj]]
                if cands:
                    obj = max(cands, key=disp.get)
            spec.update(obj=obj, obj_disp=float(disp[obj]))
            if s["stage"] == "place":
                new_support = cons[b][obj] - cons[pa][obj]
                spec.update(end_pos=snaps[b]["obj"][obj].tolist(), new_support=sorted(new_support),
                            new_support_names=[kenv.sim.model.body_id2name(x) for x in new_support])
        elif s["stage"] == "navigate":
            spec.update(base_end=None if snaps[b]["base"] is None else snaps[b]["base"].tolist())
        else:
            dq = snaps[b]["jq"] - snaps[a]["jq"]
            mv = np.nonzero(np.abs(dq) > 0.05)[0]
            spec.update(joints=[P.fx_joints[k][0] for k in mv], jq_start=snaps[a]["jq"][mv].tolist(),
                        jq_end=snaps[b]["jq"][mv].tolist(), jidx=mv.tolist())
        specs.append(spec)
    return specs, P


class SubtaskTracker:
    def __init__(self, kenv, specs, probe):
        self.e, self.specs, self.P = kenv, specs, probe
        self.k = 0
        self.anchor = None
        self.history = []  # (step, k) when subtask k completed
        self._arm()

    def _arm(self):
        self.anchor = self.P.snapshot()

    def done_k(self, k):
        sp = self.specs[k]
        st = sp["stage"]
        if sp["last"] and self.e._check_success():
            return True
        if st == "pick":
            o = sp["obj"]
            return self.P.grasped(o) and np.linalg.norm(self.P.obj_pos()[o] - self.anchor["obj"][o]) > 0.02
        if st == "place":
            o = sp["obj"]
            if self.P.grasped(o):
                return False
            if sp["new_support"]:
                if self.P.contacts(o) & set(sp["new_support"]):
                    return True
            return np.linalg.norm(self.P.obj_pos()[o] - np.array(sp["end_pos"])) < PLACE_TOL
        if st == "navigate":
            if sp.get("base_end") is None:
                return True
            return np.linalg.norm(self.P.base_xy() - np.array(sp["base_end"])) < 0.3
        # fixture / other
        if "faucet" in sp["instr"] and ("turn on" in sp["instr"] or "turn off" in sp["instr"]) and hasattr(self.e, "sink"):
            # use robocasa's own semantics (handle joint in (0.4, pi)) instead of matching every joint the demo moved
            on = bool(self.e.sink.get_handle_state(self.e).get("water_on", False))
            return on if "turn on" in sp["instr"] else not on
        if not sp.get("jidx"):
            return False  # no observable joint change in demo -> only env success can certify
        q = self.P.joints()[sp["jidx"]]
        q0, q1 = np.array(sp["jq_start"]), np.array(sp["jq_end"])
        return bool(np.all((q - q0) / (q1 - q0) >= 0.5))

    def update(self, step):
        """advance through all subtasks that are complete now; returns current index (len = all done)."""
        while self.k < len(self.specs) and self.done_k(self.k):
            self.history.append((step, self.k))
            self.k += 1
            self._arm()
        return self.k

    @property
    def instr(self):
        return self.specs[min(self.k, len(self.specs) - 1)]["instr"]
