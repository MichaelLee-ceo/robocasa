import json, os, sys
import numpy as np, robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to, set_state
z = np.load(f"{ROOT}/runs/scenes_seen11_v2/StirVegetables_s8.npz", allow_pickle=True)
env = robosuite.make("StirVegetables", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
em = json.loads(str(z["ep_meta"]))
print(em.keys()); print(json.dumps(em.get("object_cfgs"), default=str)[:3000])
reset_to(env, str(z["xml"]), em, z["state"]); env.unset_ep_meta()
m = env.sim.model; M = m._model
for o in ("spatula", "pot", "veg1", "veg2"):
    b = env.obj_body_id[o]; print(o, m.body_id2name(b), "rootname", env.objects[o].root_body)
    for g in range(M.ngeom):
        bb = M.geom_bodyid[g]; j = bb
        while j not in (0, b): j = M.body_parentid[j]
        if j == b: print("  ", g, m.geom_id2name(g), "type", M.geom_type[g], "size", np.round(M.geom_size[g], 4), "contype", M.geom_contype[g], M.geom_conaffinity[g], "group", M.geom_group[g], "aabb", np.round(M.geom_aabb[g], 4) if hasattr(M, "geom_aabb") else None, "body", m.body_id2name(bb))
gr = env.robots[0].gripper["right"]; print(gr.important_geoms)
print([m.geom_id2name(g) for g in range(M.ngeom) if m.geom_id2name(g) and "finger" in m.geom_id2name(g)])
print(gr.joints, [m.joint_name2id(j) for j in gr.joints])
print("knob", env.knob, env.stove.name, env.counter.name)
print(getattr(env.objects["spatula"], "get_bbox_points", None), env.objects["spatula"].size if hasattr(env.objects["spatula"], "size") else None)
print([n for n in dir(env.objects["pot"]) if not n.startswith("__")])
