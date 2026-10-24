import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser(os.environ["ROBOCASA_ROOT"] + "/runs/scripts"))
import robosuite, robocasa  # noqa
from robosuite.controllers import load_composite_controller_config
from demo_utils import load_demo
from oracle import reset_to, set_state
k = robosuite.make(env_name="GetToastedBread", robots="PandaOmron", controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"),
                   has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, use_object_obs=True,
                   obj_instance_split="target", layout_and_style_ids=list(zip(range(1, 11), range(1, 11))))

for ep in [int(x) for x in sys.argv[1:]]:
    d = load_demo("GetToastedBread", ep); reset_to(k, d["xml"], d["ep_meta"]); m = k.sim.model
    for f in (0, 50, 300):
        set_state(k, d["states"][f])
        bid = k.obj_body_id["obj"]; dd = k.sim.data
        con = set()
        for i in range(dd.ncon):
            c = dd.contact[i]; g1, g2 = c.geom1, c.geom2
            for a, b in ((g1, g2), (g2, g1)):
                if m.body_rootid[m.geom_bodyid[a]] == m.body_rootid[bid]:
                    con.add(m.geom_id2name(b) or f"geom{b}:{m.body_id2name(m.geom_bodyid[b])}")
        slots = [k.toaster.check_slot_contact(k, "obj", slot_pair=i) for i in range(len(k.toaster._slot_pairs))]
        print(json.dumps(dict(ep=ep, f=f, slot_contact=slots, contacts=sorted(con), obj_pos=dd.body_xpos[bid].round(3).tolist(),
                              floors=[n for n in m.geom_names if n and n.endswith("_floor") and "toaster" in n])), flush=True)
