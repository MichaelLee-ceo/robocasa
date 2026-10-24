"""StoreLeftoversInBowl training demos: which fridge shelf the bowl ends on (last stored state), per layout.
Read-only, CPU only, no env.step. usage: demo_store_shelf.py <out.json> <lerobot_dir>"""
import gzip, json, os, re, sys
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to

out, LD = sys.argv[1], sys.argv[2]
env = robosuite.make("StoreLeftoversInBowl", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
rows = []
for e in sorted(x for x in os.listdir(f"{LD}/extras") if x.startswith("episode_")):
    d0 = f"{LD}/extras/{e}"
    try:
        meta = json.load(open(f"{d0}/ep_meta.json")); xml = gzip.open(f"{d0}/model.xml.gz", "rt").read()
        st = np.load(f"{d0}/states.npz", allow_pickle=True)["states"]
        reset_to(env, xml, meta, st[-1]); env.unset_ep_meta()
        m, d = env.sim.model, env.sim.data; fr = env.fridge
        Rf = T.euler2mat(np.array([0, 0, fr.rot])); fpos = np.array(fr.pos)
        bb = env.obj_body_id["bowl"]; obj = env.objects["bowl"]
        c = Rf.T @ (np.array(d.body_xpos[bb]) - fpos)
        pts = np.array([Rf.T @ (p - fpos) for p in obj.get_bbox_points(trans=np.array(d.body_xpos[bb]), rot=T.convert_quat(d.body_xquat[bb], to="xyzw"))])
        regs = sorted((n for n in fr.get_reset_region_names() if "fridge" in n and "drawer" not in n and n in fr._regions),
                      key=lambda n: fr._regions[n]["p0"][2])
        band = None
        for i, n in enumerate(regs):
            g = fr._regions[n]
            if g["p0"][2] <= c[2] <= g["p0"][2] + 0.9 * (g["pz"][2] - g["p0"][2]):
                band = (i, n, g)
        row = dict(ep=e, layout=meta.get("layout_id"), fridge=";".join(sorted(set(re.findall(r'fixtures/fridges/([^/"]+)/', xml)))),
                   n_shelves=len(regs), bowl_centre_z_cm=round(100 * float(c[2]), 1), bowl_abs_z_cm=round(100 * float(d.body_xpos[bb][2]), 1),
                   rack_contact=bool(fr.check_rack_contact(env, "bowl")))
        if band:
            i, n, g = band; ys = sorted([g["p0"][1], g["py"][1]])  # front = -y (door side) for all fridges in our run
            D = ys[1] - ys[0]
            row.update(shelf_index=i, shelf=n, front_margin_cm=round(100 * float(pts[:, 1].min() - ys[0]), 1), tol_cm=round(5 / D, 1))
        rows.append(row)
    except Exception as ex:
        rows.append(dict(ep=e, error=repr(ex)[:200]))
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(out, "w"), indent=1)
