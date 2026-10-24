"""SetUpCuttingStation training-demo start geometry from stored initial states + ep_meta (read-only, CPU only, no env.step).
Same knife-axis / blade-tip heuristic and bearing convention as cut_start.py. usage: demo_cut_start.py <out.json> <lerobot_dir>"""
import gzip, json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to

out, LD = sys.argv[1], sys.argv[2]
env = robosuite.make("SetUpCuttingStation", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)


def subtree(m, b):
    out = set()
    for i in range(m.nbody):
        j = i
        while j not in (0, b): j = m.body_parentid[j]
        if j == b: out.add(i)
    return out


def verts(name):
    m, d = env.sim.model, env.sim.data; S = subtree(m, env.obj_body_id[name]); V = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in S and m.geom_type[g] == 7:
            mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            V.append(m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.concatenate(V)


wrap = lambda a: float((a + 180) % 360 - 180)
rows = []
eps = sorted(e for e in os.listdir(f"{LD}/extras") if e.startswith("episode_"))
for e in eps:
    d0 = f"{LD}/extras/{e}"
    try:
        meta = json.load(open(f"{d0}/ep_meta.json"))
        xml = gzip.open(f"{d0}/model.xml.gz", "rt").read()
        st = np.load(f"{d0}/states.npz", allow_pickle=True)["states"][0]
        reset_to(env, xml, meta, st); env.unset_ep_meta()
        m, d = env.sim.model, env.sim.data
        model = {c["name"]: (c["info"]["cat"], c["info"]["mjcf_path"].rstrip("/").split("/")[-2]) for c in meta["object_cfgs"]}
        bid = m.body_name2id("mobilebase0_base"); b0 = np.array(d.body_xpos[bid]); Rb = d.body_xmat[bid].reshape(3, 3)
        P = {o: np.array(d.body_xpos[env.obj_body_id[o]]) for o in ("knife", "meat", "receptacle")}
        xy = lambda a, b: round(100 * float(np.linalg.norm(a[:2] - b[:2])), 1)
        bear = lambda p: round(wrap(np.degrees(np.arctan2(*(Rb.T @ (p - b0))[[1, 0]]))), 0)
        V = verts("knife"); U = V[:, :2] - V[:, :2].mean(0)
        ax = np.linalg.svd(U, full_matrices=False)[2][0]; t = U @ ax; L = t.max() - t.min()
        lo, hi = t < t.min() + 0.25 * L, t > t.max() - 0.25 * L
        hz = lambda mk: float(V[mk, 2].max() - V[mk, 2].min())
        blade = ax if hz(hi) < hz(lo) else -ax
        bl = Rb.T @ np.array([blade[0], blade[1], 0.0])
        row = dict(ep=e, layout=meta.get("layout_id"), knife=model["knife"][1], meat_cat=model["meat"][0], meat=model["meat"][1],
                   knife_board=xy(P["knife"], P["receptacle"]), meat_board=xy(P["meat"], P["receptacle"]),
                   base_knife=xy(b0, P["knife"]), bearing_board_deg=bear(P["receptacle"]), bearing_meat_deg=bear(P["meat"]),
                   blade_yaw_deg=round(wrap(np.degrees(np.arctan2(bl[1], bl[0]))), 0))
    except Exception as ex:
        row = dict(ep=e, error=repr(ex)[:200])
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(out, "w"), indent=1)
