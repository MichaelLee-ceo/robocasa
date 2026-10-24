import json, os, sys
import numpy as np, robosuite, mujoco
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to, set_state
z = np.load(f"{ROOT}/runs/scenes_seen11_v2/StirVegetables_s0.npz", allow_pickle=True)
env = robosuite.make("StirVegetables", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
M, D = env.sim.model._model, env.sim.data._data; mujoco.mj_forward(M, D)
set_state(env, np.load(f"{ROOT}/runs/20261014_asus_seen11/StirVegetables/StirVegetables_s0_single_end.npz", allow_pickle=True)["state"]); M, D = env.sim.model._model, env.sim.data._data; mujoco.mj_forward(M, D)
print("same D", D is env.sim.data._data)
b = env.obj_body_id["pot"]; pb = D.xpos[b]; print("pot body", pb, "quat", D.xquat[b])
for g in range(M.ngeom):
    if M.geom_bodyid[g] != b: continue
    nm = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_GEOM, g)
    if M.geom_type[g] == 7:
        mid = M.geom_dataid[g]; v = M.mesh_vert[M.mesh_vertadr[mid]:M.mesh_vertadr[mid] + M.mesh_vertnum[mid]]
        w = D.geom_xpos[g] + v @ D.geom_xmat[g].reshape(3, 3).T
        r = np.linalg.norm(w[:, :2] - pb[:2], axis=1)
        print(nm, M.geom_contype[g], "z", np.round([w[:, 2].min(), w[:, 2].max()], 3), "r", np.round([r.min(), r.max()], 3), "scale?", np.round(M.mesh_scale[mid] if hasattr(M,'mesh_scale') else 0,3))
    else:
        print(nm, M.geom_type[g], np.round(D.geom_xpos[g], 3), np.round(M.geom_size[g], 3))
