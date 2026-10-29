"""Exploration: qpos layout of a scene, object bodies, robot joints, timing of set_state/forward.
usage: python -I explore.py <task> <scene npz>"""
import json, os, sys, time
import numpy as np

R = os.environ["ROBOCASA_ROOT"]
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480"]


def main():
    import atomic_eval  # noqa: F401
    from oracle import reset_to, set_state
    from analyze_stages import replay_env
    task, scene = sys.argv[1], sys.argv[2]
    t0 = time.time()
    env = replay_env(task)
    print("env built", time.time() - t0, flush=True)
    z = np.load(scene, allow_pickle=True)
    t0 = time.time()
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
    print("reset_to", time.time() - t0, flush=True)
    m, d = env.sim.model, env.sim.data
    print("nq", m.nq, "nv", m.nv, "na", m.na, "nbody", m.nbody, "ngeom", m.ngeom)
    print("flattened state len", len(env.sim.get_state().flatten()))
    for j in range(m.njnt):
        print(j, m.joint_id2name(j), "type", m.jnt_type[j], "qposadr", m.jnt_qposadr[j], "dofadr", m.jnt_dofadr[j],
              "body", m.body_id2name(m.jnt_bodyid[j]))
    print("obj_body_id", env.obj_body_id)
    rob = env.robots[0]
    print("eef site", rob.eef_site_id)
    print("gripper idx", rob._ref_gripper_joint_pos_indexes, rob.gripper["right"].joints)
    print("arm joint idx", rob._ref_joint_pos_indexes, rob._ref_joint_vel_indexes)
    print("root body", rob.robot_model.root_body)
    print("part controllers", list(rob.composite_controller.part_controllers.keys()))
    for n, c in rob.composite_controller.part_controllers.items():
        print("  ", n, type(c).__name__, getattr(c, "input_ref_frame", None))
    print("gripper geoms", [m.geom_id2name(g) for g in range(m.ngeom) if (m.body(m.geom_bodyid[g]).name.startswith("gripper0_"))][:40])
    st = env.sim.get_state().flatten()
    t0 = time.time()
    for _ in range(100):
        set_state(env, st)
    print("set_state x100", time.time() - t0)
    t0 = time.time()
    for _ in range(100):
        env.sim.set_state_from_flattened(st); env.sim.forward()
    print("raw set+forward x100", time.time() - t0)
    import mujoco
    t0 = time.time()
    for _ in range(100):
        env.sim.set_state_from_flattened(st); mujoco.mj_kinematics(m._model, d._data)
    print("raw set+kinematics x100", time.time() - t0)


if __name__ == "__main__":
    main()
