"""Put only the robot back to an atomic task's start pose, leaving objects and fixture states untouched.
Base: robocasa's own placement (EnvUtils.compute_robot_base_placement_pose for the atomic task's init_robot_base_ref,
then set_robot_base with the Kitchen default deviations and its collision check). Arm/gripper/torso: robot.reset(
deterministic=True) defaults (init_qpos). The base frame of the loaded model is fixed at the composite's anchor
orientation, so a different target orientation is reached with the mobile-base yaw joint."""
import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robosuite.utils.transform_utils as T


def robot_qpos_idx(kenv):
    m = kenv.sim.model; idx = []
    for j in range(m.njnt):
        n = m.joint_id2name(j)
        if n and (n.startswith("robot0_") or n.startswith("mobilebase0_") or n.startswith("gripper0_")):
            a = m.jnt_qposadr[j]; idx.append(a)
    return np.array(sorted(idx))


def home_robot(kenv, ref_fixture, rng=None, arm_only=False):
    obj_qpos_before = {n: kenv.sim.data.get_joint_qpos(o.joints[0]).copy() for n, o in kenv.objects.items()}
    r = kenv.robots[0]
    kenv.robot_geom_ids = None  # cached geom ids belong to the model before load_scene
    m = kenv.sim.model
    base_j = [m.get_joint_qpos_addr(n) for n in ("mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_yaw")]
    base_q = kenv.sim.data.qpos[base_j].copy()
    r.reset(deterministic=True)  # arm, gripper, torso to init_qpos; resets the controllers
    if arm_only:  # keep the base exactly where the hand-off left it
        kenv.sim.data.qpos[base_j] = base_q; kenv.sim.data.qvel[:] = 0; kenv.sim.forward()
        for c in (r.composite_controller,):
            if hasattr(c, "reset"):
                c.reset()
        moved = {n: float(np.abs(kenv.sim.data.get_joint_qpos(o.joints[0]) - obj_qpos_before[n]).max()) for n, o in kenv.objects.items()}
        return dict(placed=True, arm_only=True, collision=bool(EnvUtils.detect_robot_collision(kenv)), max_obj_moved=round(max(moved.values()), 6))
    pos, ori = EnvUtils.compute_robot_base_placement_pose(kenv, ref_fixture=ref_fixture)
    dyaw = float(ori[2] - kenv.init_robot_base_ori_anchor[2])
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
    m = kenv.sim.model
    yaw_adr = m.get_joint_qpos_addr("mobilebase0_joint_mobile_yaw")
    # same search as EnvUtils.set_robot_base, but with the yaw set first (that helper resets the yaw to ~0)
    with EnvUtils.no_collision(kenv.sim):
        kenv.sim.data.qpos[yaw_adr] = dyaw; kenv.sim.forward()
    start = kenv.sim.get_state(); ok = False; tries = 0
    dx, dy = kenv.robot_spawn_deviation_pos_x, kenv.robot_spawn_deviation_pos_y
    for expand in range(4):
        for _ in range(50):
            tries += 1
            rp = EnvUtils.generate_random_robot_pos(env=kenv, anchor_pos=pos, anchor_ori=ori, pos_dev_x=dx, pos_dev_y=dy)
            EnvUtils.set_robot_to_position(kenv, rp); kenv.sim.forward()
            if not EnvUtils.detect_robot_collision(kenv):
                ok = True; break
            kenv.sim.set_state(start)
        if ok:
            break
        dx += 0.1; dy += 0.05
    kenv.sim.data.qvel[:] = 0; kenv.sim.forward()
    for c in (r.composite_controller,):
        if hasattr(c, "reset"):
            c.reset()
    moved = {n: float(np.abs(kenv.sim.data.get_joint_qpos(o.joints[0]) - obj_qpos_before[n]).max()) for n, o in kenv.objects.items()}
    collide = bool(EnvUtils.detect_robot_collision(kenv))
    bid = kenv.sim.model.body_name2id("mobilebase0_base")
    p0 = kenv.sim.data.body_xpos[bid]; R = kenv.sim.data.body_xmat[bid].reshape(3, 3)
    return dict(placed=ok, tries=tries, base=[round(float(x), 3) for x in p0[:2]],
                ref_in_robot_frame=[round(float(x), 2) for x in R.T @ (np.array(ref_fixture.pos) - p0)], base_target=[round(float(x), 3) for x in pos], base_ori_target=round(float(ori[2]), 3), dyaw=round(dyaw, 3),
                collision=collide, max_obj_moved=round(max(moved.values()), 6))


def robot_qpos_addrs(kenv):
    """qpos addresses of every robot joint (mobile base, torso, arm, gripper): all are 1-dof hinge/slide joints."""
    m = kenv.sim.model; out = []
    for j in range(m.njnt):
        n = m.joint_id2name(j) or ""
        if n.startswith(("robot0_", "mobilebase0_", "gripper0_")):
            assert m.jnt_type[j] in (2, 3), n
            out.append(int(m.jnt_qposadr[j]))
    return np.array(sorted(out))


def robot_to_scene_start(kenv, init_state):
    """copy the robot's joint values from the scene's initial state into the current state; objects and fixtures
    keep their current values. Velocities zeroed, controllers reset."""
    obj_before = {n: kenv.sim.data.get_joint_qpos(o.joints[0]).copy() for n, o in kenv.objects.items()}
    nq = kenv.sim.model.nq
    adr = robot_qpos_addrs(kenv)
    init_qpos = np.asarray(init_state)[1:1 + nq]  # flattened MjSimState = [time, qpos, qvel, ...]
    kenv.sim.data.qpos[adr] = init_qpos[adr]; kenv.sim.data.qvel[:] = 0; kenv.sim.forward()
    kenv.robot_geom_ids = None
    c = kenv.robots[0].composite_controller
    if hasattr(c, "reset"):
        c.reset()
    moved = {n: float(np.abs(kenv.sim.data.get_joint_qpos(o.joints[0]) - obj_before[n]).max()) for n, o in kenv.objects.items()}
    return dict(collision=bool(EnvUtils.detect_robot_collision(kenv)), max_obj_moved=round(max(moved.values()), 6), n_robot_joints=len(adr))
