import robosuite
from robosuite.controllers import load_composite_controller_config
import robocasa  # noqa registers envs


def make(task):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=cc,
                         camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
                         camera_widths=128, camera_heights=128, has_renderer=False, has_offscreen_renderer=False,
                         ignore_done=True, use_object_obs=True, use_camera_obs=False, camera_depths=False, seed=None,
                         obj_instance_split="target", generative_textures=None, randomize_cameras=False,
                         layout_and_style_ids=list(zip(range(1, 11), range(1, 11))), layout_ids=None, style_ids=None,
                         translucent_robot=False)
    env.reset()
    return env
