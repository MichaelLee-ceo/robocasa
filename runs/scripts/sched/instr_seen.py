import robosuite, robocasa, json, sys
from robosuite.controllers import load_composite_controller_config
from robocasa.utils.dataset_registry import TARGET_TASKS
out={}
for t in TARGET_TASKS["composite_seen"]:
    k = robosuite.make(env_name=t, robots="PandaOmron", controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"),
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, use_object_obs=True,
        obj_instance_split="target", layout_and_style_ids=list(zip(range(1, 11), range(1, 11))), seed=0)
    L=[]
    for s in range(3):
        k.reset(); L.append(k.get_ep_meta().get("lang"))
    out[t]=L; k.close()
    print(t, "|", " || ".join(L), flush=True)
json.dump(out, open("/home/michaellee/Desktop/robocasa/runs/composite_seen_instructions.json","w"), indent=1)
