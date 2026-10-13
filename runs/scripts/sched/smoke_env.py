import time, numpy as np, imageio
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
t=time.time()
env = RoboCasaGymEnv(env_name="CloseFridge", camera_widths=256, camera_heights=256, split="target", obj_registries=("lightwheel",))
obs, info = env.reset(seed=0)
print("lang:", env.env.get_ep_meta().get("lang"))
print("construct+reset s", round(time.time()-t,1))
print({k:getattr(v,'shape',v) for k,v in obs.items()})
t=time.time()
for i in range(20):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
print("20 steps s", round(time.time()-t,2), "reward", r, term, trunc)
imgs=[v for k,v in obs.items() if k.startswith("video.")]
print("image means", [round(float(x.mean()),1) for x in imgs])
imageio.imwrite("/tmp/claude-1001/rc/frame.png", np.concatenate(imgs,1))
