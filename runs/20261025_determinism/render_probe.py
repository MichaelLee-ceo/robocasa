"""Which pixels change when the same sim state is rendered repeatedly, and which geoms are they on?
Loads a frozen scene, renders each policy camera K times at the initial state, and reports the pixels whose value is
not the same in all K renders, mapped to geom names through a segmentation render.
usage: python -I render_probe.py <scene npz> [K=50] [offsamples override, e.g. 0]"""
import collections, json, os, sys
import numpy as np
import gymnasium as gym
R = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{R}/runs/scripts")
import atomic_eval as A  # noqa: E402

scene, K = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50
task = os.path.basename(scene).split("_s")[0]
base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
kenv = base.env
A.load_scene(kenv, scene)
if len(sys.argv) > 3:
    old = kenv.sim._render_context_offscreen
    geomgroup, flags = old.vopt.geomgroup.copy(), old.vopt.flags.copy()
    kenv.sim.model.vis.quality.offsamples = int(sys.argv[3])
    kenv.sim._render_context_offscreen = None  # rebuild the GL context with the new sample count
    from robosuite.utils.binding_utils import MjRenderContextOffscreen
    MjRenderContextOffscreen(kenv.sim, device_id=-1)
    new = kenv.sim._render_context_offscreen
    new.vopt.geomgroup[:], new.vopt.flags[:] = geomgroup, flags  # keep robosuite's visual-only geom groups
mm = kenv.sim.model
if os.environ.get("PROBE_OPAQUE_GLASS"):  # diagnostic: make every fixture glass material ("...Clear...") opaque
    import mujoco
    for mid in range(mm._model.nmat):
        if "Clear" in (mujoco.mj_id2name(mm._model, mujoco.mjtObj.mjOBJ_MATERIAL, mid) or ""):
            mm._model.mat_rgba[mid][3] = 1.0
import time
t0 = time.time()
out = {"task": task, "offsamples": int(mm.vis.quality.offsamples), "egl_device": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
       "opaque_glass": bool(os.environ.get("PROBE_OPAQUE_GLASS"))}
for cam in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
    h, w = kenv.camera_heights[kenv.camera_names.index(cam)], kenv.camera_widths[kenv.camera_names.index(cam)]
    imgs = np.stack([kenv.sim.render(camera_name=cam, width=w, height=h) for _ in range(K)])
    varying = (imgs != imgs[:1]).any(axis=(0, 3)) if imgs.ndim == 4 else (imgs != imgs[:1]).any(axis=0)
    n_distinct = len({im.tobytes() for im in imgs})
    seg = kenv.sim.render(camera_name=cam, width=w, height=h, segmentation=True)[..., 1]  # geom id per pixel
    geoms = collections.Counter()
    for y, x in zip(*np.nonzero(varying)):
        gid = int(seg[y, x])
        geoms[mm.geom_id2name(gid) if gid >= 0 else "background"] += 1
    # how often a varying pixel deviates from its most common value (per render)
    dev = []
    for y, x in zip(*np.nonzero(varying)):
        vals = [v.tobytes() for v in imgs[:, y, x]]
        dev.append(1 - collections.Counter(vals).most_common(1)[0][1] / K)
    out[cam] = dict(size=[h, w], distinct_images=n_distinct, varying_pixels=int(varying.sum()),
                    renders_differing_from_first=int(sum((im != imgs[0]).any() for im in imgs)),
                    max_pixel_deviation_rate=round(max(dev), 4) if dev else 0.0, geoms=dict(geoms.most_common(8)))
    if os.environ.get("PROBE_SAVE"):  # first render of each camera, to compare renderers
        np.save(f"{os.environ['PROBE_SAVE']}_{cam}.npy", imgs[0])
out["sec_per_render"] = round((time.time() - t0) / (3 * (K + 1)), 4)
print(json.dumps(out, default=str), flush=True)
