import os
from mujoco.egl import egl_ext as EGL
devs = EGL.eglQueryDevicesEXT()
print("n devices", len(devs))
for i, d in enumerate(devs):
    try:
        s = EGL.eglQueryDeviceStringEXT(d, EGL.EGL_EXTENSIONS)
    except Exception as e:
        s = repr(e)
    print(i, s)
