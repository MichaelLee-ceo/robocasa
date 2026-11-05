import os, sys
sys.path.insert(0, "/home/michaellee/mclee/robocasa/third_party/robosuite")
from robosuite.renderers.context.egl_context import EGLGLContext
ctx = EGLGLContext(64, 64, device_id=-1)
ctx.make_current()
from OpenGL import GL
print("dev", os.environ.get("MUJOCO_EGL_DEVICE_ID"), GL.glGetString(GL.GL_VENDOR), GL.glGetString(GL.GL_RENDERER), GL.glGetString(GL.GL_VERSION))
