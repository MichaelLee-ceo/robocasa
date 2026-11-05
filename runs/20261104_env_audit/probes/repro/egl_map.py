"""Enumerate EGL devices (no context, no rendering) and map each to its CUDA device / DRM node, as robosuite indexes them."""
import ctypes, os
egl = ctypes.CDLL("libEGL.so.1")
egl.eglGetProcAddress.restype = ctypes.c_void_p
egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
QD = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))(egl.eglGetProcAddress(b"eglQueryDevicesEXT"))
QA = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_long))(egl.eglGetProcAddress(b"eglQueryDeviceAttribEXT"))
QS = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(egl.eglGetProcAddress(b"eglQueryDeviceStringEXT"))
n = ctypes.c_int(); devs = (ctypes.c_void_p * 16)()
QD(16, devs, ctypes.byref(n))
print("CUDA_VISIBLE_DEVICES=%r: %d EGL devices" % (os.environ.get("CUDA_VISIBLE_DEVICES"), n.value))
for i in range(n.value):
    v = ctypes.c_long(-1)
    ok = QA(devs[i], 0x323A, ctypes.byref(v))  # EGL_CUDA_DEVICE_NV
    ext = QS(devs[i], 0x3055) or b""  # EGL_EXTENSIONS
    drm = QS(devs[i], 0x3233) if b"EGL_EXT_device_drm" in ext else None  # EGL_DRM_DEVICE_FILE_EXT
    print(i, "cuda_dev" if ok else "", v.value if ok else "", drm, b"software" if b"MESA_device_software" in ext else b"")
