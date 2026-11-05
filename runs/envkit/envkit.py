"""The environment layer for every experiment (2026-11-04): official episodes, exact snapshots, exact replays.

Episode(task, seed)
  A fresh robocasa gym env per episode, reset with the official reset (env.reset(seed=seed)) and wrapped as GR00T's
  eval client wraps it (gr00t/eval/simulation.py): MultiStepWrapper with video / state delta indices [0], 16 action
  steps and max_episode_steps = get_task_horizon(task) (RoboCasa v1.0.1). The env starts in the frozen post-reset
  state of scene (task, seed): one official reset, frozen whole by freeze() (RoboCasa's reset itself is not
  reproducible). A fresh env per episode makes it independent of whatever ran before in the process.
step(chunk)
  One policy chunk through MultiStepWrapper.step, exactly as the official client. The chunk's success is
  info["success"][0], i.e. the env's _check_success() after the last executed step of the chunk. The episode
  succeeds if any chunk end does (run_simulation), and ends at the horizon (the last chunk may be partial).
  Every executed action, every step's success, and a hash of every observation are recorded.
snapshot() / restore(snap)
  The whole MjData (a full copy; restored with mj_setState of the integration state, i.e. state, warm start,
  controls, applied forces, then every derived array and the contacts, so the next step and any check at the node
  see exactly what a continuous run sees) plus the whole Python object graph reachable from the wrapper stack: task
  attributes, fixtures, objects, robots and controllers, observables, wrapper buffers, RNG states. It is written back
  in place, so object identity and aliasing survive. A restore does not call set_state, forward or update_state.
replay(task, seed, chunks)
  Re-executes recorded chunks in a fresh episode through the same step; this reproduces the run bit for bit.
Stage checks are not part of this module; they must only read the env (tests/ checks that they do).
usage: import envkit; ep = envkit.Episode("KettleBoiling", 3); ep.step(chunk); s = ep.snapshot(); ep.restore(s)"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # EGL device index = physical GPU (robosuite egl_context)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import collections, copy, gc, hashlib, json, platform, random, socket, subprocess, sys, types
import xml.etree.ElementTree as ET
import numpy as np

ROOT = os.path.abspath(os.environ.get("ROBOCASA_ROOT")  # the repository
                       or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{ROOT}/third_party/Isaac-GR00T")
import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import robocasa  # noqa: E402,F401  (registers robocasa/<Task>)
from robocasa.utils.dataset_registry_utils import get_task_horizon  # noqa: E402
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper  # noqa: E402
from robosuite.utils import binding_utils as _rs_binding  # noqa: E402

# Render contexts (found 2026-11-05). Every hard reset builds a new MjSim and render context. The old sim and its
# context form a reference cycle, so only Python's cyclic garbage collector frees them, at an arbitrary later time.
# robosuite's MjRenderContext.__del__ then frees MuJoCo's GL objects (framebuffers, textures, display lists) in
# whichever GL context is current, usually the live one. The live context's framebuffers are deleted, and every later
# camera image is stale (another camera's last image) or garbage while the physics is untouched. Measured: one
# gc.collect() after a step corrupts all later images of KettleBoiling. A context now frees its GL objects in its own
# GL context. Episode also collects the old contexts right after its reset. Rendering itself is unchanged.
def _free_render_context(self):
    gl, con = vars(self).get("gl_ctx"), vars(self).get("con")
    try:
        if gl is not None and getattr(gl, "_context", None):
            gl.make_current()  # (if gl was freed first, EGL released the thread: no context is current)
        if con is not None:
            con.free()
    except Exception:
        pass
    finally:
        if gl is not None:
            try:
                gl.free()  # unbinds it; the live context's next render() makes itself current again
            except Exception:
                pass


_rs_binding.MjRenderContext.__del__ = _free_render_context


# Counter regions (found 2026-11-05). Counter.get_reset_regions de-duplicates the candidate top geoms with
# list(set(valid_geoms)). The geoms are xml Elements, hashed by memory address, so their order (and the region names
# geom_0, geom_1 that rng.choice picks from) changes from process to process. With the same generator state, an object
# then lands left or right of the sink. A failed placement makes _load_model start over, which shifts every later draw
# (e.g. PreSoakPan s1: two different models). The same line, de-duplicated in order: the region is still drawn
# uniformly by the fixture's rng, so the scene distribution is unchanged, and the reset becomes reproducible.
def _patch_counter_regions():
    import inspect, textwrap
    from robocasa.models.fixtures import counter
    old, new = "valid_geoms = list(set(valid_geoms))", "valid_geoms = list(dict.fromkeys(valid_geoms))"
    src = textwrap.dedent(inspect.getsource(counter.Counter.get_reset_regions))
    assert src.count(old) == 1, "robocasa changed: re-check Counter.get_reset_regions"
    ns = {}
    exec(compile(src.replace(old, new), counter.__file__, "exec"), counter.__dict__, ns)
    counter.Counter.get_reset_regions = ns["get_reset_regions"]


_patch_counter_regions()

# RoboCasa draws from unseeded generators (np.random.default_rng() with no seed, in fixtures, placement and object
# sampling) and from the global np.random (counter colours). Within an episode these come from a stream derived from
# (task, seed) instead, so the model and the env's structure are the same every time; values that still differ
# (placements) are overwritten by the frozen scene.
_default_rng = np.random.default_rng
_STREAM = {"rng": None}


def _seeded_default_rng(seed=None, *a, **k):
    if seed is None and _STREAM["rng"] is not None:
        seed = int(_STREAM["rng"].integers(2 ** 63))
    return _default_rng(seed, *a, **k)


np.random.default_rng = _seeded_default_rng


def _seed_process(task, seed):
    import zlib
    h = [zlib.crc32(task.encode()), int(seed)]
    _STREAM["rng"] = _default_rng(h)
    np.random.seed(h); random.seed(f"{task}:{seed}")


TASKS = ("KettleBoiling", "PreSoakPan", "StackBowlsCabinet", "SteamInMicrowave", "PrepareCoffee", "ScrubCuttingBoard",
         "WashLettuce", "RinseSinkBasin", "LoadDishwasher", "StoreLeftoversInBowl", "SetUpCuttingStation",
         "StirVegetables", "SearingMeat", "PackIdenticalLunches", "DeliverStraw", "GetToastedBread")  # Composite-Seen
N_ACTION_STEPS = 16


# ---------------------------------------------------------------- the episode

class _StepLog(gym.Wrapper):
    """Between the robocasa gym env and MultiStepWrapper; observes each env step, changes nothing."""

    def __init__(self, env):
        super().__init__(env)
        self.steps = []

    def step(self, action):
        out = self.env.step(action)
        self.steps.append(bool(out[4]["success"]))
        return out


SCENES = os.path.join(ROOT, "runs", "envkit", "scenes")  # frozen post-reset states, <task>_s<seed>.pkl.gz (freeze())


def scene_path(task, seed):
    return os.path.join(SCENES, f"{task}_s{int(seed)}.pkl.gz")


class Episode:
    """Episode(task, seed): a fresh env put into the frozen post-reset state of scene (task, seed) (see freeze()).
    Episode.official(task, seed): a fresh env with the official reset itself (only used to freeze scenes: RoboCasa's
    reset is not reproducible, see ../20261104_env_audit/README.md)."""

    def __init__(self, task, seed, horizon=None, scene=None, _official=False):
        self.task, self.seed = task, int(seed)
        self.horizon = int(horizon or get_task_horizon(task))
        _seed_process(task, seed)
        base = gym.make(f"robocasa/{task}", split="target", enable_render=True)
        self._log = _StepLog(base)
        self.env = MultiStepWrapper(self._log, video_delta_indices=np.array([0]), state_delta_indices=np.array([0]),
                                    n_action_steps=N_ACTION_STEPS, max_episode_steps=self.horizon)
        self.kenv = base.unwrapped.env  # the robosuite / robocasa kitchen env
        self.obs, _ = self.env.reset(seed=self.seed)  # the official reset (same code path, so the same structure)
        gc.collect()  # free the render contexts the reset replaced now, not at an arbitrary step (see _free_render_context)
        if not _official:
            self.obs = _load_scene(self, scene or scene_path(task, seed))
        self.t, self.success, self.success_step, self.done = 0, False, None, False
        self.chunks = []  # the executed path: one record per chunk

    @classmethod
    def official(cls, task, seed, horizon=None):
        return cls(task, seed, horizon, _official=True)

    @property
    def instruction(self):
        return self.kenv.get_ep_meta()["lang"]

    def step(self, chunk):
        """chunk: {action key: (16, d)} as the policy returns it (no batch dim). Returns (obs, chunk success, done)."""
        assert not self.done, "episode is over"
        m = min(len(v) for v in chunk.values())
        if m < N_ACTION_STEPS:
            # a recorded last chunk is cut at the horizon; the official MultiStepWrapper indexes row `step` before it
            # checks `done`, so pad it with copies of its last row (never executed: the episode ends first)
            assert m >= self.horizon - self.t, f"a chunk of {m} actions must end the episode ({self.horizon - self.t} steps left)"
            chunk = {k: np.concatenate([v, np.repeat(np.asarray(v)[-1:], N_ACTION_STEPS - len(v), 0)]) for k, v in chunk.items()}
        self._log.steps = []
        obs, _, done, _, info = self.env.step(chunk)
        n = len(self._log.steps)
        ok = bool(info["success"][0])
        self.chunks.append(dict(t=self.t, n=n, actions={k: np.array(v[:n]) for k, v in chunk.items()},
                                step_success=list(self._log.steps), success=ok, obs=obs_digest(obs)))
        self.t += n
        if ok and not self.success:
            self.success, self.success_step = True, self.t
        self.done, self.obs = bool(done), obs
        return obs, ok, self.done

    def snapshot(self):
        return Snapshot(self)

    def restore(self, snap):
        snap.restore(self)

    def close(self):
        self.env.close()
        gc.collect()

    def record(self):
        """the episode as a plain record (json-able except the action arrays)"""
        return dict(task=self.task, seed=self.seed, horizon=self.horizon, steps=self.t, success=self.success,
                    success_step=self.success_step, chunks=self.chunks, machine=machine())


def replay(task, seed, chunks, horizon=None, scene=None):
    """re-execute recorded chunks in a fresh episode (from the frozen scene); returns the episode"""
    ep = Episode(task, seed, horizon, scene=scene)
    for c in chunks:
        ep.step(c["actions"])
    return ep


# ---------------------------------------------------------------- snapshots

def _mjdata(kenv):
    return kenv.sim.data._data


def _mjmodel(kenv):
    return kenv.sim.model._model


_INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION  # time, qpos, qvel, act, plugin, ctrl, applied forces, mocap, userdata, warm start
_SIZES = ("ne", "nf", "nl", "nefc", "nJ", "nA", "nisland", "nidof", "ncon")  # sizes of the per-step (arena) arrays


class _MjDump:
    """every array, size and contact field of an MjData, as plain numpy (picklable); _mj_copy_into reads it like an
    MjData"""

    def __init__(self, model, d):
        buf = np.empty(mujoco.mj_stateSize(model, _INTEGRATION)); mujoco.mj_getState(model, d, buf, _INTEGRATION)
        self._state = buf
        self._fields = {}
        for name in dir(d):
            if name.startswith("_") or name == "contact":
                continue
            try:
                v = getattr(d, name)
            except Exception:
                continue
            if isinstance(v, np.ndarray):
                self._fields[name] = v.copy()
            elif isinstance(v, (int, float)) and name in _SIZES:
                self._fields[name] = v
        self._contact = {f: getattr(d.contact, f).copy() for f in dir(d.contact)
                         if not f.startswith("_") and isinstance(getattr(d.contact, f), np.ndarray)}

    def __dir__(self):
        return list(self._fields)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "contact":
            return types.SimpleNamespace(**self._contact)
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name)


def _mj_copy_into(model, dst, src):
    """make dst equal src in place: the integration state through mj_setState (what the future depends on), then the
    sizes and every array of equal shape (derived kinematics, contacts, sensors: what a check reads at this point)"""
    if isinstance(src, _MjDump):
        buf = src._state
    else:
        buf = np.empty(mujoco.mj_stateSize(model, _INTEGRATION)); mujoco.mj_getState(model, src, buf, _INTEGRATION)
    mujoco.mj_setState(model, dst, buf, _INTEGRATION)
    for n in _SIZES:
        if hasattr(src, n):
            try:
                setattr(dst, n, getattr(src, n))
            except (AttributeError, TypeError):
                pass
    for name in dir(src):
        if name.startswith("_") or name == "contact" or name.startswith(_SCRATCH):
            continue
        try:
            a, b = getattr(dst, name), getattr(src, name)
        except Exception:
            continue
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and a.flags.writeable:
            np.copyto(a, b)
    for f in dir(src.contact):
        if not f.startswith("_"):
            a, b = getattr(dst.contact, f), getattr(src.contact, f)
            if isinstance(a, np.ndarray) and a.shape == b.shape:
                np.copyto(a, b)


_OPAQUE_TYPES = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType,
                 types.BuiltinMethodType, types.CodeType, type, ET.Element, ET.ElementTree)
_OPAQUE_MODULES = ("mujoco", "robosuite.utils.binding_utils", "robosuite.renderers", "OpenGL", "glfw", "zmq", "torch")
_IMMUTABLE = (type(None), bool, int, float, complex, str, bytes, frozenset, np.generic, range, slice, type(Ellipsis))


def _opaque(x):
    if isinstance(x, _OPAQUE_TYPES):
        return True
    m = type(x).__module__ or ""
    return any(m == p or m.startswith(p + ".") for p in _OPAQUE_MODULES)


class _Graph:
    """The state of a Python object graph, recorded so it can be written back in place.
    Nodes: ("v", value) immutable; ("r", ref) kept as is (modules, functions, mujoco objects, xml);
    ("a", ref, copy) ndarray; ("l"/"q"/"s", ref, children) list / deque / set; ("t", children) tuple;
    ("d", ref, [(key, child)]) dict; ("o", ref) object, its attributes in self.attrs[id]; ("g", ref, state) numpy
    Generator; ("rs", ref, state) numpy RandomState. Shared objects map to one node (aliasing is kept)."""

    def __init__(self, root):
        self.memo, self.attrs, self.paths = {}, {}, {}
        self.root = self._walk(root, "env")

    def _walk(self, x, path):
        if isinstance(x, _IMMUTABLE):
            return ("v", x)
        key = id(x)
        if key in self.memo:
            return self.memo[key]
        if _opaque(x):
            node = ("r", x)
        elif isinstance(x, np.ndarray):
            node = ("a", x, x.copy())
        elif isinstance(x, np.random.Generator):
            node = ("g", x, x.bit_generator.state)
        elif isinstance(x, np.random.RandomState):
            node = ("rs", x, x.get_state())
        elif isinstance(x, tuple):
            node = ("t", [self._walk(c, f"{path}[{i}]") for i, c in enumerate(x)])
        elif isinstance(x, (list, collections.deque, set, dict)):
            # the node goes into the memo before its children are walked, so a cycle back to it finds the same node
            kind = "d" if isinstance(x, dict) else "q" if isinstance(x, collections.deque) else "s" if isinstance(x, set) else "l"
            kids = []
            node = (kind, x, kids); self.memo[key] = node
            if kind == "d":
                kids.extend((k, self._walk(v, f"{path}[{k!r}]")) for k, v in list(x.items()))
            elif kind == "s":
                kids.extend(self._walk(c, f"{path}{{}}") for c in list(x))
            else:
                kids.extend(self._walk(c, f"{path}[{i}]") for i, c in enumerate(list(x)))
        elif hasattr(x, "__dict__"):
            node = ("o", x); self.memo[key] = node; self.paths[key] = path
            self.attrs[key] = {a: self._walk(v, f"{path}.{a}") for a, v in vars(x).items()}
        else:
            node = ("r", x)  # e.g. objects with __slots__ only; listed by tests/run_tests.py if they appear
        self.memo[key] = node
        return node

    def write_back(self):
        done = {}

        def mat(node):
            kind = node[0]
            if kind in ("v", "r"):
                return node[1]
            if kind == "t":
                return tuple(mat(c) for c in node[1])
            ref = node[1]
            if id(ref) in done:
                return ref
            done[id(ref)] = True
            if kind == "a":
                if ref.flags.writeable:
                    np.copyto(ref, node[2])
            elif kind == "g":
                ref.bit_generator.state = node[2]
            elif kind == "rs":
                ref.set_state(node[2])
            elif kind == "l":
                ref[:] = [mat(c) for c in node[2]]
            elif kind == "q":
                items = [mat(c) for c in node[2]]; ref.clear(); ref.extend(items)
            elif kind == "s":
                items = [mat(c) for c in node[2]]; ref.clear(); ref.update(items)
            elif kind == "d":
                items = [(k, mat(c)) for k, c in node[2]]; ref.clear(); ref.update(items)
            elif kind == "o":
                saved = {a: mat(c) for a, c in self.attrs[id(ref)].items()}
                d = vars(ref)
                for a in [a for a in d if a not in saved]:
                    del d[a]
                d.update(saved)
            return ref

        mat(self.root)

    def digest(self):
        """{path: hash} of every recorded value, to compare two graphs"""
        out, seen = {}, set()

        def h(node, path):
            kind = node[0]
            if kind == "v":  # sets: sorted, since their order follows the per-process string hash
                v = node[1]
                out[path] = repr(sorted(v, key=repr)) if isinstance(v, (set, frozenset)) else repr(v); return
            if kind == "r":
                out[path] = f"ref:{type(node[1]).__name__}"; return
            if kind == "t":
                for i, c in enumerate(node[1]):
                    h(c, f"{path}[{i}]")
                return
            if id(node[1]) in seen:
                return
            seen.add(id(node[1]))
            if kind == "a":
                a = node[2]
                out[path] = f"{a.dtype}{a.shape}:" + hashlib.sha1(np.ascontiguousarray(a).view(np.uint8)).hexdigest()[:16] if a.dtype != object else repr(a.tolist())
            elif kind in ("g", "rs"):
                out[path] = hashlib.sha1(repr(node[2]).encode()).hexdigest()[:16]
            elif kind in ("l", "q"):
                out[path] = f"len={len(node[2])}"
                for i, c in enumerate(node[2]):
                    h(c, f"{path}[{i}]")
            elif kind == "s":
                out[path] = f"set:{sorted(repr(c) for c in node[2])}"
            elif kind == "d":
                out[path] = f"keys={len(node[2])}"
                for k, c in node[2]:
                    h(c, f"{path}[{k!r}]")
            elif kind == "o":
                for a, c in self.attrs[id(node[1])].items():
                    h(c, f"{path}.{a}")

        h(self.root, "env")
        return out


class Snapshot:
    """everything needed to put an Episode back exactly where it was"""

    def __init__(self, ep):
        self.mjdata = copy.copy(_mjdata(ep.kenv))  # a complete, independent copy
        self.graph = _Graph(ep.env)
        self.np_random, self.py_random = np.random.get_state(), random.getstate()
        self.stream = None if _STREAM["rng"] is None else _STREAM["rng"].bit_generator.state
        self.ep = dict(t=ep.t, success=ep.success, success_step=ep.success_step, done=ep.done, obs=ep.obs,
                       n_chunks=len(ep.chunks))

    def restore(self, ep):
        _mj_copy_into(_mjmodel(ep.kenv), _mjdata(ep.kenv), self.mjdata)
        self.graph.write_back()
        np.random.set_state(self.np_random); random.setstate(self.py_random)
        if self.stream is not None:
            _STREAM["rng"] = _default_rng(0); _STREAM["rng"].bit_generator.state = self.stream
        e = self.ep
        ep.t, ep.success, ep.success_step, ep.done, ep.obs = e["t"], e["success"], e["success_step"], e["done"], e["obs"]
        del ep.chunks[e["n_chunks"]:]


# ---------------------------------------------------------------- frozen scenes
# RoboCasa's reset is not reproducible: unseeded default_rng() in fixtures, placement and object sampling, the global
# np.random in counter colours, and placement choices that differ even with equal generator states. A scene is
# therefore frozen once from an official reset and every episode loads it: the pre-compile MJCF and episode metadata
# (to build the same model), MuJoCo's integration state (then forward(), as robosuite's reset ends), and every value
# of the Python object graph, stored by attribute path (shared objects as references to their first path).

def freeze_graph(root):
    """{path: {attr: rep}} for every object reachable from root; a path is a tuple of ("a", attr) / ("i", index) /
    ("k", key) steps. reps: ("v", value), ("a", array), ("g" | "rs", rng state), ("t" | "l" | "q" | "s", [reps]),
    ("d", [(key, rep)]), ("o",) an object recorded at this path, ("@", path) the object first seen at path, ("r",)
    opaque (kept as the loading env has it)."""
    canon, objs, alive = {}, {}, []

    def rep(x, path):
        if isinstance(x, _IMMUTABLE):
            return ("v", x)
        if id(x) in canon:
            return ("@", canon[id(x)])
        if _opaque(x):
            return ("r",)
        canon[id(x)] = path; alive.append(x)
        if isinstance(x, np.ndarray):
            return ("a", x.copy())
        if isinstance(x, np.random.Generator):
            return ("g", x.bit_generator.state)
        if isinstance(x, np.random.RandomState):
            return ("rs", x.get_state())
        if isinstance(x, tuple):
            return ("t", [rep(c, path + (("i", i),)) for i, c in enumerate(x)])
        if isinstance(x, dict):
            return ("d", [(k, rep(v, path + (("k", k),))) for k, v in list(x.items())])
        if isinstance(x, (list, collections.deque)):
            return ("q" if isinstance(x, collections.deque) else "l", [rep(c, path + (("i", i),)) for i, c in enumerate(list(x))])
        if isinstance(x, set):
            return ("s", [rep(c, path) for c in list(x)])
        if hasattr(x, "__dict__"):
            attrs = objs[path] = {}  # pre-order: a parent is listed before its children
            for a, v in list(vars(x).items()):
                attrs[a] = rep(v, path + (("a", a),))
            return ("o",)
        return ("r",)

    rep(root, ())
    return objs


def _resolve(root, path):
    x = root
    for kind, key in path:
        x = getattr(x, key) if kind == "a" else x[key]
    return x


def thaw_graph(root, objs):
    """write a frozen graph into the live graph under root (an env built from the same model), in place. Objects are
    visited in recording order (parents first); every value rebuilt at a path is remembered, so a reference to that
    path gets the same object (the live env may not have it yet, e.g. a wrapper's observation queue before reset)."""
    made = {}

    def mat(r, live, path):
        out = _mat(r, live, path)
        if r[0] not in ("v", "@", "t"):
            made[path] = out
        return out

    def _mat(r, live, path):
        kind = r[0]
        if kind == "v":
            return r[1]
        if kind == "r" or kind == "o":
            return live
        if kind == "@":
            return made[r[1]] if r[1] in made else _resolve(root, r[1])
        if kind == "a":
            if isinstance(live, np.ndarray) and live.shape == r[1].shape and live.dtype == r[1].dtype and live.flags.writeable:
                np.copyto(live, r[1]); return live
            return r[1].copy()
        if kind == "g":
            g = live if isinstance(live, np.random.Generator) else np.random.default_rng(0)
            g.bit_generator.state = r[1]; return g
        if kind == "rs":
            g = live if isinstance(live, np.random.RandomState) else np.random.RandomState(0)
            g.set_state(r[1]); return g
        if kind == "t":
            return tuple(mat(c, live[i] if isinstance(live, tuple) and i < len(live) else None, path + (("i", i),)) for i, c in enumerate(r[1]))
        if kind in ("l", "q"):
            old = list(live) if isinstance(live, (list, collections.deque)) else []
            items = [mat(c, old[i] if i < len(old) else None, path + (("i", i),)) for i, c in enumerate(r[1])]
            if kind == "l" and isinstance(live, list):
                live[:] = items; return live
            if kind == "q" and isinstance(live, collections.deque):
                live.clear(); live.extend(items); return live
            return items if kind == "l" else collections.deque(items)
        if kind == "s":
            items = [mat(c, None, path) for c in r[1]]
            if isinstance(live, set):
                live.clear(); live.update(items); return live
            return set(items)
        if kind == "d":
            old = live if isinstance(live, dict) else {}
            items = [(k, mat(c, old.get(k), path + (("k", k),))) for k, c in r[1]]
            if isinstance(live, dict):
                live.clear(); live.update(items); return live
            return dict(items)
        raise ValueError(kind)

    made[()] = root
    for path in objs:  # recording order: parents first
        obj = made[path] if path in made else _resolve(root, path)
        made[path] = obj
        saved, d = objs[path], vars(obj)
        new = {a: mat(r, d.get(a), path + (("a", a),)) for a, r in saved.items()}
        for a in [a for a in d if a not in saved]:
            del d[a]
        d.update(new)


def freeze(task, seed, path=None):
    """an official reset of (task, seed) in a fresh env, frozen to path (default scene_path(task, seed)); returns
    the record (it includes the state digest and the first chunks' check data written by tests/run_tests.py)"""
    import gzip, pickle
    ep = Episode.official(task, seed)
    k = ep.kenv
    m, d = _mjmodel(k), _mjdata(k)
    rec = dict(task=task, seed=int(seed), xml=k.model.get_xml(), ep_meta=k.get_ep_meta(), mj=_MjDump(m, d),
               graph=freeze_graph(ep.env), obs=ep.obs, digest=state_digest(ep), model=model_hash(k), machine=machine())
    path = path or scene_path(task, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path + ".tmp", "wb", compresslevel=4) as f:
        pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)
    return ep, rec


def _load_scene(ep, path):
    import gzip, pickle
    with gzip.open(path, "rb") as f:
        sc = pickle.load(f)
    assert sc["task"] == ep.task and sc["seed"] == ep.seed, (path, sc["task"], sc["seed"])
    k = ep.kenv
    if model_hash(k) != sc["model"]:
        raise RuntimeError(f"{path}: the reset built a different model than the frozen scene's")
    _mj_copy_into(_mjmodel(k), _mjdata(k), sc["mj"])  # the frozen MjData exactly, derived arrays included
    thaw_graph(ep.env, sc["graph"])
    # the graph carries the freezing episode's horizon (the official one); keep this episode's (the only horizon in
    # the chain that ends episodes: robosuite's own has ignore_done=True)
    ep.env.max_episode_steps = ep.horizon
    ep.scene_digest = sc["digest"]
    return sc["obs"]


def model_hash(kenv):
    m = _mjmodel(kenv)
    buf = np.zeros(mujoco.mj_sizeModel(m), dtype=np.uint8); mujoco.mj_saveModel(m, None, buf)
    return hashlib.sha1(buf.tobytes()).hexdigest()[:16]


# MuJoCo's per-step constraint and island scratch arrays: every mj_step rebuilds them from the state; their unused
# tails keep stale values, so they are left out of comparisons (the tests check that the trajectory stays equal)
_SCRATCH = ("efc_", "island_", "dof_island", "tree_island", "map_efc2iefc", "map_iefc2efc", "iefc_", "iLD", "iM")


def mjdata_digest(kenv):
    """{field: hash} of every array (except the per-step scratch arrays), size and contact field of the live MjData"""
    d = _mjdata(kenv); out = {n: str(getattr(d, n)) for n in _SIZES + ("time",) if hasattr(d, n)}
    for f in dir(d.contact):
        if not f.startswith("_") and isinstance(getattr(d.contact, f), np.ndarray):
            out[f"contact.{f}"] = hashlib.sha1(np.ascontiguousarray(getattr(d.contact, f)).view(np.uint8)).hexdigest()[:16]
    for name in dir(d):
        if name.startswith("_") or name.startswith(_SCRATCH):
            continue
        try:
            v = getattr(d, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray):
            out[name] = hashlib.sha1(np.ascontiguousarray(v).view(np.uint8)).hexdigest()[:16]
    return out


def is_image_key(key):
    """camera images held in the Python state (observable caches, the wrapper's observation queue and render cache)"""
    return "_image" in key or "video." in key or key.endswith(".render_cache")


def state_digest(ep, images=True):
    """{key: hash} of the whole state of an episode: MjData fields and the Python graph. images=False leaves out the
    camera images, which can differ by render noise (MSAA) while everything else is equal"""
    out = {f"mj.{k}": v for k, v in mjdata_digest(ep.kenv).items()}
    out.update(_Graph(ep.env).digest())
    return out if images else {k: v for k, v in out.items() if not is_image_key(k)}


# ---------------------------------------------------------------- records

def obs_digest(obs):
    """short hash per observation key (images and state), for checking that a replay sees the same inputs"""
    out = {}
    for k, v in obs.items():
        b = v.encode() if isinstance(v, str) else np.ascontiguousarray(v).view(np.uint8).tobytes()
        out[k] = hashlib.sha1(b).hexdigest()[:12]
    return out


_MACHINE = None


def machine():
    """where a result was produced: host, GPU used for rendering, CPU, BLAS kernel, library versions"""
    global _MACHINE
    if _MACHINE is None:
        def sh(cmd):
            try:
                return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
            except Exception:
                return None
        egl = os.environ.get("MUJOCO_EGL_DEVICE_ID") or os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu = sh(f"nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader -i {egl.split(',')[0]}") if egl else None
        try:
            from threadpoolctl import threadpool_info
            blas = [(i.get("internal_api"), i.get("architecture")) for i in threadpool_info()]
        except Exception:
            blas = None
        import robosuite
        _MACHINE = dict(host=socket.gethostname(), render_gpu=gpu, cpu=platform.processor() or sh("grep -m1 'model name' /proc/cpuinfo"),
                        blas=blas, openblas_coretype=os.environ.get("OPENBLAS_CORETYPE"), mujoco=mujoco.__version__,
                        robosuite=robosuite.__version__, robocasa=getattr(robocasa, "__version__", None),
                        repo=sh(f"git -C {ROOT} rev-parse --short HEAD"), cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
                        egl_device=egl)
    return _MACHINE
