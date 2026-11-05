"""The GR00T N1.5 policy as envkit episodes use it (2026-11-05).

Client(port)       a client of serve_seeded.py (the official run_eval.py server with seeded action sampling)
chunk_seed(tag, policy_seed, ci)
                   the seed of the action noise of chunk ci: a stable hash, so a rollout is reproducible given its images
act(client, obs, seed)
                   one chunk for an observation of envkit.Episode (the official client's request: the MultiStepWrapper
                   observation with a batch dimension); returns {action key: (16, d)}
rollout(ep, client, tag, policy_seed=0, on_chunk=None, max_chunks=None)
                   run the policy until the episode ends (the horizon; success does not stop it, as in the official
                   eval). on_chunk(ep) is called after every chunk. Every executed action is recorded by envkit.
Exactness: the physics and the checks replay bit for bit from the recorded actions (envkit.replay). Re-running the
policy live (also after a restore) can take another path: the official rendering (MSAA 4) changes a few edge pixels by
+-1 at random on any render, even in one process. Success is judged on the physics, so this acts like extra policy
noise; an exact rerun is a replay of the recorded actions.
Servers: runs/envkit/servers.sh."""
import hashlib, os, sys
import numpy as np

ROOT = os.path.abspath(os.environ.get("ROBOCASA_ROOT")  # the repository
                       or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{ROOT}/third_party/Isaac-GR00T")
from gr00t.eval.service import BaseInferenceClient  # noqa: E402

CHECKPOINT = f"{ROOT}/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"


class Client(BaseInferenceClient):
    def __init__(self, port, host="localhost", timeout_ms=600000):
        super().__init__(host=host, port=port, timeout_ms=timeout_ms)

    def get_action(self, obs):
        return self.call_endpoint("get_action", obs)


def chunk_seed(tag, policy_seed, ci):
    return int.from_bytes(hashlib.sha256(f"{tag}|{policy_seed}|{ci}".encode()).digest()[:8], "little")


def act(client, obs, seed):
    req = {k: ((v,) if isinstance(v, str) else np.asarray(v)[None]) for k, v in obs.items()}
    req["_seed"] = int(seed)
    out = client.get_action(req)
    assert out.pop("_seed_echo", None) == int(seed), "the server did not seed this request (not serve_seeded.py?)"
    return {k: np.asarray(v)[0] for k, v in out.items()}


def rollout(ep, client, tag, policy_seed=0, on_chunk=None, max_chunks=None):
    ci = len(ep.chunks)
    while not ep.done and (max_chunks is None or len(ep.chunks) - ci < max_chunks):
        ep.step(act(client, ep.obs, chunk_seed(tag, policy_seed, len(ep.chunks))))
        if on_chunk is not None:
            on_chunk(ep)
    return ep
