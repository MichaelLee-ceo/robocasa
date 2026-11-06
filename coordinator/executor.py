"""The executor: the frozen GR00T N1.5 policy (served by envkit/servers.sh). It sees only its own observation
(3 views, proprio) and a sentence, never the coordinator's memory.

act(ep_obs, memory, chunk_index): one 16-step chunk. The sentence is memory.gr00t_instruction (None: the task
sentence); the noise is policy.chunk_seed(f"{tag}/{memory.stream}", 0, chunk index), exactly as
runs/20261105_recovery/loop.py seeds it (tag "loop/<point>/p<seed>" for failure-point runs, for every config: the same
noise for the same attempt in every config, as loop.py had for every arm). "Fresh samples" = a new stream name; a rerun
with the same names repeats the same noise (images can still differ by MSAA render noise, envkit/README.md)."""
from .paths import use_envkit

use_envkit()
import policy  # noqa: E402


class Executor:
    def __init__(self, port, tag):
        self.client, self.tag = policy.Client(port), tag

    def act(self, ep_obs, memory, chunk_index):
        seed = policy.chunk_seed(f"{self.tag}/{memory.stream}", 0, chunk_index)
        return policy.act(self.client, ep_obs, seed, instruction=memory.gr00t_instruction)
