"""GR00T N1.5 policy server with reproducible action sampling.

The policy is built by third_party/Isaac-GR00T/scripts/run_eval.py's own run_server() (same data config, embodiment tag,
denoising_steps=4 as `run_eval.py --server`); only the server class is swapped. Added on top:
  * deterministic torch settings: cuDNN deterministic / no benchmark, torch.use_deterministic_algorithms(True),
    CUBLAS_WORKSPACE_CONFIG (set before CUDA initialises);
  * a get_action request may carry "_seed" (int). It is removed from the observation, python / numpy / torch (all CUDA
    devices) are seeded with it right before get_action, and the response carries "_seed_echo" so the client can
    assert that a seeded server handled the request. The ZeroMQ REP loop serves one request at a time, so concurrent
    workers cannot interleave between seeding and sampling. Requests without "_seed" are served unseeded, as before.
usage (cwd third_party/Isaac-GR00T, like run_eval.py):
  python $ROBOCASA_ROOT/runs/scripts/serve_seeded.py --model_path <checkpoint dir> --port 5581
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, random, sys
import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

ROOT = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa"))
sys.path.insert(0, f"{ROOT}/third_party/Isaac-GR00T")
sys.path.insert(0, f"{ROOT}/third_party/Isaac-GR00T/scripts")
import run_eval  # noqa: E402
from gr00t.eval.robot import RobotInferenceServer  # noqa: E402


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % 2 ** 32); torch.manual_seed(seed)  # torch.manual_seed seeds every CUDA device too


class SeededServer(RobotInferenceServer):
    def __init__(self, model, host="*", port=5555, api_token=None):
        super().__init__(model, host=host, port=port, api_token=api_token)
        self.n_seeded, self.n_unseeded = 0, 0

        def get_action(obs):
            obs = dict(obs)
            seed = obs.pop("_seed", None)
            if seed is None:
                self.n_unseeded += 1
                if self.n_unseeded == 1:
                    print("WARNING: get_action request without _seed, served unseeded", flush=True)
                return model.get_action(obs)
            seed = int(seed)
            seed_all(seed)
            self.n_seeded += 1
            out = model.get_action(obs)
            out["_seed_echo"] = seed
            return out
        self.register_endpoint("get_action", get_action)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--data_config", default="panda_omron")  # run_eval.py defaults
    ap.add_argument("--embodiment_tag", default="new_embodiment")
    a = ap.parse_args()
    print(f"serve_seeded: torch {torch.__version__}, cudnn.deterministic={torch.backends.cudnn.deterministic}, "
          f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}, "
          f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}", flush=True)
    run_eval.RobotInferenceServer = SeededServer  # run_server() looks the class up in its module globals
    run_eval.run_server(data_config=a.data_config, model_path=a.model_path, embodiment_tag=a.embodiment_tag, port=a.port)
