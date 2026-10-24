"""Sanity check of the Atomic-Seen checkpoint on its own official atomic tasks (official client, split=target)."""
import sys, os
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/third_party/Isaac-GR00T/scripts"))
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/third_party/Isaac-GR00T"))
import run_eval
import robocasa.utils.dataset_registry as R
R.TASK_SET_REGISTRY["sanity"] = sys.argv[1].split(",")
run_eval.TASK_SET_REGISTRY = R.TASK_SET_REGISTRY
run_eval.run_client("localhost", 5556, ["sanity"], os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/runs/20261013_pretrain_sanity"),
                    "target", int(sys.argv[2]), 1, 16)
