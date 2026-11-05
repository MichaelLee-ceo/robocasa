import json, sys
sys.path.insert(0, "/home/michaellee/mclee/robocasa")
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
H = json.load(open("/home/michaellee/mclee/robocasa/runs/horizons_original_plus.json"))
for t, h in H.items():
    print(t, h, get_task_horizon(t), "OK" if h == get_task_horizon(t) else "DIFF")
print({k: v for k, v in TASK_SET_REGISTRY.items() if "seen" in k.lower() or "composite" in k.lower()})
