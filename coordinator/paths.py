"""Where things are. The coordinator and the environment layer (envkit) live at the repository's top level; the
recorded failure points and the experiment outputs (records) live under runs/."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
ENVKIT = os.path.join(ROOT, "envkit")                       # the verified environment layer (episodes, GR00T client)
RECOVERY = os.path.join(RUNS, "20261105_recovery")          # phase 0 runs and failure points, the recovery study
POINTS = os.path.join(RECOVERY, "phase0", "points.json")
RECORDS = os.path.join(RECOVERY, "phase0", "records")
PHASE0_PLAIN = os.path.join(RECOVERY, "phase0", "plain.jsonl")
OUT = os.path.join(RUNS, "20261106_coordinator")            # default output folder of scripts/run.py
ENV_FILE = os.path.join(ROOT, ".env")                       # git-ignored; API keys (never printed)
GROOT_PY = os.path.join(ROOT, "envs", "groot", "bin", "python")


def use_envkit():
    """make envkit's modules (envkit, stages, events, policy) importable; they import each other by plain names"""
    if ENVKIT not in sys.path:
        sys.path.insert(0, ENVKIT)
